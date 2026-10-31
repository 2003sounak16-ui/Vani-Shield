"""
VaniShield AI — Per-Call Streaming Session State

The REST endpoints in `app.py` are one-shot: one clip in, one verdict out. A
live call is not. It arrives as a stream of overlapping 3.5s windows at roughly
four packets a second, and every window is evaluated independently by
`vanishield_engine.evaluate_live_call`. That makes each frame stateless and
therefore forgetful: a caller who establishes authority in the first ten
seconds and asks for an OTP two minutes later looks, frame by frame, like two
unrelated harmless calls.

This module holds the only mutable per-call state there is:

1. A smoothed acoustic score (exponential moving average), because a single
   1.5s window of line noise is not evidence and must not be allowed to flip
   the dashboard on its own.
2. The accumulated set of social-engineering stages seen so far in the call.
   Stages are monotonic for the lifetime of the call: evidence observed is
   evidence, and a UI that drops the AUTHORITY chip once the caller stops
   talking about the bank throws away the entire point of staging.
3. An append-only, bounded event log — one line per transition actually worth
   showing a human, not one line per packet.

Privacy posture, inherited from `ephemeral_buffer` (DPDP Act 2023):
  * No disk writes of any kind from this module.
  * No raw audio is retained; only the scalar scores derived from it.
  * At most the single latest transcript is held, never a history of it, and
    it is kept in a mutable byte buffer that is zeroed on `end()` — the same
    "overwrite with zeros on purge" discipline the RAM ring buffer applies to
    PCM chunks.
  * The event log carries stage labels and risk bands only: no transcript text
    and no caller number.

Thread safety: FastAPI dispatches `evaluate_live_call` into a
ThreadPoolExecutor, so two frames of the same call can land in `update()`
concurrently. Every mutation here is guarded by a per-session lock, and the
registry guards its own map.

Language: the timeline is shown to the person on the call, so every event line
is written in the call's language (en / hi / bn) at the moment it is logged.
The session's language can change mid-call — a call that opens with an English
greeting and switches to Bengali once the ASR has more audio is the common
case — so `set_lang` exists and each event records the language it was written
in.
"""

from __future__ import annotations

import logging
import threading
import time
from collections import deque
from dataclasses import dataclass
from itertools import islice
from typing import Dict, Iterable, List, Optional, Sequence

from scam_stages import (
    ACOUSTIC_SYNTHETIC_THRESHOLD,
    BAND_AMBER,
    BAND_GREEN,
    BAND_RED,
    DEFAULT_LANG,
    STAGE_ORDER,
    event_text,
    normalize_lang,
    stage_label,
)

logger = logging.getLogger(__name__)

# Call lifecycle states. `app.py` and the dashboard switch on these strings,
# so they are part of the contract.
CALL_STATE_LIVE = "LIVE"
CALL_STATE_ENDED = "ENDED"

# Event kinds, roughly in the order they appear in a real vishing call.
KIND_CALL = "call"
KIND_STAGE = "stage"
KIND_ACOUSTIC = "acoustic"
KIND_RISK = "risk"
KIND_BREAKER = "breaker"
KIND_PAYMENT = "payment"
KIND_ERROR = "error"

SEV_INFO = "info"
SEV_ACCENT = "accent"
SEV_SUCCESS = "success"
SEV_WARNING = "warning"
SEV_DANGER = "danger"

_BAND_SEVERITY: Dict[str, str] = {
    BAND_RED: SEV_DANGER,
    BAND_AMBER: SEV_WARNING,
    BAND_GREEN: SEV_SUCCESS,
}

# Band -> event-string key. The transition variants, not the `risk_state_*`
# ones: those exist for the one-shot REST path, which has no previous frame to
# compare a band against. A live call always does.
_RISK_KEY: Dict[str, str] = {
    BAND_RED: "risk_red",
    BAND_AMBER: "risk_amber",
    BAND_GREEN: "risk_green",
}

# Upper bound on the retained transcript. A live call's rolling transcript is
# bounded by the 3.5s audio window in practice; this is a belt-and-braces cap
# so a misbehaving client cannot grow per-session memory without limit.
MAX_TRANSCRIPT_CHARS = 4000


# ---------------------------------------------------------------------------
# Events
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CallEvent:
    """
    One human-readable line in the call timeline.

    `lang` is the language the line was written in, which is the session's
    language at the moment it was logged — not necessarily the session's
    current one. A client that renders a mixed log can therefore still mark
    each line correctly.
    """

    t: float          # seconds since call start, rounded to 0.1
    kind: str         # call | stage | acoustic | risk | breaker | payment | error
    severity: str     # info | accent | success | warning | danger
    text: str
    lang: str = DEFAULT_LANG

    def as_dict(self) -> dict:
        return {
            "t": self.t,
            "kind": self.kind,
            "severity": self.severity,
            "text": self.text,
            "lang": self.lang,
        }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, value))


def _as_unit_prob(value: object) -> Optional[float]:
    """
    Coerces one window score to a 0..1 probability.

    `vanishield_engine` reports `window_probs` in *percent* (0..100) while this
    module's contract is 0..1, and the two are easy to confuse at the call
    site. Anything above 1.0 is therefore read as a percentage rather than
    silently clamped to a certain-synthetic 1.0, which would promote a 42%
    window to a maximum-confidence alert.
    """
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if number != number:  # NaN
        return None
    if number > 1.0:
        number = number / 100.0
    return _clamp(number)


def _mask_number(phone_number: str) -> str:
    """Keeps the last four digits only, so logs and timelines carry no PII."""
    digits = [c for c in str(phone_number) if c.isdigit()]
    if len(digits) <= 4:
        return "".join(digits) or "unknown number"
    return "..." + "".join(digits[-4:])


# ---------------------------------------------------------------------------
# Session
# ---------------------------------------------------------------------------


class CallSession:
    """
    Mutable state for one live call.

    All public methods are safe to call from several threads. Nothing here
    touches the filesystem.
    """

    def __init__(
        self,
        session_id: str,
        phone_number: str = "",
        ema_alpha: float = 0.5,
        max_events: int = 200,
        lang: str = DEFAULT_LANG,
    ) -> None:
        self.session_id = str(session_id)
        self.phone_number = str(phone_number or "")
        self.ema_alpha = _clamp(float(ema_alpha), 0.01, 1.0)
        self.max_events = max(1, int(max_events))
        self._lang = normalize_lang(lang)

        self._lock = threading.Lock()
        self._started_monotonic = time.monotonic()
        self._last_activity = self._started_monotonic
        self._ended_monotonic: Optional[float] = None
        self._call_state = CALL_STATE_LIVE

        # Acoustic smoothing. None until the first window seeds it, so the
        # first real score is adopted outright instead of being halved against
        # an imaginary zero.
        self._ema: Optional[float] = None
        self._windows_seen = 0

        self._stages: set = set()
        self._latest_transcript = bytearray()

        # Latches for the emit-once transitions.
        self._acoustic_flagged = False
        self._payment_flagged = False
        self._breaker_engaged = False
        self._band = BAND_GREEN
        self._high_sensitivity_credential = False

        self.last_risk_score = 0.0
        self.last_latency_ms: Optional[float] = None

        self._events: deque = deque(maxlen=self.max_events)
        self._total_events = 0

        self._line_silent = False

        self._append_key_locked(KIND_CALL, SEV_INFO, "call_started")
        logger.info(
            "Call session %s opened (lang=%s).", self.session_id, self._lang
        )

    # -- read-only views ---------------------------------------------------

    @property
    def elapsed_sec(self) -> int:
        with self._lock:
            return int(self._elapsed_locked())

    @property
    def stages(self) -> list:
        with self._lock:
            return [s for s in STAGE_ORDER if s in self._stages]

    @property
    def acoustic_ema(self) -> float:
        with self._lock:
            return round(self._ema if self._ema is not None else 0.0, 4)

    @property
    def call_state(self) -> str:
        with self._lock:
            return self._call_state

    @property
    def lang(self) -> str:
        """The language new event lines will be written in: en | hi | bn."""
        with self._lock:
            return self._lang

    @property
    def latest_transcript(self) -> str:
        """The most recent window's transcript only. There is no history."""
        with self._lock:
            return self._latest_transcript.decode("utf-8", errors="replace")

    @property
    def idle_sec(self) -> float:
        with self._lock:
            return max(0.0, time.monotonic() - self._last_activity)

    # -- mutation ----------------------------------------------------------

    def set_caller(self, phone_number: str) -> None:
        """
        Records the caller identity for this call.

        The number is held in volatile memory only, is never written to disk,
        and is masked in both the log line and the timeline event; the full
        value exists solely so `evaluate_live_call` can run its telecom lookup.
        """
        cleaned = str(phone_number or "").strip()
        with self._lock:
            if not cleaned or cleaned == self.phone_number:
                return
            self.phone_number = cleaned
            self._last_activity = time.monotonic()
            self._append_locked(
                KIND_CALL, SEV_INFO, "Caller shown as " + _mask_number(cleaned) + "."
            )
        logger.info(
            "Call session %s caller set to %s.", self.session_id, _mask_number(cleaned)
        )

    def set_lang(self, lang: str) -> None:
        """
        Switches the language future event lines are written in.

        The ASR refines its language guess as the call goes on, and a call that
        opens with an English greeting and switches to Bengali is the common
        case rather than an edge case, so this is expected to be called
        mid-call.

        Decision: lines already in the log keep the language they were written
        in. Rewriting the visible history under the user mid-call is worse than
        a log with two languages in it, and the alternative — storing a
        template id plus its fields per event and re-rendering on every poll —
        buys nothing here. Each event carries its own `lang` so a client can
        label the mixed lines correctly.
        """
        normalised = normalize_lang(lang)
        with self._lock:
            if normalised == self._lang:
                return
            self._lang = normalised
        logger.info("Call session %s language set to %s.", self.session_id, normalised)

    def update(
        self,
        *,
        window_probs: Optional[Iterable] = None,
        stages: Optional[Sequence[str]] = None,
        payment_attempt: bool = False,
        risk_score: float = 0.0,
        band: str = BAND_GREEN,
        latency_ms: Optional[float] = None,
        transcript: str = "",
        circuit_breaker_engaged: bool = False,
        high_sensitivity_credential: bool = False,
        line_silent: bool = False,
    ) -> dict:
        """
        Folds one evaluated frame into the call's state.

        Emits timeline events only for genuine transitions. At a 250ms packet
        rate a per-frame log would produce four lines a second and bury the
        five moments that actually matter, so each of the following fires at
        most once per call unless the underlying condition really changes
        again:

          * a stage seen for the first time in this call,
          * the smoothed acoustic score first crossing the synthetic
            threshold,
          * a risk band change, in either direction,
          * a payment attempt,
          * the banking circuit breaker engaging,
          * the line falling silent (`line_silent`, from the engine's
            STANDBY_SILENCE frame), re-emitted only when silence returns after
            speech.
        """
        emitted: List[CallEvent] = []

        with self._lock:
            now = time.monotonic()
            self._last_activity = now

            if self._call_state == CALL_STATE_ENDED:
                # A late frame from the executor, after the client hung up.
                # Dropped rather than resurrecting purged state.
                logger.debug(
                    "Ignoring post-end update for call session %s.", self.session_id
                )
                return self._frame_locked([])

            # 1. Acoustic EMA. The first window seeds the average.
            for raw in list(window_probs or []):
                prob = _as_unit_prob(raw)
                if prob is None:
                    continue
                self._windows_seen += 1
                if self._ema is None:
                    self._ema = prob
                else:
                    self._ema = (
                        self.ema_alpha * prob + (1.0 - self.ema_alpha) * self._ema
                    )

            # 2. Stages accumulate and are never removed.
            if high_sensitivity_credential:
                self._high_sensitivity_credential = True
            incoming = {s for s in (stages or []) if s in STAGE_ORDER}
            for stage in STAGE_ORDER:
                if stage in incoming and stage not in self._stages:
                    self._stages.add(stage)
                    emitted.append(
                        self._append_key_locked(
                            KIND_STAGE,
                            SEV_DANGER,
                            "stage_detected",
                            label=stage_label(stage, self._lang),
                        )
                    )

            # 3. First crossing of the synthetic-voice threshold.
            if (
                not self._acoustic_flagged
                and self._ema is not None
                and self._ema >= ACOUSTIC_SYNTHETIC_THRESHOLD
            ):
                self._acoustic_flagged = True
                emitted.append(
                    self._append_key_locked(
                        KIND_ACOUSTIC,
                        SEV_DANGER,
                        "synthetic",
                        pct=round(self._ema * 100.0, 1),
                    )
                )

            # 4. Risk band movement, in either direction.
            try:
                self.last_risk_score = float(risk_score or 0.0)
            except (TypeError, ValueError):
                self.last_risk_score = 0.0
            new_band = band if band in _BAND_SEVERITY else self._band
            if new_band != self._band:
                self._band = new_band
                emitted.append(
                    self._append_key_locked(
                        KIND_RISK,
                        _BAND_SEVERITY[new_band],
                        _RISK_KEY[new_band],
                        score=round(self.last_risk_score, 1),
                    )
                )

            # 5. Payment attempt — once is enough; the UI keeps the chip.
            if payment_attempt and not self._payment_flagged:
                self._payment_flagged = True
                emitted.append(
                    self._append_key_locked(KIND_PAYMENT, SEV_WARNING, "payment")
                )

            # 6. Circuit breaker. Re-emitted only if it genuinely re-engages
            # after having been released.
            breaker = bool(circuit_breaker_engaged)
            if breaker and not self._breaker_engaged:
                emitted.append(
                    self._append_key_locked(KIND_BREAKER, SEV_DANGER, "breaker")
                )
            self._breaker_engaged = breaker

            # 7. Line silence. One line per silent stretch, not per packet.
            silent = bool(line_silent)
            if silent and not self._line_silent:
                emitted.append(
                    self._append_key_locked(KIND_CALL, SEV_INFO, "mic_silent")
                )
            self._line_silent = silent

            # 8. Latest transcript only. Overwritten in place, never appended.
            if transcript:
                self._set_transcript_locked(str(transcript))

            if latency_ms is not None:
                try:
                    self.last_latency_ms = round(float(latency_ms), 2)
                except (TypeError, ValueError):
                    self.last_latency_ms = None

            return self._frame_locked(emitted)

    def log_error(self, message: str) -> dict:
        """
        Records a frame-level failure (audio decode error, engine exception) on
        the timeline. Deliberately outside the emit-once latch set: a recurring
        failure is worth seeing every time it happens.

        The message is passed through verbatim and is therefore the one event
        kind that is *not* localised: there is no agreed string for it, and a
        machine-translated stack-trace summary would be worse than an English
        one. Callers that want a localised line should use an agreed key
        instead of this method.
        """
        with self._lock:
            self._last_activity = time.monotonic()
            event = self._append_locked(
                KIND_ERROR, SEV_WARNING, str(message or "Analysis frame failed.")
            )
            return self._frame_locked([event])

    def end(self, reason: str = "ENDED") -> dict:
        """
        Closes the call and purges what this session holds.

        The transcript buffer is zeroed before release, and the caller number,
        stage set and smoothed score are cleared, mirroring
        `EphemeralRAMBuffer.purge_session`. The timeline itself survives for
        the client's final poll: it holds stage labels and risk bands only,
        with no transcript text and no caller number in it.
        """
        with self._lock:
            if self._call_state == CALL_STATE_ENDED:
                return self._frame_locked([])

            stage_count = len(self._stages)
            event = self._append_key_locked(
                KIND_CALL,
                SEV_DANGER if stage_count else SEV_SUCCESS,
                "call_ended",
            )

            self._ended_monotonic = time.monotonic()
            self._call_state = CALL_STATE_ENDED
            self._last_activity = self._ended_monotonic

            # Zero and clear. Nothing derived from the caller's voice, words or
            # number outlives the call.
            self._zero_transcript_locked()
            self.phone_number = ""
            self._stages.clear()
            self._ema = 0.0
            self._windows_seen = 0
            self._acoustic_flagged = False
            self._payment_flagged = False
            self._breaker_engaged = False
            self._line_silent = False
            self._high_sensitivity_credential = False
            self.last_risk_score = 0.0

            frame = self._frame_locked([event])

        logger.info("Call session %s ended (%s). State purged.", self.session_id, reason)
        return frame

    # -- timeline ----------------------------------------------------------

    def events_since(self, cursor: int = 0) -> tuple:
        """
        Returns `(events, new_cursor)` for everything logged after `cursor`.

        The cursor counts events *ever* appended, not positions in the bounded
        deque, so it stays meaningful after truncation. A cursor pointing at an
        event that has since been dropped returns the oldest survivors rather
        than raising or quietly returning the wrong slice — the client loses
        lines it was never going to see anyway, but never loses its place.
        """
        with self._lock:
            total = self._total_events
            dropped = total - len(self._events)
            try:
                start = int(cursor)
            except (TypeError, ValueError):
                start = 0
            start = max(0, min(start, total))
            start = max(start, dropped)
            offset = start - dropped
            return [e.as_dict() for e in islice(self._events, offset, None)], total

    def as_frame_fields(self) -> dict:
        """The session-owned fields merged into every outbound frame."""
        with self._lock:
            return {
                "session_id": self.session_id,
                "elapsed_sec": int(self._elapsed_locked()),
                "call_state": self._call_state,
                "acoustic_ema": round(self._ema if self._ema is not None else 0.0, 4),
                "stages": [s for s in STAGE_ORDER if s in self._stages],
                "lang": self._lang,
            }

    # -- internals (callers must hold self._lock) --------------------------

    def _elapsed_locked(self) -> float:
        end = (
            self._ended_monotonic
            if self._ended_monotonic is not None
            else time.monotonic()
        )
        return max(0.0, end - self._started_monotonic)

    def _append_locked(self, kind: str, severity: str, text: str) -> CallEvent:
        event = CallEvent(
            t=round(self._elapsed_locked(), 1),
            kind=kind,
            severity=severity,
            text=text,
            lang=self._lang,
        )
        self._events.append(event)   # bounded deque drops the oldest for us
        self._total_events += 1
        return event

    def _append_key_locked(
        self, kind: str, severity: str, key: str, **fields
    ) -> CallEvent:
        """
        Appends one line from the shared `scam_stages.EVENT_STRINGS` table in
        the session's current language. The table lives there, not here,
        because the one-shot REST path renders the same lines and two copies of
        a translation table would drift apart.
        """
        return self._append_locked(
            kind, severity, event_text(key, self._lang, **fields)
        )

    def _frame_locked(self, emitted: Sequence[CallEvent]) -> dict:
        return {
            "new_events": [e.as_dict() for e in emitted],
            "event_cursor": self._total_events,
            "acoustic_ema": round(self._ema if self._ema is not None else 0.0, 4),
            "stages": [s for s in STAGE_ORDER if s in self._stages],
            "elapsed_sec": int(self._elapsed_locked()),
            "call_state": self._call_state,
        }

    def _set_transcript_locked(self, transcript: str) -> None:
        self._zero_transcript_locked()
        self._latest_transcript = bytearray(
            transcript[:MAX_TRANSCRIPT_CHARS].encode("utf-8")
        )

    def _zero_transcript_locked(self) -> None:
        buf = self._latest_transcript
        for i in range(len(buf)):
            buf[i] = 0
        del buf[:]


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


class SessionRegistry:
    """
    The live-call table. One entry per WebSocket connection.

    Guarded by its own lock so that `get_or_create` is genuinely
    once-per-session: two frames arriving together must not produce two
    sessions, which would split the accumulated stage set across both.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._sessions: Dict[str, CallSession] = {}

    def get_or_create(
        self, session_id: str, phone_number: str = "", lang: str = DEFAULT_LANG
    ) -> CallSession:
        """
        Returns the session for `session_id`, creating it if this is the first
        frame of the call.

        `phone_number` and `lang` are applied to an existing session too, so
        passing them is equivalent to `set_caller` / `set_lang`. That matters
        in the per-frame path: pass the language the call is *currently* in,
        not a hardcoded default, or every frame will drag a Bengali call back
        to English. `get()` is the right call when there is nothing to update.
        """
        key = str(session_id)
        with self._lock:
            session = self._sessions.get(key)
            created = session is None
            if created:
                session = CallSession(key, phone_number=phone_number, lang=lang)
                self._sessions[key] = session
        if not created:
            # Set outside the registry lock: these take the session lock.
            if phone_number:
                session.set_caller(phone_number)
            if lang:
                session.set_lang(lang)
        return session

    def get(self, session_id: str):
        with self._lock:
            return self._sessions.get(str(session_id))

    def drop(self, session_id: str) -> None:
        """Ends and forgets a session, zeroing what it held."""
        key = str(session_id)
        with self._lock:
            session = self._sessions.pop(key, None)
        if session is not None:
            session.end(reason="PURGED")

    def reap_idle(self, max_idle_sec: float = 300.0) -> int:
        """
        Drops sessions that have had no frames for `max_idle_sec` and returns
        how many were reaped. A dropped WebSocket does not always raise, so
        without this a half-open connection would pin its state in memory for
        the lifetime of the process.

        The comparison is inclusive, so `reap_idle(0.0)` reaps everything —
        useful on shutdown, and not at the mercy of the ~15ms resolution of
        the Windows monotonic clock, which can report a just-created session
        as having been idle for exactly zero seconds.
        """
        cutoff = max(0.0, float(max_idle_sec))
        with self._lock:
            stale = [
                key
                for key, session in self._sessions.items()
                if session.idle_sec >= cutoff
            ]
        reaped = 0
        for key in stale:
            with self._lock:
                session = self._sessions.pop(key, None)
            if session is not None:
                session.end(reason="IDLE_TIMEOUT")
                reaped += 1
        if reaped:
            logger.info("Reaped %d idle call session(s).", reaped)
        return reaped

    def active_count(self) -> int:
        with self._lock:
            return len(self._sessions)


# Singleton instance, mirroring `ram_ring_buffer` in ephemeral_buffer.
registry = SessionRegistry()
