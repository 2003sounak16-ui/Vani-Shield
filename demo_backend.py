"""
VaniShield AI — Explicitly Labelled Demo Stand-In for the Acoustic Pillar

WARNING — READ THIS BEFORE USING ANYTHING IN THIS MODULE.
`DemoAcousticScorer` IS NOT A VOICE-CLONE DETECTOR. It is an untrained
heuristic that reads four cheap signal statistics off each audio window and
maps them onto a 0..1 number shaped like a probability. Its output is NOT a
detection result, carries no accuracy claim whatsoever, and must never be
presented to a user, a pilot partner or an evaluator as evidence that a voice
is synthetic or genuine. It exists for exactly one reason: so the server, the
WebSocket stream, the stage classifier, the session timeline and the dashboard
can be run and reviewed on a machine that has no fine-tuned Wav2Vec 2.0
checkpoint — today `VaniShieldCore.__init__` raises in that situation,
`core_engine` is None, and every new feature is therefore unverifiable and the
product undemonstrable.

Design constraints that follow from that purpose:

* Off by default, and impossible to enable by accident. Nothing here activates
  unless `VANISHIELD_DEMO` is explicitly set to a truthy value, so a
  production deployment that simply forgets about this file gets the real
  engine or a hard failure — never a stub quietly scoring live calls.
* Deterministic. Every score is derived from the window's own samples
  (spectral flatness, zero-crossing rate, RMS, and a hash of the rounded
  samples), so the same clip always scores the same and a demo or test is
  reproducible. There is no use of `random`.
* Honest in its own metadata. `model_name` and `describe()` both say in plain
  words that this is a stub, so any log line, health payload or UI badge that
  renders them inherits the warning.
* numpy and the standard library only. No torch, librosa, transformers or
  speechbrain — being importable on a machine with none of the heavy
  dependencies installed is the entire point of the module.
* No disk writes.

The cues are the same ones the trained model is known to key on (see
`extract_prosodic_dsp` and the vocoder-physics checks in
`vanishield_engine.evaluate_live_call`), so the shape of the curve across
windows is plausible for a demo sparkline. Plausible shape is not accuracy.
"""

from __future__ import annotations

import hashlib
import logging
import os
from typing import Iterable, List, Optional, Sequence

import numpy as np

logger = logging.getLogger(__name__)

DEMO_ENV_VAR = "VANISHIELD_DEMO"

_TRUTHY = frozenset({"1", "true", "yes", "on"})

DEMO_BANNER = (
    "DEMO MODE ACTIVE: acoustic scores come from an untrained heuristic stub, "
    "not a trained voice-clone detector. Output is NOT a detection result."
)

# Framing for the cue statistics. 1024 samples at 16kHz is 64ms, matching the
# frame length `extract_prosodic_dsp` uses, so the numbers are comparable to
# the ones the real DSP pillar reports.
_FRAME_LEN = 1024
_HOP_LEN = 512
_EPS = 1e-10

# Cue reference points, taken from the thresholds the real pipeline already
# uses: spectral flatness around 0.012 is where the engine starts calling the
# harmonic structure vocoder-like, and natural speech ZCR sits near 0.10.
_FLATNESS_FULL_SCALE = 0.020
_ZCR_FLOOR = 0.05
_ZCR_SPAN = 0.25
_RMS_FULL_SCALE = 0.10

# Voice-activity gate, copied from the silence check in `evaluate_live_call`
# so the stub agrees with the real pipeline about what counts as speech. A
# window with no speech in it is scored down hard rather than being allowed to
# ride a noise-floor statistic: an all-zero buffer has a spectral flatness of
# 1.0, which without this gate would read as maximally vocoder-like.
_VAD_RMS = 0.005
_VAD_PEAK = 0.015
_SILENT_SCALE = 0.15

# How far the content hash is allowed to move a score. Enough to keep a
# sparkline from flat-lining across near-identical windows, far too little to
# invent a verdict.
_JITTER_RANGE = 0.06

_SCORE_FLOOR = 0.03
_SCORE_SPAN = 0.88


def demo_enabled() -> bool:
    """
    True only when `VANISHIELD_DEMO` is explicitly set to 1 / true / yes / on.

    Any other value, including an empty string, an unset variable or "0",
    leaves the stub off. Enabling the stub is a deliberate act.
    """
    raw = os.environ.get(DEMO_ENV_VAR)
    if raw is None:
        return False
    return raw.strip().lower() in _TRUTHY


def _as_mono_float(clip) -> np.ndarray:
    """Coerces one window to finite 1-D float32 without copying when possible."""
    arr = np.asarray(clip, dtype=np.float32)
    if arr.ndim > 1:
        arr = arr.reshape(arr.shape[0], -1).mean(axis=1) if arr.shape[0] else arr.ravel()
    arr = np.ravel(arr)
    if arr.size and not np.all(np.isfinite(arr)):
        arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
    return arr


def _frames(arr: np.ndarray) -> List[np.ndarray]:
    if arr.size < _FRAME_LEN:
        return [arr] if arr.size else []
    return [
        arr[start: start + _FRAME_LEN]
        for start in range(0, arr.size - _FRAME_LEN + 1, _HOP_LEN)
    ]


def _spectral_flatness(frames: Sequence[np.ndarray]) -> float:
    """
    Geometric mean over arithmetic mean of the power spectrum, averaged across
    frames. This is the same quantity `librosa.feature.spectral_flatness`
    reports, computed with numpy so the module stays dependency-light.
    """
    if not frames:
        return 0.0
    values = []
    for frame in frames:
        if frame.size < 8:
            continue
        window = np.hanning(frame.size).astype(np.float32)
        spectrum = np.abs(np.fft.rfft(frame * window)) ** 2
        if spectrum.size == 0:
            continue
        arithmetic = float(np.mean(spectrum))
        if arithmetic <= _EPS:
            # A frame with no energy at all. Reported as 0.0 rather than as the
            # 1.0 the ratio below would yield, because "perfectly flat" is a
            # statistical artefact of silence, not a cue.
            values.append(0.0)
            continue
        geometric = float(np.exp(np.mean(np.log(spectrum + _EPS))))
        values.append(geometric / (arithmetic + _EPS))
    if not values:
        return 0.0
    return float(np.mean(values))


def _zero_crossing_rate(arr: np.ndarray) -> float:
    if arr.size < 2:
        return 0.0
    signs = np.signbit(arr)
    return float(np.mean(signs[1:] != signs[:-1]))


def _content_jitter(arr: np.ndarray, seed: int) -> float:
    """
    A deterministic, content-derived offset in +/- `_JITTER_RANGE` / 2.

    The samples are quantised to 16-bit PCM before hashing, which is the form
    the ephemeral ring buffer receives them in. That makes the hash stable
    under a float -> int16 -> float round trip, so the same recording scores
    identically whether it arrives as a live PCM packet or as a decoded upload.
    Rounding the floats to a few decimals instead would not: the int16 step is
    3e-5, small enough that any one of 24,000 samples could land on the far
    side of a rounding boundary and change the digest.
    """
    if arr.size == 0:
        payload = b""
    else:
        quantised = np.clip(np.round(arr.astype(np.float64) * 32768.0), -32768, 32767)
        payload = quantised.astype("<i2").tobytes()
    digest = hashlib.blake2b(
        payload, digest_size=8, person=b"vanidemo", key=_seed_key(seed)
    ).digest()
    unit = int.from_bytes(digest, "big") / float(1 << 64)
    return (unit - 0.5) * _JITTER_RANGE


def _seed_key(seed: int) -> bytes:
    # blake2b keys are capped at 64 bytes; the seed is folded in as one.
    return int(seed).to_bytes(8, "big", signed=True)


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


class DemoAcousticScorer:
    """
    A stand-in for the Wav2Vec 2.0 acoustic pillar. NOT a trained detector.

    `score_windows` accepts the same per-window clip list that
    `evaluate_live_call` builds and returns one 0..1 number per window, so the
    stub can be dropped in where the real batched forward pass would run. The
    numbers are heuristic, not predictions.
    """

    def __init__(self, seed: int = 20260101) -> None:
        self.seed = int(seed)
        self.model_name = "DEMO STUB (not a trained detector)"
        self.is_trained = False
        logger.warning("%s DemoAcousticScorer seed=%d.", DEMO_BANNER, self.seed)

    def score_windows(self, clips: Optional[Iterable]) -> list:
        """
        Scores each window independently and deterministically.

        Returns a list of floats in 0..1, one per input window, in input order.
        An empty or silent window scores low rather than being dropped, so the
        caller can zip the result against its own window list without
        re-aligning indices.
        """
        if clips is None:
            return []
        scores: List[float] = []
        for clip in clips:
            scores.append(self._score_one(_as_mono_float(clip)))
        return scores

    def score_windows_percent(self, clips: Optional[Iterable]) -> list:
        """
        The same scores on the 0..100 scale `evaluate_live_call` uses for
        `window_probs`, so a caller substituting this stub for the real forward
        pass does not have to remember which scale it is on.
        """
        return [round(s * 100.0, 1) for s in self.score_windows(clips)]

    def describe(self) -> dict:
        """
        Self-description for health endpoints, logs and UI badges. Every
        consumer of this dict is handed the warning along with the name.
        """
        return {
            "model_name": self.model_name,
            "is_trained": False,
            "is_detection_result": False,
            "pillar": "acoustic",
            "kind": "deterministic_heuristic_stub",
            "seed": self.seed,
            "deterministic": True,
            "env_var": DEMO_ENV_VAR,
            "enabled": demo_enabled(),
            "cues": [
                "spectral_flatness",
                "zero_crossing_rate",
                "rms_energy",
                "content_hash",
            ],
            "dependencies": ["numpy", "stdlib"],
            "warning": DEMO_BANNER,
            "accuracy": "none claimed; this module has never been evaluated "
                        "against any dataset and cannot be",
            "do_not_use_for": [
                "detection verdicts",
                "benchmarks or published metrics",
                "anything a user or pilot partner is shown as a result",
            ],
        }

    # -- internals ---------------------------------------------------------

    def _score_one(self, arr: np.ndarray) -> float:
        if arr.size == 0:
            return 0.0

        rms = float(np.sqrt(np.mean(np.square(arr, dtype=np.float64))))
        peak = float(np.max(np.abs(arr)))
        if peak <= 0.0:
            # Digital silence. Nothing to score, and nothing to pretend about.
            return 0.0

        frames = _frames(arr)
        flatness = _spectral_flatness(frames)
        zcr = _zero_crossing_rate(arr)

        flatness_cue = _clamp01(flatness / _FLATNESS_FULL_SCALE)
        zcr_cue = _clamp01((zcr - _ZCR_FLOOR) / _ZCR_SPAN)
        level_cue = _clamp01(rms / _RMS_FULL_SCALE)

        # Flatness carries the most weight because unnatural harmonic flatness
        # is the cue the real vocoder-physics checks lean on hardest. Level is
        # a confidence damper, not evidence: a near-silent window should not
        # score high off a noise-floor statistic.
        blend = 0.55 * flatness_cue + 0.30 * zcr_cue + 0.15 * level_cue
        blend *= 0.45 + 0.55 * level_cue

        score = _SCORE_FLOOR + _SCORE_SPAN * blend + _content_jitter(arr, self.seed)

        if rms < _VAD_RMS and peak < _VAD_PEAK:
            # Room tone, line hiss or a muted mic. Scaled rather than floored so
            # the sparkline still moves, but kept well clear of the 0.70
            # synthetic threshold the session timeline latches on.
            score *= _SILENT_SCALE

        return round(_clamp01(score), 4)
