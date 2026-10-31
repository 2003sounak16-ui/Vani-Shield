"""

VaniShield AI — Production FastAPI Server & Real-Time Threat Intelligence Engine

Integrates:

1. Audio Ear: Wav2Vec 2.0 (FP16 CUDA) fine-tuned on 4,447 samples

2. Decision Brain: Jev-Omni 12B GGUF + Calibrated FP32 Decision Head

3. Fast ASR & Multilingual Trigger Spotter: Faster-Whisper (INT8)

4. Prosodic DSP & Biometrics: Librosa & SpeechBrain ECAPA-TDNN + FAISS

5. Telecom Intelligence: Google Phonenumbers + Scammer Blacklist

6. Privacy Layer: In-Process Ephemeral Volatile RAM Ring Buffer (DPDP Act 2023 Compliant)

7. Frontend: 3-Panel Split Screen UI with Automated Banking Circuit Breaker

"""



import os

import sys

if hasattr(sys.stdout, "reconfigure"):

    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import io
import re

import time

import json

import glob

import uuid

import asyncio

import logging
import functools

# Required by the TTS rate limiter below (_tts_lock). Its absence made
# app.py raise NameError at import time, so the server could not start.
import threading
import hmac

from concurrent.futures import ThreadPoolExecutor

from typing import Optional, Dict, Any, List



try:

    import dotenv

    dotenv.load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env"))

except Exception:

    pass



os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

try:

    from huggingface_hub.utils import logging as hf_hub_logging

    hf_hub_logging.set_verbosity_error()

except Exception:

    pass



import numpy as np

import soundfile as sf

import torch

import torchaudio

from fastapi import (FastAPI, File, UploadFile, WebSocket, WebSocketDisconnect, HTTPException, Form, Header)

from fastapi.middleware.cors import CORSMiddleware

from fastapi.staticfiles import StaticFiles

from fastapi.responses import HTMLResponse, JSONResponse

from pydantic import BaseModel



from vanishield_engine import core_engine, JEV_SERVER_URL, PROJECT_ROOT, threat_type_from_fusion

from ephemeral_buffer import ram_ring_buffer, hash_caller_identity

import scam_stages
from scam_stages import fuse_risk
from call_session import registry as session_registry



logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

logger = logging.getLogger("VaniShieldAPI")



# Bounded Worker Executor and Admission Control for CPU/GPU heavy analysis tasks (Issue #6, #37)

MAX_CONCURRENT_WORKERS = min(4, os.cpu_count() or 2)

ANALYSIS_EXECUTOR = ThreadPoolExecutor(max_workers=MAX_CONCURRENT_WORKERS)

UPLOAD_SEMAPHORE = asyncio.Semaphore(MAX_CONCURRENT_WORKERS)



# Guard limits against memory exhaustion (Issue #5)

# Mutating the persisted blacklist is gated behind a shared control PIN, the
# same mechanism backend/api/banking_hold.py already uses to release a hold.
# Without it any client that can reach the socket could write an arbitrary
# number into a list that forces a 99.0 risk score and locks banking - which
# is a denial of service against whoever that number belongs to, and a way to
# poison a list every later call is scored against.
#
# Unlike banking_hold this has no default PIN. A well-known default on a
# persisted, shared list is worse than not shipping the feature, so an
# unconfigured deployment refuses the write instead of accepting it.
CONTROL_PIN_ENV_VARS = (
    "VANISHIELD_CONTROL_PIN",
    # Both of the names backend/api/banking_hold.py already accepts, so a
    # deployment that configured either of them does not silently get 503.
    "VANISHIELD_BANKING_PIN",
    "VANISHIELD_UNLOCK_PIN",
)
CONTROL_PIN_HEADER = "X-VaniShield-Pin"


def _configured_control_pin() -> Optional[str]:
    for name in CONTROL_PIN_ENV_VARS:
        value = os.environ.get(name, "").strip()
        if value:
            return value
    return None


def require_control_pin(pin: Optional[str]) -> None:
    """Raises unless `pin` matches the configured control PIN."""
    expected = _configured_control_pin()
    if not expected:
        raise HTTPException(
            status_code=503,
            detail=(
                "Blacklist changes are disabled: no control PIN is configured. "
                f"Set {CONTROL_PIN_ENV_VARS[0]} to enable them."
            ),
        )
    # compare_digest rather than ==, so a wrong PIN cannot be recovered by
    # timing the response. Encoded first: compare_digest rejects str inputs
    # that are not ASCII-only with TypeError, which would turn a non-ASCII PIN
    # into a 500 instead of an authorisation decision.
    if not pin:
        raise HTTPException(status_code=403, detail="Invalid or missing control PIN.")
    if not hmac.compare_digest(str(pin).encode("utf-8"), expected.encode("utf-8")):
        raise HTTPException(status_code=403, detail="Invalid or missing control PIN.")



MAX_UPLOAD_BYTES = 25 * 1024 * 1024               # 25 MB max upload

MAX_AUDIO_SECONDS = 180                           # 3 minutes max duration

MAX_DECODED_SAMPLES = MAX_AUDIO_SECONDS * 16000  # 2,880,000 samples @ 16kHz

# Minimum wall time between two inference passes on one live session. Clients
# stream ~250ms packets; a full pass over the 3.5s window costs more than that,
# so without this the executor queue grows without bound and every reported
# latency becomes a measure of the backlog rather than of the pipeline.
# Deliberately decoupled from the client's packet rate: a caller streaming
# 250ms packets gets a frame about every 500ms, not one per packet. Clients
# must treat frames as a stream of updates rather than request/response.
# Set VANISHIELD_MIN_EVAL_INTERVAL=0 to score every packet.
DEFAULT_MIN_EVAL_INTERVAL_SECONDS = 0.5


def _eval_interval_from_env() -> float:
    """
    Reads the interval, falling back rather than refusing to boot.

    A typo in an environment variable should not be the difference between a
    running scam detector and a stack trace, so an unparseable or negative
    value is logged loudly and the default used instead.
    """
    raw = os.environ.get("VANISHIELD_MIN_EVAL_INTERVAL", "").strip()
    if not raw:
        return DEFAULT_MIN_EVAL_INTERVAL_SECONDS
    try:
        value = float(raw)
    except (TypeError, ValueError):
        logger.warning(
            "VANISHIELD_MIN_EVAL_INTERVAL=%r is not a number; using %.2fs.",
            raw, DEFAULT_MIN_EVAL_INTERVAL_SECONDS,
        )
        return DEFAULT_MIN_EVAL_INTERVAL_SECONDS
    if value < 0 or value != value:  # negative, or NaN
        logger.warning(
            "VANISHIELD_MIN_EVAL_INTERVAL=%r is not a valid interval; using %.2fs.",
            raw, DEFAULT_MIN_EVAL_INTERVAL_SECONDS,
        )
        return DEFAULT_MIN_EVAL_INTERVAL_SECONDS
    return value


MIN_EVAL_INTERVAL_SECONDS = _eval_interval_from_env()



app = FastAPI(

    title="VaniShield AI — Real-Time Voice Cloning Defense & Banking Circuit Breaker",

    description="Full-stack AI Audio Forensics, Jev-Omni Brain, and UPI Lockdown Server",

    version="2.0.0"

)



# Startup validation: Fail startup clearly if trained weights are missing (Issue #1)

@app.on_event("startup")

async def startup_check():

    # A control PIN that cannot travel in an HTTP header is a silent trap: the
    # websocket path (JSON) accepts it while REST can never receive it, so the
    # operator sees a permanent, inexplicable 403 on half the surface. Warn
    # rather than refuse, since a websocket-only deployment is still coherent.
    _pin = _configured_control_pin()
    if _pin:
        try:
            _pin.encode("latin-1")
        except UnicodeEncodeError:
            logger.warning(
                "Control PIN contains non-ASCII characters. HTTP headers cannot carry it, "
                "so %s will always reject it and the blacklist endpoints will be unreachable. "
                "The websocket observer path will still accept it. Use an ASCII PIN.",
                CONTROL_PIN_HEADER,
            )


    if core_engine is None:

        logger.critical("FATAL: VaniShield Core failed to initialize because no trained Wav2Vec 2.0 checkpoint was found.")

        raise RuntimeError(

            "VaniShield AI server cannot start without trained checkpoint weights. "

            "Please provide a fine-tuned model checkpoint directory containing config.json."

        )



# Explicit allowed origins to secure endpoints and prevent CWE-942

allowed_origins_env = os.environ.get("VANISHIELD_ALLOWED_ORIGINS")

if allowed_origins_env:

    allowed_origins = [o.strip() for o in allowed_origins_env.split(",") if o.strip()]

else:

    allowed_origins = [

        "http://localhost:8000",

        "http://127.0.0.1:8000",

        "http://localhost:3000",

        "http://127.0.0.1:3000"

    ]



app.add_middleware(

    CORSMiddleware,

    allow_origins=allowed_origins,

    allow_credentials=True,

    allow_methods=["GET", "POST", "OPTIONS"],

    allow_headers=["*"],

)



# Mount Static Files for 3-Panel Split-Screen Frontend

STATIC_DIR = os.path.join(PROJECT_ROOT, "static")

os.makedirs(STATIC_DIR, exist_ok=True)

app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")



def _serve_static_page(filename: str, missing_hint: str) -> str:
    path = os.path.join(STATIC_DIR, filename)
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return f.read()
    return f"<h1>VaniShield AI Backend Online</h1><p>{missing_hint}</p>"


@app.get("/", response_class=HTMLResponse)
async def serve_index():
    return _serve_static_page("index.html", "static/index.html is missing.")


@app.get("/dashboard", response_class=HTMLResponse)
async def serve_dashboard():
    """Mission control: full pillar breakdown, per-window scores, event log."""
    return _serve_static_page("mission-control.html", "static/mission-control.html is missing.")


@app.get("/call", response_class=HTMLResponse)
async def serve_call_screen():
    """The on-call alert screen a person actually sees when the phone rings."""
    return _serve_static_page("call-screen.html", "static/call-screen.html is missing.")



# -------------------------------------------------------------

# PRESET SIMULATOR (English, Bengali, Hindi Attack & Safe Calls)

# Resolves paths relative to PROJECT_ROOT using shipped fixtures (Issue #7)

# -------------------------------------------------------------

PRESET_CALLS = {

    "en_safe": {

        "audio_file": os.path.join(PROJECT_ROOT, "demo_samples", "real", "call_en_safe.wav"),

        "phone": "+91-9830112233 (Bengaluru, Karnataka - Airtel)",

        "transcript": "Hi team, let's reschedule our design review meeting to Thursday afternoon at 3 PM. Let me know if that works."

    },

    "en_scam": {

        "audio_file": os.path.join(PROJECT_ROOT, "demo_samples", "fake", "call_en_scam.wav"),

        "phone": "+91-1409988776 (VoIP Spoofed - International Gateway)",

        "transcript": "Urgent security notice from State Bank of India fraud team. Your account has an illegal debit of 65,000 rupees. Share your 6-digit OTP and CVV right now to reverse it."

    },

    "bn_safe": {

        "audio_file": os.path.join(PROJECT_ROOT, "demo_samples", "real", "call_bn_safe.wav"),

        "phone": "+91-9433098765 (Kolkata, West Bengal - BSNL)",

        "transcript": "নমস্কার কাকিমা, আমি শুভঙ্কর বলছি। কাল বিকেলে আপনাদের বাড়ি আসব, কেমন আছেন সবাই?"

    },

    "bn_scam": {

        "audio_file": os.path.join(PROJECT_ROOT, "demo_samples", "fake", "call_bn_scam.wav"),

        "phone": "+91-1401122334 (Spoofed Virtual PBX)",

        "transcript": "জরুরি ব্যাংক সতর্কতা: আপনার স্টেট ব্যাংক একাউন্ট স্থগিত করা হয়েছে। অবিলম্বে ৬ সংখ্যার ওটিপি ও ইউপিআই পিন প্রদান করুন অন্যথায় অ্যাকাউন্ট স্থায়ীভাবে বন্ধ হবে।"

    },

    "hi_safe": {

        "audio_file": os.path.join(PROJECT_ROOT, "demo_samples", "real", "call_hi_safe.wav"),

        "phone": "+91-9811234567 (New Delhi - Jio)",

        "transcript": "नमस्ते भाई, कल कॉलेज की क्लास सुबह नौ बजे शुरू होगी। असाइनमेंट पूरा हो गया क्या?"

    },

    "hi_scam": {

        "audio_file": os.path.join(PROJECT_ROOT, "demo_samples", "fake", "call_hi_scam.wav"),

        "phone": "+91-1407766554 (Unregistered VoIP Line)",

        "transcript": "नमस्कार, एचडीएफसी बैंक हेड ऑफिस। आपके पैन कार्ड में विसंगति पाई गई है। तुरंत अपना एटीएम पिन, सीवीवी और ओटीपी शेयर करके पैन री-वेरीफाई करें।"

    }

}



def load_wav_16k_mono(path: str) -> np.ndarray:
    """
    Reads a fixture WAV as 16kHz mono float32.

    Uses soundfile rather than torchaudio.load: recent torchaudio delegates
    decoding to TorchCodec and raises without it, which broke every preset on
    an otherwise working install. soundfile reads WAV directly and is already
    a declared dependency. torchaudio stays as a fallback for containers
    soundfile cannot open.

    Also resolves fixtures that moved into demo_samples/{real,fake}/, so a
    preset entry written against either layout keeps working.
    """
    if not os.path.isabs(path):
        path = os.path.join(PROJECT_ROOT, path)

    if not os.path.exists(path):
        fname = os.path.basename(path)
        for sub in ("real", "fake", ""):
            cand = (
                os.path.join(PROJECT_ROOT, "demo_samples", sub, fname)
                if sub
                else os.path.join(PROJECT_ROOT, "demo_samples", fname)
            )
            if os.path.exists(cand):
                path = cand
                break
        else:
            raise FileNotFoundError(f"Preset audio file fixture not found: {path}")

    try:
        data, sr = sf.read(path, dtype="float32", always_2d=True)
        audio = data.mean(axis=1)
    except Exception as e:
        logger.warning(f"soundfile could not read {path} ({e}); falling back to torchaudio.")
        wav, sr = torchaudio.load(path)
        if wav.shape[0] > 1:
            wav = wav.mean(dim=0, keepdim=True)
        audio = wav.squeeze(0).numpy()

    if sr != 16000:
        import librosa
        audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)

    audio = np.ascontiguousarray(audio, dtype=np.float32)
    if len(audio) > MAX_DECODED_SAMPLES:
        audio = audio[:MAX_DECODED_SAMPLES]
    return audio


def _sync_simulate_preset(preset: dict) -> dict:
    audio_np = load_wav_16k_mono(preset["audio_file"])

    return core_engine.evaluate_live_call(

        audio_chunk_16k=audio_np,

        phone_number=preset["phone"],

        known_transcript=preset["transcript"],

        is_upload=True

    )



@app.get("/api/simulate/{preset_id}")

async def simulate_preset(preset_id: str):

    if core_engine is None:

        raise HTTPException(status_code=503, detail="VaniShield core engine not initialized.")

    preset = PRESET_CALLS.get(preset_id)

    if not preset:

        raise HTTPException(status_code=404, detail="Unknown preset ID")



    try:

        loop = asyncio.get_running_loop()

        result = await loop.run_in_executor(ANALYSIS_EXECUTOR, _sync_simulate_preset, preset)

        return result

    except FileNotFoundError as fnf:

        raise HTTPException(status_code=404, detail=str(fnf))

    except Exception as e:

        logger.error(f"Error during preset simulation '{preset_id}': {e}")

        raise HTTPException(status_code=500, detail=f"Simulation error: {e}")



def decode_audio_bytes(content: bytes, max_samples: int = MAX_DECODED_SAMPLES) -> np.ndarray:

    """Decodes audio container in volatile RAM, guarded against memory exhaustion (Issue #5)."""

    try:

        import av

        container = av.open(io.BytesIO(content))

        resampler = av.AudioResampler(format="fltp", layout="mono", rate=16000)

        frames = []

        accumulated_samples = 0

        for frame in container.decode(audio=0):

            for rf in resampler.resample(frame):

                arr_slice = rf.to_ndarray()

                needed = max_samples - accumulated_samples

                if needed <= 0:

                    break

                if arr_slice.shape[-1] > needed:

                    arr_slice = arr_slice[..., :needed]

                frames.append(arr_slice)

                accumulated_samples += arr_slice.shape[-1]

                if accumulated_samples >= max_samples:

                    break

            if accumulated_samples >= max_samples:

                break

        if frames:

            arr = np.concatenate(frames, axis=1).squeeze()

            if arr.ndim > 1:

                arr = arr.mean(axis=0)

            if len(arr) > max_samples:

                arr = arr[:max_samples]

            arr = np.clip(arr, -1.0, 1.0)

            return arr.astype(np.float32)

    except Exception as e:

        logger.warning(f"PyAV decode fallback triggered: {e}")



    try:

        with sf.SoundFile(io.BytesIO(content)) as sfile:

            orig_sr = sfile.samplerate

            max_frames_to_read = int(max_samples * (orig_sr / 16000.0)) + 1000 if orig_sr else max_samples

            data = sfile.read(frames=max_frames_to_read, dtype="float32")

            if data.ndim > 1:

                data = data.mean(axis=1)

            if orig_sr != 16000:

                import librosa

                data = librosa.resample(data, orig_sr=orig_sr, target_sr=16000)

            if len(data) > max_samples:

                data = data[:max_samples]

            return data.astype(np.float32)

    except Exception as e:

        logger.error(f"Audio decode failed: {e}")

        raise HTTPException(status_code=400, detail=f"Unsupported or unreadable audio format: {e}")



def _sync_evaluate_audio(content: bytes, phone_number: str, transcript: Optional[str]) -> dict:

    audio_np = decode_audio_bytes(content)

    clean_caller = phone_number.strip() if phone_number else "Direct Audio File Upload"

    result = core_engine.evaluate_live_call(

        audio_chunk_16k=audio_np,

        phone_number=clean_caller,

        known_transcript=transcript,

        is_upload=True

    )

    result["duration_seconds"] = round(len(audio_np) / 16000.0, 2)

    return result



# -------------------------------------------------------------

# REST ENDPOINT: AUDIO FILE EVALUATION & DROP UPLOAD

# -------------------------------------------------------------

@app.post("/api/evaluate/audio")

@app.post("/api/evaluate/upload")

async def evaluate_audio_file(

    file: UploadFile = File(...),

    phone_number: Optional[str] = Form("+91-0000000000 (Direct Audio Upload)"),

    transcript: Optional[str] = Form(None)

):

    """Processes dropped/uploaded audio file in volatile RAM with byte/duration limits offloaded to worker pool."""

    if core_engine is None:

        raise HTTPException(status_code=503, detail="VaniShield core engine not initialized.")



    logger.info(f"Received audio file upload: {file.filename} (content_type: {file.content_type})")



    # Admission control: Prevent memory exhaustion under concurrent upload floods (Issue #37)

    try:

        await asyncio.wait_for(UPLOAD_SEMAPHORE.acquire(), timeout=0.1)

    except asyncio.TimeoutError:

        raise HTTPException(

            status_code=503,

            detail="Server is currently at maximum upload evaluation capacity. Please retry shortly."

        )



    try:

        # Read uploaded bytes with bounded memory guard (Issue #5)

        total_bytes = 0

        chunks = []

        chunk_size = 64 * 1024

        while True:

            chunk = await file.read(chunk_size)

            if not chunk:

                break

            total_bytes += len(chunk)

            if total_bytes > MAX_UPLOAD_BYTES:

                raise HTTPException(

                    status_code=413,

                    detail=f"Uploaded file exceeds maximum limit of {MAX_UPLOAD_BYTES // (1024 * 1024)}MB."

                )

            chunks.append(chunk)



        if total_bytes == 0:

            raise HTTPException(status_code=400, detail="Uploaded file is empty.")



        content = b"".join(chunks)



        # Offload heavy decoding & inference to bounded executor (Issue #6)

        loop = asyncio.get_running_loop()

        result = await loop.run_in_executor(

            ANALYSIS_EXECUTOR,

            _sync_evaluate_audio,

            content,

            phone_number,

            transcript

        )

        result["filename"] = file.filename

        return result

    finally:

        UPLOAD_SEMAPHORE.release()
# -------------------------------------------------------------
# STAGE METADATA — lets a client render chips and thresholds before
# the first call arrives, in the language it asks for.
# -------------------------------------------------------------
@app.get("/api/stages")
async def get_stage_metadata(lang: str = "en"):
    lang = scam_stages.normalize_lang(lang)
    return {
        "lang": lang,
        "supported_langs": list(scam_stages.SUPPORTED_LANGS),
        "stages": [
            {
                "key": key,
                "label": scam_stages.stage_label(key, lang),
                "weight": scam_stages.STAGE_WEIGHTS[key],
                "explanation": scam_stages.stage_explanation(key, lang),
            }
            for key in scam_stages.STAGE_ORDER
        ],
        "thresholds": {
            "red": scam_stages.RED_THRESHOLD,
            "amber": scam_stages.AMBER_THRESHOLD,
            "acoustic_synthetic": scam_stages.ACOUSTIC_SYNTHETIC_THRESHOLD,
            "circuit_breaker": 70.0,
        },
        "band_labels": {
            band: scam_stages.band_label(band, lang)
            for band in (scam_stages.BAND_RED, scam_stages.BAND_AMBER, scam_stages.BAND_GREEN)
        },
    }


# -------------------------------------------------------------
# SESSION-ACCUMULATED FRAME ASSEMBLY
#
# The engine evaluates one window at a time and is deliberately stateless.
# A call, though, is cumulative: a scammer who established authority thirty
# seconds ago has still established it now. So the frame a client receives is
# re-fused from the session's accumulated stages and whole-call EMA, not from
# the single window that happened to arrive last.
# -------------------------------------------------------------
# -------------------------------------------------------------
# OBSERVERS
#
# A call happens on one socket (the phone) but is useful to watch from
# another (the operator dashboard). Frames are therefore fanned out to any
# observer subscribed to that session id, read-only.
#
# This is gated, unlike the rest of the read surface, because an observer
# receives the live transcript - the actual words someone is speaking into
# their phone. An ungated version of this endpoint would turn a product whose
# premise is "nothing is recorded" into a wiretap that anyone on the box can
# attach to. Observers authenticate with the same control PIN as the
# blacklist, sent as the first message rather than in the URL so it does not
# end up in access logs or browser history.
# -------------------------------------------------------------
OBSERVER_AUTH_TIMEOUT_SECONDS = 5.0

# Session id prefixes. Only live calls are observable; a replay is a fixture
# being scored, not a person on a phone.
LIVE_SESSION_PREFIX = "call_"
REPLAY_SESSION_PREFIX = "replay_"

_observers: Dict[str, set] = {}
_observers_lock = asyncio.Lock()


async def _register_observer(session_id: str, websocket: WebSocket) -> None:
    async with _observers_lock:
        _observers.setdefault(session_id, set()).add(websocket)


async def _unregister_observer(session_id: str, websocket: WebSocket) -> None:
    async with _observers_lock:
        watchers = _observers.get(session_id)
        if not watchers:
            return
        watchers.discard(websocket)
        if not watchers:
            _observers.pop(session_id, None)


async def _publish_to_observers(session_id: str, frame: dict) -> None:
    """
    Fans one frame out to every observer of a session.

    A failing observer is dropped rather than allowed to break the call it is
    watching: the person on the phone must not lose their scam warning because
    a dashboard tab went away.
    """
    async with _observers_lock:
        watchers = list(_observers.get(session_id, ()))
    if not watchers:
        return

    payload = dict(frame)
    payload["observed"] = True
    for watcher in watchers:
        try:
            await watcher.send_json(payload)
        except Exception:
            await _unregister_observer(session_id, watcher)



class _CallAccumulator:
    """Sticky per-call evidence that the stateless engine cannot carry."""

    def __init__(self):
        self.payment_attempt = False
        self.session_blocked = False
        self.high_sensitivity = False

    def observe(self, stage_payload: dict):
        if stage_payload.get("payment_attempt"):
            self.payment_attempt = True
        if stage_payload.get("high_sensitivity_credential"):
            self.high_sensitivity = True


def _assemble_frame(result: dict, session, accumulator: "_CallAccumulator") -> dict:
    """Folds one engine result into the session and returns the client frame."""
    acoustic = result.get("acoustic") or {}
    stage_payload = result.get("stage_analysis") or {}
    telecom = result.get("telecom") or {}
    accumulator.observe(stage_payload)

    lang = scam_stages.normalize_lang(result.get("lang"))
    session.set_lang(lang)

    update = session.update(
        window_probs=acoustic.get("window_probs") or [],
        stages=stage_payload.get("stages") or [],
        payment_attempt=accumulator.payment_attempt,
        risk_score=float((result.get("fusion") or {}).get("risk_score") or 0.0),
        band=(result.get("fusion") or {}).get("band") or scam_stages.BAND_GREEN,
        latency_ms=result.get("latency_ms"),
        transcript=result.get("transcript") or "",
        circuit_breaker_engaged=bool((result.get("circuit_breaker") or {}).get("engaged")),
        high_sensitivity_credential=accumulator.high_sensitivity,
        line_silent=(result.get("status") == "STANDBY_SILENCE"),
    )

    accumulated_stages = update["stages"]
    call_ema = update["acoustic_ema"]

    fusion = fuse_risk(
        accumulated_stages,
        acoustic_prob=call_ema,
        payment_attempt=accumulator.payment_attempt,
        blacklisted=bool(telecom.get("is_blacklisted", False)) or accumulator.session_blocked,
        voip_spoofed=bool(telecom.get("is_voip_spoofed", False)),
        credential_high_sensitivity=accumulator.high_sensitivity,
    )
    breaker_engaged = bool(
        telecom.get("is_blacklisted", False) or accumulator.session_blocked or fusion["risk_score"] >= 70.0
    )

    # CallSession names a running call "LIVE"; the client contract calls it
    # "ACTIVE". Normalise here rather than in either of them, so the wire
    # format stays stable whichever side is edited next.
    call_state = update["call_state"]
    if call_state == "LIVE":
        call_state = "ACTIVE"

    frame = dict(result)
    frame["session_id"] = session.session_id
    frame["elapsed_sec"] = update["elapsed_sec"]
    frame["call_state"] = call_state
    frame["events"] = update["new_events"]
    frame["event_cursor"] = update["event_cursor"]
    frame["lang"] = lang
    frame["acoustic"] = {
        **acoustic,
        "ema": call_ema,
        "synthetic": bool(call_ema >= scam_stages.ACOUSTIC_SYNTHETIC_THRESHOLD),
    }
    frame["stage_analysis"] = {
        **stage_payload,
        "stages": accumulated_stages,
        "stage_count": len(accumulated_stages),
        "labels": scam_stages.stage_labels(lang),
        "explanations": [
            scam_stages.stage_explanation(s, lang) for s in accumulated_stages
        ],
        "payment_attempt": accumulator.payment_attempt,
        "high_sensitivity_credential": accumulator.high_sensitivity,
        "lang": lang,
    }
    frame["fusion"] = {
        **fusion,
        "band_label": scam_stages.band_label(fusion["band"], lang),
    }
    frame["unified_risk_score"] = fusion["risk_score"]
    # Re-derive from the accumulated evidence, for the same reason the engine
    # does: a badge that contradicts the score is worse than no badge.
    frame["threat_type"] = threat_type_from_fusion(fusion, accumulated_stages, call_ema)
    frame["circuit_breaker"] = {
        "engaged": breaker_engaged,
        "action": "LOCK_BANKING_AND_UPI" if breaker_engaged else "ALLOW_NORMAL_CALL",
        "threshold": 70.0,
    }
    return frame


# -------------------------------------------------------------
# WEBSOCKET: PRESET REPLAY
# Streams a shipped sample call through the real pipeline in steps, so the
# dashboard's staged reveal is driven by actual inference rather than by a
# scripted timer. Audio grows cumulatively and the transcript is revealed one
# sentence at a time, which is what a live call actually looks like.
# -------------------------------------------------------------
REPLAY_STEP_SECONDS = 2.4
REPLAY_MIN_STEPS = 5
REPLAY_MAX_STEPS = 8
_SENTENCE_SPLIT = re.compile(r"(?<=[.!?।])\s+")


def _split_transcript(transcript: str) -> List[str]:
    if not transcript:
        return []
    parts = [p.strip() for p in _SENTENCE_SPLIT.split(transcript) if p.strip()]
    return parts or [transcript.strip()]


def _load_preset_audio(preset: dict) -> np.ndarray:
    return load_wav_16k_mono(preset["audio_file"])


@app.websocket("/ws/replay/{preset_id}")
async def replay_preset_websocket(websocket: WebSocket, preset_id: str):
    await websocket.accept()
    session_id = f"replay_{int(time.time()*1000)}_{uuid.uuid4().hex[:8]}"

    preset = PRESET_CALLS.get(preset_id)
    if preset is None:
        await websocket.send_json({"status": "error", "error": f"Unknown preset '{preset_id}'"})
        await websocket.close()
        return
    if core_engine is None:
        await websocket.send_json({
            "status": "error",
            "error": "VaniShield core engine not initialized (trained weights missing).",
        })
        await websocket.close()
        return

    loop = asyncio.get_running_loop()
    session = session_registry.get_or_create(session_id, preset["phone"])
    accumulator = _CallAccumulator()

    try:
        audio = await loop.run_in_executor(ANALYSIS_EXECUTOR, _load_preset_audio, preset)
        sentences = _split_transcript(preset.get("transcript", ""))
        step_samples = int(REPLAY_STEP_SECONDS * 16000)

        # Pace by audio length, not by sentence count. Keying the step count to
        # sentences made the Bengali fixture — two long sentences — finish in
        # about two seconds, which is too fast to read and makes the staged
        # reveal pointless. Clamped so a very short or very long clip still
        # plays over a watchable span.
        audio_steps = max(1, -(-len(audio) // step_samples))
        total_steps = max(REPLAY_MIN_STEPS, min(REPLAY_MAX_STEPS, audio_steps))

        for step in range(total_steps):
            progress = (step + 1) / total_steps
            # Cumulative audio, so each step re-scores everything heard so far.
            upto = len(audio) if step == total_steps - 1 else max(
                step_samples, min(len(audio), round(len(audio) * progress))
            )
            chunk = audio[:upto]
            # Reveal the transcript in step with the audio rather than one
            # sentence per frame, so the two stay in sync at any step count.
            if sentences:
                shown = max(1, min(len(sentences), round(len(sentences) * progress)))
                revealed = " ".join(sentences[:shown])
            else:
                revealed = None

            result = await loop.run_in_executor(
                ANALYSIS_EXECUTOR,
                functools.partial(
                    core_engine.evaluate_live_call,
                    audio_chunk_16k=chunk,
                    phone_number=preset["phone"],
                    known_transcript=revealed,
                    is_upload=False,
                    persist_blacklist=False,
                ),
            )
            frame = _assemble_frame(result, session, accumulator)
            frame["replay"] = {
                "preset_id": preset_id,
                "step": step + 1,
                "total_steps": total_steps,
            }
            if step == total_steps - 1:
                frame["call_state"] = "ENDED"
            await websocket.send_json(frame)
            await _publish_to_observers(session_id, frame)

            if step < total_steps - 1:
                await asyncio.sleep(REPLAY_STEP_SECONDS)

        session.end("ENDED")
    except WebSocketDisconnect:
        logger.info(f"Replay {session_id} disconnected by client.")
    except FileNotFoundError as fnf:
        try:
            await websocket.send_json({"status": "error", "error": str(fnf)})
        except Exception:
            pass
    except Exception as e:
        logger.error(f"Replay error for preset '{preset_id}': {e}")
        try:
            await websocket.send_json({"status": "error", "error": f"Replay failed: {e}"})
        except Exception:
            pass
    finally:
        session_registry.drop(session_id)
        try:
            await websocket.close()
        except Exception:
            pass







# -------------------------------------------------------------

# WEBSOCKET: REAL-TIME STREAMING & EPHEMERAL RAM BUFFER

# -------------------------------------------------------------

@app.get("/api/sessions")
async def list_active_sessions(pin: Optional[str] = Header(None, alias=CONTROL_PIN_HEADER)):
    """
    Lists calls currently in progress, so a dashboard can pick one to watch.

    PIN-gated and numbers are masked to their last four digits: knowing who is
    on the phone right now is itself disclosure, and the picker does not need
    the full number to work.
    """
    require_control_pin(pin)

    # Drop sessions whose socket went away without an END_CALL, so the picker
    # cannot offer a call that is no longer happening.
    session_registry.reap_idle()

    sessions = []
    for session in session_registry.all():
        # Replays are fixtures, not calls. Listing them here would present a
        # demo as someone else's live call - the precise confusion the
        # observing banner exists to prevent, reintroduced one layer down.
        if not session.session_id.startswith(LIVE_SESSION_PREFIX):
            continue
        phone = session.phone_number or ""
        digits = "".join(ch for ch in phone if ch.isdigit())
        sessions.append({
            "session_id": session.session_id,
            "caller_masked": f"\u2026{digits[-4:]}" if len(digits) >= 4 else "unknown",
            "elapsed_sec": session.elapsed_sec,
            "lang": session.lang,
            "stages": session.stages,
            "call_state": "ACTIVE" if session.call_state == "LIVE" else session.call_state,
            "observers": len(_observers.get(session.session_id, ())),
        })
    sessions.sort(key=lambda s: s["elapsed_sec"], reverse=True)
    return {"sessions": sessions, "count": len(sessions)}


@app.websocket("/ws/observe/{session_id}")
async def observe_session_websocket(websocket: WebSocket, session_id: str):
    """
    Read-only subscription to another session's frames.

    Accepts no control messages at all. An observer cannot set the caller,
    block, or end the call it is watching - it only receives what the call's
    own socket is already being sent.
    """
    await websocket.accept()

    # Authenticate before anything about the call is disclosed, including
    # whether a session by this id exists.
    try:
        raw = await asyncio.wait_for(websocket.receive_text(), timeout=OBSERVER_AUTH_TIMEOUT_SECONDS)
        pin = (json.loads(raw) or {}).get("pin")
    except (asyncio.TimeoutError, ValueError, TypeError, WebSocketDisconnect):
        await websocket.close(code=4401)
        return

    try:
        require_control_pin(pin)
    except HTTPException as exc:
        try:
            await websocket.send_json({"status": "error", "error": exc.detail})
        finally:
            await websocket.close(code=4403)
        return

    if session_registry.get(session_id) is None or not session_id.startswith(LIVE_SESSION_PREFIX):
        # Same answer for "ended", "never existed" and "is a replay", so the
        # endpoint does not confirm which to an unauthorised guess.
        await websocket.send_json({"status": "error", "error": f"No active session '{session_id}'."})
        await websocket.close(code=4404)
        return

    await _register_observer(session_id, websocket)
    logger.info(f"Observer attached to session {session_id}.")

    # The acknowledgement is inside the try, not before it: registration has
    # already happened, so a send that fails here must still reach the finally
    # and unregister. Otherwise a dead socket lingers in the fan-out set until
    # some later publish happens to prune it.
    try:
        await websocket.send_json({"status": "observing", "session_id": session_id})
        # Stay open until the observer leaves. Anything it sends is ignored:
        # observers watch, they do not steer.
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        pass
    except Exception as e:
        logger.debug(f"Observer on {session_id} closed: {e}")
    finally:
        await _unregister_observer(session_id, websocket)
        logger.info(f"Observer detached from session {session_id}.")



@app.websocket("/ws/live-call")

async def live_call_websocket(websocket: WebSocket):

    """

    Bi-directional streaming WebSocket.

    Audio flows through the in-memory DPDP Ephemeral RAM Buffer.

    Emits real-time scam scores and triggers the Banking Circuit Breaker.

    """

    await websocket.accept()

    # Unique session key per connection (Issue #11)

    session_id = f"call_{int(time.time()*1000)}_{uuid.uuid4().hex[:8]}"

    logger.info(f"WebSocket client connected. Session ID: {session_id}")



    # Isolated caller identity per connection (Issue #3)

    # Default to an inbound caller identity specific to this session, allowing client overrides

    caller_phone = (

        websocket.query_params.get("phone")

        or websocket.query_params.get("caller_id")

        or f"+91-0000000000 (Live Call {session_id[-6:]})"

    )



    loop = asyncio.get_running_loop()
    # Reap abandoned sessions here rather than on a timer: a half-open socket
    # that never sent END_CALL would otherwise pin its state indefinitely, and
    # a new connection is the only moment we are guaranteed to be running.
    reaped = session_registry.reap_idle()
    if reaped:
        logger.info(f"Reaped {reaped} idle call session(s) before accepting {session_id}.")

    session = session_registry.get_or_create(session_id, caller_phone)
    accumulator = _CallAccumulator()
    last_eval = 0.0

    try:
        while True:
            # Receive binary PCM packet from client
            message = await websocket.receive()
            if "bytes" in message:
                pcm_bytes = message["bytes"]
                # 1. Push to Ephemeral RAM Ring Buffer (sliding window, 0 disk I/O)
                ram_ring_buffer.push_chunk(session_id, pcm_bytes)

                # Throttle inference. Clients send ~250ms packets, but a full
                # pass over the 3.5s window costs far more than 250ms of GPU,
                # so evaluating every packet would queue up unboundedly and
                # make the reported latency meaningless. Buffer every packet,
                # score at most every MIN_EVAL_INTERVAL.
                now = time.perf_counter()
                if now - last_eval < MIN_EVAL_INTERVAL_SECONDS:
                    continue
                last_eval = now

                # 2. Extract current evaluation window from RAM
                eval_audio = ram_ring_buffer.get_eval_window(session_id)

                if core_engine is None:
                    await websocket.send_json({
                        "status": "error",
                        "session_id": session_id,
                        "error": "Core engine uninitialized (trained weights missing).",
                    })
                    continue

                # 3. Offload evaluation to the bounded worker pool, telling the
                #    engine the language this session has settled on so a
                #    silent window still renders in the right language.
                eval_result = await loop.run_in_executor(
                    ANALYSIS_EXECUTOR,
                    functools.partial(
                        core_engine.evaluate_live_call,
                        audio_chunk_16k=eval_audio,
                        phone_number=caller_phone,
                        lang=session.lang,
                    ),
                )

                # 4. Stream the session-accumulated frame back to the client
                frame = _assemble_frame(eval_result, session, accumulator)
                await websocket.send_json(frame)
                await _publish_to_observers(session_id, frame)

            elif "text" in message:
                try:
                    data = json.loads(message["text"])
                except (ValueError, TypeError):
                    logger.warning(f"Session {session_id} sent malformed control text; ignoring.")
                    continue

                action = (data.get("action") or "").upper()
                if action == "SET_CALLER":
                    new_phone = data.get("phone") or data.get("caller_id")
                    if new_phone:
                        caller_phone = str(new_phone).strip()
                        session.set_caller(caller_phone)
                        logger.info(f"Session {session_id} updated caller ID to: {caller_phone}")
                elif action == "BLOCK_CALLER":
                    # Scoped to this call only, and never written to disk.
                    # The caller number here is whatever the client asserted
                    # (query param or SET_CALLER), so honouring it as a
                    # persistent, global block would let any unauthenticated
                    # client blacklist a number it simply made up. Persisting
                    # a block is an administrative action and lives behind the
                    # PIN-gated REST endpoint instead.
                    accumulator.session_blocked = True
                    logger.info(f"Session {session_id} blocked its caller for the duration of the call.")
                    await websocket.send_json({
                        "status": "blocked",
                        "session_id": session_id,
                        "caller": core_engine.normalize_caller_key(caller_phone) if core_engine else caller_phone,
                        "persisted": False,
                        "scope": "session",
                    })

                elif action == "END_CALL":
                    # Hard purge from volatile memory upon call completion
                    ram_ring_buffer.purge_session(session_id)
                    ended = session.end("ENDED")
                    try:
                        await websocket.send_json({
                            "status": "ended",
                            "session_id": session_id,
                            "call_state": "ENDED",
                            "events": (ended or {}).get("new_events", []),
                        })
                    except Exception:
                        pass
                    logger.info(f"Session {session_id} ended. Volatile RAM purged.")
                    break
    except WebSocketDisconnect:
        ram_ring_buffer.purge_session(session_id)
        logger.info(f"WebSocket client disconnected. Session {session_id} purged from RAM.")
    except Exception as e:
        ram_ring_buffer.purge_session(session_id)
        logger.error(f"WebSocket error: {e}")
    finally:
        session_registry.drop(session_id)



# -------------------------------------------------------------

# HEALTH & BLACKLIST ENDPOINTS

# -------------------------------------------------------------

@app.get("/health")
async def health_check():
    demo_scorer = getattr(core_engine, "demo_scorer", None) if core_engine else None

    # Name the acoustic pillar honestly. A caller cannot otherwise tell a
    # trained detector from the demo stub, and a UI badge that reads "healthy"
    # over a heuristic would be the single most misleading thing here.
    if core_engine is None:
        acoustic_pillar = {"trained": False, "model": "unavailable"}
    elif demo_scorer is not None:
        acoustic_pillar = {"trained": False, **demo_scorer.describe()}
    else:
        acoustic_pillar = {"trained": True, "model": "Wav2Vec 2.0 (fine-tuned checkpoint)"}

    return {
        "status": "healthy" if core_engine is not None else "degraded_missing_weights",
        "engine": "VaniShield AI Core v2.0",
        "device": core_engine.device if core_engine is not None else "unavailable",
        "cuda_available": torch.cuda.is_available(),
        "jev_server": JEV_SERVER_URL,
        "dpdp_privacy": "Ephemeral In-Process RAM Ring Buffer Active",
        "demo_mode": demo_scorer is not None,
        "acoustic_pillar": acoustic_pillar,
        "asr_engine": getattr(core_engine, "asr_name", "unavailable") if core_engine else "unavailable",
        "stage_pillar": {
            "stages": list(scam_stages.STAGE_ORDER),
            "languages": list(scam_stages.SUPPORTED_LANGS),
        },
        "active_calls": session_registry.active_count(),
    }



@app.get("/api/blacklist")

async def get_blacklist():

    if core_engine is not None:

        return core_engine.blacklist

    return {}


class BlockRequest(BaseModel):
    phone_number: str
    reason: Optional[str] = "Blocked by user"


@app.post("/api/blacklist")
async def add_to_blacklist(
    payload: BlockRequest,
    pin: Optional[str] = Header(None, alias=CONTROL_PIN_HEADER),
):
    """Persistently blocks a caller. Requires the control PIN."""
    require_control_pin(pin)
    if core_engine is None:
        raise HTTPException(status_code=503, detail="VaniShield core engine not initialized.")
    try:
        result = core_engine.block_caller(payload.phone_number, reason=payload.reason or "Blocked by user")
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    logger.info(f"Blacklist add accepted for {result['caller']}.")
    return {"status": "blocked", **result}


@app.delete("/api/blacklist/{phone_number:path}")
async def remove_from_blacklist(
    phone_number: str,
    pin: Optional[str] = Header(None, alias=CONTROL_PIN_HEADER),
):
    """
    Unblocks a caller.

    The circuit breaker writes entries automatically at risk >= 70, so without
    this a single false positive would permanently block a real contact with no
    way back.
    """
    require_control_pin(pin)
    if core_engine is None:
        raise HTTPException(status_code=503, detail="VaniShield core engine not initialized.")
    removed = core_engine.unblock_caller(phone_number)
    if not removed:
        raise HTTPException(status_code=404, detail=f"'{phone_number}' is not on the blacklist.")



# -------------------------------------------------------------

# SARVAM AI STT & ELEVENLABS TTS INTEGRATION ENDPOINTS

# -------------------------------------------------------------

class SynthesizeRequest(BaseModel):

    text: str

    voice_id: Optional[str] = None

    model_id: Optional[str] = "eleven_multilingual_v2"

    phone_number: Optional[str] = "+91-1409988776 (ElevenLabs Cloned Call)"

    use_known_transcript: Optional[bool] = False



@app.get("/api/stt/status")

async def get_stt_status():

    from sarvam_stt import is_sarvam_available

    sarvam_on = is_sarvam_available()

    return {

        "engine": getattr(core_engine, "asr_name", "Faster-Whisper") if core_engine else "Unavailable",

        "sarvam_ai_active": sarvam_on,

        "provider": "Sarvam AI Saaras" if sarvam_on else "Faster-Whisper (Local)"

    }



MAX_TTS_TEXT_LENGTH = 500
_tts_lock = threading.Lock()
_tts_timestamps = []
MAX_TTS_PER_MINUTE = 10

def _check_tts_budget_and_length(text: str):
    stripped = text.strip()
    if not stripped:
        raise HTTPException(status_code=400, detail="Text field cannot be empty.")
    if len(stripped) > MAX_TTS_TEXT_LENGTH:
        raise HTTPException(
            status_code=400,
            detail=f"Text exceeds maximum permitted length ({MAX_TTS_TEXT_LENGTH} characters)."
        )
    now = time.time()
    with _tts_lock:
        _tts_timestamps[:] = [t for t in _tts_timestamps if now - t < 60.0]
        if len(_tts_timestamps) >= MAX_TTS_PER_MINUTE:
            raise HTTPException(
                status_code=429,
                detail="TTS generation rate limit / budget quota reached. Please try again in 1 minute."
            )
        _tts_timestamps.append(now)


@app.get("/api/tts/voices")

async def get_tts_voices():

    from elevenlabs_tts import is_elevenlabs_available, fetch_account_voices, get_curated_voices

    has_key = is_elevenlabs_available()

    loop = asyncio.get_running_loop()
    voices = await loop.run_in_executor(ANALYSIS_EXECUTOR, fetch_account_voices) if has_key else get_curated_voices()

    return {

        "elevenlabs_active": has_key,

        "provider": "ElevenLabs Neural TTS" if has_key else "Edge-TTS (Fallback)",

        "voices": voices

    }



@app.post("/api/tts/synthesize")

async def tts_synthesize(req: SynthesizeRequest):

    """Generates audio for given text via ElevenLabs API (or fallback) and streams audio/mpeg."""

    from elevenlabs_tts import synthesize_with_fallback

    from fastapi.responses import Response



    if not req.text.strip():

        raise HTTPException(status_code=400, detail="Text field cannot be empty.")



    _check_tts_budget_and_length(req.text)
    loop = asyncio.get_running_loop()
    audio_bytes, meta, err = await loop.run_in_executor(
        ANALYSIS_EXECUTOR,
        synthesize_with_fallback,
        req.text,
        req.voice_id or "21m00Tcm4TlvDq8ikWAM"
    )

    if err and not audio_bytes:

        raise HTTPException(status_code=500, detail=f"TTS Synthesis error: {err}")



    return Response(content=audio_bytes, media_type="audio/mpeg", headers={

        "X-TTS-Engine": meta.get("engine", "TTS"),

        "X-Voice-ID": meta.get("voice_id", "default")

    })



@app.post("/api/tts/simulate-attack")

async def tts_simulate_attack(req: SynthesizeRequest):

    """

    End-to-End Test: Synthesizes text via ElevenLabs TTS and immediately runs

    it through the full VaniShield 8-pillar forensic pipeline (including Sarvam STT & Wav2Vec 2.0).

    """

    import base64

    from elevenlabs_tts import synthesize_with_fallback



    if not req.text.strip():

        raise HTTPException(status_code=400, detail="Text field cannot be empty.")



    _check_tts_budget_and_length(req.text)
    loop = asyncio.get_running_loop()
    audio_bytes, meta, err = await loop.run_in_executor(
        ANALYSIS_EXECUTOR,
        synthesize_with_fallback,
        req.text,
        req.voice_id or "21m00Tcm4TlvDq8ikWAM"
    )

    if err and not audio_bytes:

        raise HTTPException(status_code=500, detail=f"TTS Synthesis error: {err}")



    loop = asyncio.get_running_loop()

    # If req.use_known_transcript is False, Sarvam AI STT will actively transcribe the audio live!

    transcript_arg = req.text if req.use_known_transcript else None

    analysis = await loop.run_in_executor(

        ANALYSIS_EXECUTOR,

        _sync_evaluate_audio,

        audio_bytes,

        req.phone_number or "+91-1409988776 (ElevenLabs Cloned Call)",

        transcript_arg

    )



    return {

        "analysis": analysis,

        "audio_base64": base64.b64encode(audio_bytes).decode("ascii"),

        "tts_metadata": meta

    }



if __name__ == "__main__":

    import uvicorn

    print("\n" + "=" * 70)

    print("VaniShield AI: Starting Unified FastAPI Backend & Web Dashboard")

    print("Dashboard URL: http://127.0.0.1:8000")

    print("WebSocket URL: ws://127.0.0.1:8000/ws/live-call")

    print("=" * 70 + "\n")

    uvicorn.run(app, host="127.0.0.1", port=8000, log_level="info")

