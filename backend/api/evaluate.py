"""
VaniShield AI — Unified Evaluation, Preset Simulator & Voice Forensics Router
Team CypherEye | Project VaniShield AI
Provides endpoints for audio upload evaluation, preset scam simulation, STT status, and bounded TTS testing.
"""

import os
import sys
import io
import time
import hmac
import json
import base64
import asyncio
import logging
import threading
from concurrent.futures import ThreadPoolExecutor
from typing import Optional, Dict, Any, List

import numpy as np
try:
    import soundfile as sf
except ImportError:
    sf = None

try:
    import torch
    import torchaudio
    TORCH_AVAILABLE = True
except ImportError:
    torch = None
    torchaudio = None
    TORCH_AVAILABLE = False

from fastapi import APIRouter, File, UploadFile, HTTPException, Form
from fastapi.responses import HTMLResponse, Response, JSONResponse
from pydantic import BaseModel

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BACKEND_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
PROJECT_ROOT = os.path.abspath(os.path.join(BACKEND_DIR, ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from backend.engine.caller_tracker import load_blacklist
from backend.api.banking_hold import state as banking_state

try:
    from vanishield_engine import core_engine, JEV_SERVER_URL
except ImportError:
    try:
        from backend.engine.vanishield_engine import core_engine, JEV_SERVER_URL
    except ImportError:
        core_engine = None
        JEV_SERVER_URL = "http://127.0.0.1:9931"

logger = logging.getLogger("VaniShieldEvaluate")

router = APIRouter(tags=["Evaluation & Simulation"])

# Bounded Worker Executor and Admission Control
MAX_CONCURRENT_WORKERS = min(4, os.cpu_count() or 2)
ANALYSIS_EXECUTOR = ThreadPoolExecutor(max_workers=MAX_CONCURRENT_WORKERS)
UPLOAD_SEMAPHORE = asyncio.Semaphore(MAX_CONCURRENT_WORKERS)

MAX_UPLOAD_BYTES = 25 * 1024 * 1024               # 25 MB max upload
MAX_AUDIO_SECONDS = 180                           # 3 minutes max duration
MAX_DECODED_SAMPLES = MAX_AUDIO_SECONDS * 16000  # 2,880,000 samples @ 16kHz
MAX_TTS_TEXT_LENGTH = 500

# Rate limiting and budget control for paid TTS providers
_tts_lock = threading.Lock()
_tts_timestamps: List[float] = []
_tts_ip_timestamps: dict = {}
MAX_TTS_PER_MINUTE = 10
MAX_TTS_PER_IP_PER_MINUTE = 5

def check_tts_access(request: Request, body_token: Optional[str] = None) -> None:
    """Verify authorization before allowing speech synthesis via paid ElevenLabs quota."""
    try:
        from elevenlabs_tts import is_elevenlabs_available
        has_elevenlabs = is_elevenlabs_available()
    except Exception:
        has_elevenlabs = bool(os.environ.get("ELEVENLABS_API_KEY"))

    expected = os.environ.get("TTS_ACCESS_TOKEN", "").strip()
    # If ElevenLabs is configured with an API key, require authorization
    if has_elevenlabs and not expected:
        expected = os.environ.get("VANISHIELD_API_KEY", os.environ.get("VANISHIELD_BANKING_PIN", os.environ.get("VANISHIELD_UNLOCK_PIN", "1234"))).strip()

    if expected:
        provided = (
            request.headers.get("X-TTS-Token")
            or request.headers.get("Authorization", "").replace("Bearer ", "")
            or (body_token or "")
        ).strip()
        if not provided or not hmac.compare_digest(provided, expected):
            raise HTTPException(
                status_code=401,
                detail="TTS endpoint requires authorization (valid X-TTS-Token header or access_token)."
            )

def check_tts_budget_and_length(text: str, request: Request = None) -> None:
    """Enforces text bounds and rate/budget limits before invoking paid voice providers."""
    stripped = text.strip()
    if not stripped:
        raise HTTPException(status_code=400, detail="Text field cannot be empty.")
    if len(stripped) > MAX_TTS_TEXT_LENGTH:
        raise HTTPException(
            status_code=400,
            detail=f"Text exceeds maximum permitted length ({MAX_TTS_TEXT_LENGTH} characters)."
        )
    now = time.time()
    client_ip = (request.client.host if request and request.client else "unknown")
    with _tts_lock:
        _tts_timestamps[:] = [t for t in _tts_timestamps if now - t < 60.0]
        if len(_tts_timestamps) >= MAX_TTS_PER_MINUTE:
            raise HTTPException(
                status_code=429,
                detail="TTS generation rate limit / budget quota reached. Please try again in 1 minute."
            )
        ip_ts = _tts_ip_timestamps.setdefault(client_ip, [])
        ip_ts[:] = [t for t in ip_ts if now - t < 60.0]
        if len(ip_ts) >= MAX_TTS_PER_IP_PER_MINUTE:
            raise HTTPException(
                status_code=429,
                detail=f"Per-client TTS rate limit reached. Maximum {MAX_TTS_PER_IP_PER_MINUTE} requests per minute."
            )
        _tts_timestamps.append(now)
        ip_ts.append(now)

# Presets definition
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

def load_audio_as_16k_np(path_or_bytes) -> np.ndarray:
    """Robust audio loader supporting torchaudio, soundfile, and scipy."""
    if TORCH_AVAILABLE and torchaudio is not None and isinstance(path_or_bytes, str):
        wav, sr = torchaudio.load(path_or_bytes)
        if wav.shape[0] > 1:
            wav = wav.mean(dim=0, keepdim=True)
        if sr != 16000:
            wav = torchaudio.transforms.Resample(sr, 16000)(wav)
        return wav.squeeze(0).numpy()

    if sf is not None:
        bio = io.BytesIO(path_or_bytes) if isinstance(path_or_bytes, bytes) else path_or_bytes
        data, sr = sf.read(bio)
        if data.ndim > 1:
            data = data.mean(axis=1)
        if sr != 16000:
            if TORCH_AVAILABLE and torchaudio is not None:
                tensor = torch.from_numpy(data).float().unsqueeze(0)
                tensor_16k = torchaudio.transforms.Resample(orig_freq=sr, new_freq=16000)(tensor)
                data = tensor_16k.squeeze(0).numpy()
            else:
                num_samples = int(len(data) * 16000 / sr)
                indices = np.linspace(0, len(data) - 1, num_samples)
                data = np.interp(indices, np.arange(len(data)), data).astype(np.float32)
        return data

    import wave
    with wave.open(path_or_bytes, 'rb') as wf:
        n_channels = wf.getnchannels()
        sampwidth = wf.getsampwidth()
        frames = wf.readframes(wf.getnframes())
        if sampwidth == 2:
            data = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0
        else:
            data = np.frombuffer(frames, dtype=np.float32)
        if n_channels > 1:
            data = data.reshape(-1, n_channels).mean(axis=1)
        return data

def _sync_simulate_preset(preset: dict) -> dict:
    path = preset["audio_file"]
    if not os.path.isabs(path):
        path = os.path.join(PROJECT_ROOT, path)
    if not os.path.exists(path):
        fname = os.path.basename(path)
        for sub in ["real", "fake", ""]:
            cand = os.path.join(PROJECT_ROOT, "demo_samples", sub, fname) if sub else os.path.join(PROJECT_ROOT, "demo_samples", fname)
            if os.path.exists(cand):
                path = cand
                break
    if not os.path.exists(path):
        raise FileNotFoundError(f"Preset audio file fixture not found: {path}")

    audio_np = load_audio_as_16k_np(path)

    if core_engine is not None:
        return core_engine.evaluate_live_call(
            audio_np,
            phone_number=preset["phone"],
            known_transcript=preset.get("transcript"),
            is_upload=True
        )
    # Lightweight fallback when running locally without full ML weights
    is_scam = any(k in preset.get("transcript", "").lower() for k in ["otp", "cvv", "illegal", "pin", "warrant", "debit"])
    risk = 98.5 if is_scam else 1.2
    return {
        "status": "success",
        "latency_ms": 15.0,
        "asr_engine": "Faster-Whisper (Demo Mode)",
        "w2v_fake_prob": 98.9 if is_scam else 0.8,
        "jev_scam_prob": 97.5 if is_scam else 1.0,
        "unified_risk_score": risk,
        "threat_type": "AI_VOICE_CLONE_FINANCIAL_SCAM" if is_scam else "AUTHENTIC_SAFE_CALL",
        "circuit_breaker": {
            "engaged": is_scam,
            "action": "LOCK_BANKING_AND_UPI" if is_scam else "ALLOW_NORMAL_CALL",
            "threshold": 70.0
        },
        "transcript": preset.get("transcript", ""),
        "telecom": {
            "caller_id": preset["phone"],
            "is_voip_spoofed": is_scam,
            "is_blacklisted": is_scam,
            "location": "Bengaluru"
        }
    }

def _sync_evaluate_audio(audio_bytes: bytes, phone: str = "+91-1409988776 (Direct Audio Stream)", known_transcript: Optional[str] = None) -> dict:
    data = load_audio_as_16k_np(audio_bytes)
    if len(data) > MAX_DECODED_SAMPLES:
        data = data[:MAX_DECODED_SAMPLES]

    if core_engine is not None:
        return core_engine.evaluate_live_call(
            data,
            phone_number=phone,
            known_transcript=known_transcript,
            is_upload=True
        )
    # Lightweight fallback for audio evaluation
    rms = float(np.sqrt(np.mean(data**2))) if len(data) > 0 else 0.0
    is_scam = bool(known_transcript and any(k in known_transcript.lower() for k in ["otp", "cvv", "pin", "kyc"]))
    risk = 95.0 if is_scam else (35.0 if rms > 0.05 else 5.0)
    return {
        "status": "success",
        "latency_ms": 18.0,
        "asr_engine": "Faster-Whisper (Demo Mode)",
        "w2v_fake_prob": 92.0 if is_scam else 10.0,
        "jev_scam_prob": 95.0 if is_scam else 5.0,
        "unified_risk_score": risk,
        "threat_type": "SUSPICIOUS_UNVERIFIED_CALL" if risk >= 40.0 else "AUTHENTIC_SAFE_CALL",
        "circuit_breaker": {
            "engaged": risk >= 70.0,
            "action": "LOCK_BANKING_AND_UPI" if risk >= 70.0 else "ALLOW_NORMAL_CALL",
            "threshold": 70.0
        },
        "transcript": known_transcript or "Audio stream analyzed.",
        "telecom": {"caller_id": phone, "is_voip_spoofed": False, "location": "Unknown"}
    }

class SynthesizeRequest(BaseModel):
    text: str
    voice_id: Optional[str] = None
    model_id: Optional[str] = "eleven_multilingual_v2"
    phone_number: Optional[str] = "+91-1409988776 (ElevenLabs Cloned Call)"
    use_known_transcript: Optional[bool] = False
    access_token: Optional[str] = None

@router.get("/", response_class=HTMLResponse)
async def serve_index():
    static_index = os.path.join(PROJECT_ROOT, "static", "index.html")
    if os.path.exists(static_index):
        with open(static_index, "r", encoding="utf-8") as f:
            return f.read()
    return "<h1>VaniShield AI Backend Online</h1>"

@router.get("/api/simulate/{preset_id}")
async def simulate_preset(preset_id: str):
    if preset_id not in PRESET_CALLS:
        raise HTTPException(status_code=404, detail=f"Preset '{preset_id}' not found. Available: {list(PRESET_CALLS.keys())}")
    preset = PRESET_CALLS[preset_id]
    loop = asyncio.get_running_loop()
    try:
        result = await loop.run_in_executor(ANALYSIS_EXECUTOR, _sync_simulate_preset, preset)
        if result.get("circuit_breaker", {}).get("engaged"):
            banking_state.locked = True
            banking_state.risk_score = result.get("unified_risk_score", 95.0)
            banking_state.reason = f"Simulated Threat: {result.get('threat_type')}"
        return result
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Error executing preset {preset_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/api/evaluate/audio")
@router.post("/api/evaluate/upload")
async def evaluate_audio_upload(
    audio: Optional[UploadFile] = File(None),
    file: Optional[UploadFile] = File(None),
    phone: Optional[str] = Form(None),
    known_transcript: Optional[str] = Form(None)
):
    upload_item = audio or file
    if upload_item is None:
        raise HTTPException(status_code=400, detail="Missing audio upload file ('audio' or 'file').")
    caller_phone = phone.strip() if (phone and phone.strip()) else "+91-1409988776 (User Upload Call)"
    async with UPLOAD_SEMAPHORE:
        content = await upload_item.read(MAX_UPLOAD_BYTES + 1024)
        if len(content) > MAX_UPLOAD_BYTES:
            raise HTTPException(status_code=413, detail=f"Uploaded file exceeds {MAX_UPLOAD_BYTES // (1024*1024)}MB maximum limit.")

        loop = asyncio.get_running_loop()
        try:
            result = await loop.run_in_executor(
                ANALYSIS_EXECUTOR,
                _sync_evaluate_audio,
                content,
                caller_phone,
                known_transcript.strip() if known_transcript else None
            )
            if result.get("circuit_breaker", {}).get("engaged"):
                banking_state.locked = True
                banking_state.risk_score = result.get("unified_risk_score", 95.0)
                banking_state.reason = f"Uploaded Threat: {result.get('threat_type')}"
            return result
        except Exception as e:
            logger.error(f"Error evaluating audio upload: {e}")
            raise HTTPException(status_code=500, detail=f"Audio processing failed: {str(e)}")

@router.get("/api/blacklist")
async def get_blacklist():
    if core_engine is not None and hasattr(core_engine, "blacklist"):
        return core_engine.blacklist
    return load_blacklist()

@router.get("/api/stt/status")
async def get_stt_status():
    try:
        from sarvam_stt import is_sarvam_available
        sarvam_on = is_sarvam_available()
    except Exception:
        sarvam_on = False

    return {
        "engine": getattr(core_engine, "asr_name", "Faster-Whisper") if core_engine else "Unavailable",
        "sarvam_ai_active": sarvam_on,
        "provider": "Sarvam AI Saaras" if sarvam_on else "Faster-Whisper (Local)"
    }

@router.get("/api/tts/voices")
async def get_tts_voices():
    try:
        from elevenlabs_tts import is_elevenlabs_available, fetch_account_voices, get_curated_voices
        has_key = is_elevenlabs_available()
        loop = asyncio.get_running_loop()
        voices = await loop.run_in_executor(ANALYSIS_EXECUTOR, fetch_account_voices) if has_key else get_curated_voices()
    except Exception as e:
        has_key = False
        voices = []

    return {
        "elevenlabs_active": has_key,
        "provider": "ElevenLabs Neural TTS" if has_key else "Edge-TTS (Fallback)",
        "voices": voices
    }

@router.post("/api/tts/synthesize")
async def tts_synthesize(req: SynthesizeRequest, request: Request):
    check_tts_access(request, req.access_token)
    check_tts_budget_and_length(req.text, request)
    from elevenlabs_tts import synthesize_with_fallback
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

@router.post("/api/tts/simulate-attack")
async def tts_simulate_attack(req: SynthesizeRequest, request: Request):
    check_tts_access(request, req.access_token)
    check_tts_budget_and_length(req.text, request)
    from elevenlabs_tts import synthesize_with_fallback
    loop = asyncio.get_running_loop()
    audio_bytes, meta, err = await loop.run_in_executor(
        ANALYSIS_EXECUTOR,
        synthesize_with_fallback,
        req.text,
        req.voice_id or "21m00Tcm4TlvDq8ikWAM"
    )
    if err and not audio_bytes:
        raise HTTPException(status_code=500, detail=f"TTS Synthesis error: {err}")

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
