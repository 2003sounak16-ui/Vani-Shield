try:
    from backend.api.banking_hold import router as banking_router, state as banking_state
except ImportError:
    from api.banking_hold import router as banking_router, state as banking_state
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
import hmac

import sys

if hasattr(sys.stdout, "reconfigure"):

    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import io

import time

import json

import glob

import uuid

import asyncio

import logging

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

from fastapi import FastAPI, File, UploadFile, WebSocket, WebSocketDisconnect, HTTPException, Form, Request

from fastapi.middleware.cors import CORSMiddleware

from fastapi.staticfiles import StaticFiles

from fastapi.responses import HTMLResponse, JSONResponse

from pydantic import BaseModel



from vanishield_engine import core_engine, JEV_SERVER_URL, PROJECT_ROOT

from ephemeral_buffer import ram_ring_buffer, hash_caller_identity



logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

logger = logging.getLogger("VaniShieldAPI")



# Bounded Worker Executor and Admission Control for CPU/GPU heavy analysis tasks (Issue #6, #37)

MAX_CONCURRENT_WORKERS = min(4, os.cpu_count() or 2)

ANALYSIS_EXECUTOR = ThreadPoolExecutor(max_workers=MAX_CONCURRENT_WORKERS)

UPLOAD_SEMAPHORE = asyncio.Semaphore(MAX_CONCURRENT_WORKERS)



# Guard limits against memory exhaustion (Issue #5)

MAX_UPLOAD_BYTES = 25 * 1024 * 1024               # 25 MB max upload

MAX_AUDIO_SECONDS = 180                           # 3 minutes max duration

MAX_DECODED_SAMPLES = MAX_AUDIO_SECONDS * 16000  # 2,880,000 samples @ 16kHz



app = FastAPI(

    title="VaniShield AI — Real-Time Voice Cloning Defense & Banking Circuit Breaker",

    description="Full-stack AI Audio Forensics, Jev-Omni Brain, and UPI Lockdown Server",

    version="2.0.0"

)
app.include_router(banking_router)




# Startup validation: Fail startup clearly if trained weights are missing (Issue #1)

@app.on_event("startup")

async def startup_check():

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



@app.get("/", response_class=HTMLResponse)

async def serve_index():

    index_path = os.path.join(STATIC_DIR, "index.html")

    if os.path.exists(index_path):

        with open(index_path, "r", encoding="utf-8") as f:

            return f.read()

    return "<h1>VaniShield AI Backend Online</h1>"



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



    wav, sr = torchaudio.load(path)

    if wav.shape[0] > 1:

        wav = wav.mean(dim=0, keepdim=True)

    if sr != 16000:

        wav = torchaudio.transforms.Resample(sr, 16000)(wav)

    audio_np = wav.squeeze(0).numpy()



    if len(audio_np) > MAX_DECODED_SAMPLES:

        audio_np = audio_np[:MAX_DECODED_SAMPLES]



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

    cb_engaged = bool(result.get("circuit_breaker", {}).get("engaged", False))
    if cb_engaged and not banking_state.locked:
        banking_state.locked = True
        banking_state.risk_score = result.get("unified_risk_score", 95.0)
        banking_state.reason = f"Automated Defense: {result.get('threat_type', 'Scam detected')}"

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

# WEBSOCKET: REAL-TIME STREAMING & EPHEMERAL RAM BUFFER

# -------------------------------------------------------------

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

    caller_phone = (

        websocket.query_params.get("phone")

        or websocket.query_params.get("caller_id")

        or f"+91-0000000000 (Live Call {session_id[-6:]})"

    )

    # Send initial SESSION_INIT handshake frame carrying circuit breaker state
    try:
        await websocket.send_text(json.dumps({
            "type": "SESSION_INIT",
            "session_id": session_id,
            "caller": {
                "number": caller_phone,
                "carrier": "VoIP / Cellular Gateway",
                "location": "National / Metro Node"
            },
            "circuit_breaker": banking_state.locked
        }))
    except Exception as e:
        logger.warning(f"Failed to send SESSION_INIT: {e}")



    loop = asyncio.get_running_loop()



    try:

        while True:

            # Receive binary PCM packet from client

            message = await websocket.receive()

            if "bytes" in message:

                pcm_bytes = message["bytes"]

                # 1. Push to Ephemeral RAM Ring Buffer (500ms sliding window, 0 disk I/O)

                ram_ring_buffer.push_chunk(session_id, pcm_bytes)



                # 2. Extract current evaluation window from RAM

                eval_audio = ram_ring_buffer.get_eval_window(session_id)



                if core_engine is not None:

                    # 3. Offload evaluation to worker pool (Issue #6) using connection-specific caller ID (Issue #3)

                    eval_result = await loop.run_in_executor(

                        ANALYSIS_EXECUTOR,

                        core_engine.evaluate_live_call,

                        eval_audio,

                        caller_phone

                    )

                else:

                    eval_result = {

                        "status": "error",

                        "error": "Core engine uninitialized (trained weights missing)."

                    }



                cb_engaged = bool(eval_result.get("circuit_breaker", {}).get("engaged", False))
                if cb_engaged and not banking_state.locked:
                    banking_state.locked = True
                    banking_state.risk_score = eval_result.get("unified_risk_score", 95.0)
                    banking_state.reason = f"Automated Defense: {eval_result.get('threat_type', 'Scam detected')}"

                # 4. Stream analysis back to client

                await websocket.send_json(eval_result)



            elif "text" in message:

                data = json.loads(message["text"])

                action = data.get("action")

                if action == "SIMULATE_SCAM_CALL":
                    banking_state.locked = True
                    banking_state.risk_score = 97.9
                    banking_state.reason = "Automated Defense: High-Risk AI Voice Clone & Financial Vishing Attack Intercepted."
                    await websocket.send_json({
                        "type": "TELEMETRY_FRAME",
                        "session_id": session_id,
                        "acoustic_clone": 99.4,
                        "intent_scam": 96.0,
                        "unified_risk": 97.9,
                        "threat_level": "CRITICAL SCAM (LOCKED)",
                        "circuit_breaker_locked": True,
                        "transcript": "Urgent security notice from State Bank of India fraud team. Share your 6-digit OTP and CVV immediately to block illegal debit.",
                        "caller_geo": {
                            "number": caller_phone,
                            "carrier": "Virtual VoIP Gateway (Flagged)",
                            "city": "Jamtara / Cyber Node",
                            "lat": 23.9632,
                            "lon": 86.8029,
                            "is_scam": True
                        }
                    })
                elif action == "SIMULATE_SAFE_CALL":
                    await websocket.send_json({
                        "type": "TELEMETRY_FRAME",
                        "session_id": session_id,
                        "acoustic_clone": 0.2,
                        "intent_scam": 0.0,
                        "unified_risk": 0.1,
                        "threat_level": "AUTHENTIC (SAFE)",
                        "circuit_breaker_locked": banking_state.locked,
                        "transcript": "Hi Rahul, are we still meeting for lunch today at 1 PM? Let me know when you reach.",
                        "caller_geo": {
                            "number": caller_phone,
                            "carrier": "Reliance Jio Infocomm",
                            "city": "Bengaluru Circle",
                            "lat": 12.9716,
                            "lon": 77.5946,
                            "is_scam": False
                        }
                    })
                elif action == "SET_CALLER":

                    new_phone = data.get("phone") or data.get("caller_id")

                    if new_phone:

                        caller_phone = str(new_phone).strip()

                        logger.info(f"Session {session_id} updated caller ID to: {caller_phone}")

                elif action == "END_CALL":

                    # Hard purge from volatile memory upon call completion

                    ram_ring_buffer.purge_session(session_id)

                    logger.info(f"Session {session_id} ended. Volatile RAM purged.")

                    break

    except WebSocketDisconnect:

        ram_ring_buffer.purge_session(session_id)

        logger.info(f"WebSocket client disconnected. Session {session_id} purged from RAM.")

    except Exception as e:

        ram_ring_buffer.purge_session(session_id)

        logger.error(f"WebSocket error: {e}")



# -------------------------------------------------------------

# HEALTH & BLACKLIST ENDPOINTS

# -------------------------------------------------------------

@app.get("/health")

async def health_check():

    return {

        "status": "healthy" if core_engine is not None else "degraded_missing_weights",

        "engine": "VaniShield AI Core v2.0",

        "device": core_engine.device if core_engine is not None else "unavailable",

        "cuda_available": torch.cuda.is_available(),

        "jev_server": JEV_SERVER_URL,

        "dpdp_privacy": "Ephemeral In-Process RAM Ring Buffer Active"

    }



@app.get("/api/blacklist")

async def get_blacklist():

    if core_engine is not None:

        return core_engine.blacklist

    return {}



# -------------------------------------------------------------

# SARVAM AI STT & ELEVENLABS TTS INTEGRATION ENDPOINTS

# -------------------------------------------------------------

class SynthesizeRequest(BaseModel):

    text: str

    voice_id: Optional[str] = None

    model_id: Optional[str] = "eleven_multilingual_v2"

    phone_number: Optional[str] = "+91-1409988776 (ElevenLabs Cloned Call)"

    use_known_transcript: Optional[bool] = False

    access_token: Optional[str] = None



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
_tts_timestamps = []  # global sliding-window timestamps
_tts_ip_timestamps: dict = {}  # per-IP sliding-window timestamps
MAX_TTS_PER_MINUTE = 10          # global cap: calls per minute
MAX_TTS_PER_IP_PER_MINUTE = 5   # per-source-IP cap: calls per minute


def _check_tts_access(request: Request, body_token: Optional[str] = None) -> None:
    """Verify authorization before allowing speech synthesis via paid ElevenLabs quota."""
    try:
        from elevenlabs_tts import is_elevenlabs_available
        has_elevenlabs = is_elevenlabs_available()
    except Exception:
        has_elevenlabs = bool(os.environ.get("ELEVENLABS_API_KEY"))

    expected = os.environ.get("TTS_ACCESS_TOKEN", "").strip()
    # When ElevenLabs is configured with an API key, authorization is mandatory to protect paid quota
    if has_elevenlabs and not expected:
        expected = os.environ.get("VANISHIELD_API_KEY", os.environ.get("VANISHIELD_BANKING_PIN", os.environ.get("VANISHIELD_UNLOCK_PIN", "1234"))).strip()

    if expected or has_elevenlabs:
        target_token = expected or "1234"
        provided = (
            request.headers.get("X-TTS-Token")
            or request.headers.get("Authorization", "").replace("Bearer ", "")
            or (body_token or "")
        ).strip()
        if not provided or not hmac.compare_digest(provided, target_token):
            raise HTTPException(
                status_code=401,
                detail="TTS endpoint requires authorization (valid X-TTS-Token header or access_token)."
            )


def _check_tts_budget_and_length(text: str, request: Request = None) -> None:
    """Enforce text-length bound and sliding-window rate limits (global + per-IP)."""
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
        # Global rate limit
        _tts_timestamps[:] = [t for t in _tts_timestamps if now - t < 60.0]
        if len(_tts_timestamps) >= MAX_TTS_PER_MINUTE:
            raise HTTPException(
                status_code=429,
                detail="TTS generation rate limit / budget quota reached. Please try again in 1 minute."
            )
        # Per-IP rate limit
        ip_ts = _tts_ip_timestamps.setdefault(client_ip, [])
        ip_ts[:] = [t for t in ip_ts if now - t < 60.0]
        if len(ip_ts) >= MAX_TTS_PER_IP_PER_MINUTE:
            raise HTTPException(
                status_code=429,
                detail=f"Per-client TTS rate limit reached. Maximum {MAX_TTS_PER_IP_PER_MINUTE} requests per minute."
            )
        _tts_timestamps.append(now)
        ip_ts.append(now)


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

async def tts_synthesize(req: SynthesizeRequest, request: Request):

    """Generates audio for given text via ElevenLabs API (or fallback) and streams audio/mpeg."""

    from elevenlabs_tts import synthesize_with_fallback

    from fastapi.responses import Response



    _check_tts_access(request, req.access_token)



    _check_tts_budget_and_length(req.text, request)
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

async def tts_simulate_attack(req: SynthesizeRequest, request: Request):

    """

    End-to-End Test: Synthesizes text via ElevenLabs TTS and immediately runs

    it through the full VaniShield 8-pillar forensic pipeline (including Sarvam STT & Wav2Vec 2.0).

    """

    import base64

    from elevenlabs_tts import synthesize_with_fallback



    _check_tts_access(request, req.access_token)



    _check_tts_budget_and_length(req.text, request)
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

