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
from fastapi import FastAPI, File, UploadFile, WebSocket, WebSocketDisconnect, HTTPException, Form
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
        "audio_file": os.path.join(PROJECT_ROOT, "test_multilingual_calls", "call_en_safe.wav"),
        "phone": "+91-9830112233 (Bengaluru, Karnataka - Airtel)",
        "transcript": "Hi team, let's reschedule our design review meeting to Thursday afternoon at 3 PM. Let me know if that works."
    },
    "en_scam": {
        "audio_file": os.path.join(PROJECT_ROOT, "test_multilingual_calls", "call_en_scam.wav"),
        "phone": "+91-1409988776 (VoIP Spoofed - International Gateway)",
        "transcript": "Urgent security notice from State Bank of India fraud team. Your account has an illegal debit of 65,000 rupees. Share your 6-digit OTP and CVV right now to reverse it."
    },
    "bn_safe": {
        "audio_file": os.path.join(PROJECT_ROOT, "test_multilingual_calls", "call_bn_safe.wav"),
        "phone": "+91-9433098765 (Kolkata, West Bengal - BSNL)",
        "transcript": "নমস্কার কাকিমা, আমি শুভঙ্কর বলছি। কাল বিকেলে আপনাদের বাড়ি আসব, কেমন আছেন সবাই?"
    },
    "bn_scam": {
        "audio_file": os.path.join(PROJECT_ROOT, "test_multilingual_calls", "call_bn_scam.wav"),
        "phone": "+91-1401122334 (Spoofed Virtual PBX)",
        "transcript": "জরুরি ব্যাংক সতর্কতা: আপনার স্টেট ব্যাংক একাউন্ট স্থগিত করা হয়েছে। অবিলম্বে ৬ সংখ্যার ওটিপি ও ইউপিআই পিন প্রদান করুন অন্যথায় অ্যাকাউন্ট স্থায়ীভাবে বন্ধ হবে।"
    },
    "hi_safe": {
        "audio_file": os.path.join(PROJECT_ROOT, "test_multilingual_calls", "call_hi_safe.wav"),
        "phone": "+91-9811234567 (New Delhi - Jio)",
        "transcript": "नमस्ते भाई, कल कॉलेज की क्लास सुबह नौ बजे शुरू होगी। असाइनमेंट पूरा हो गया क्या?"
    },
    "hi_scam": {
        "audio_file": os.path.join(PROJECT_ROOT, "test_multilingual_calls", "call_hi_scam.wav"),
        "phone": "+91-1407766554 (Unregistered VoIP Line)",
        "transcript": "नमस्कार, एचडीएफसी बैंक हेड ऑफिस। आपके पैन कार्ड में विसंगति पाई गई है। तुरंत अपना एटीएम पिन, सीवीवी और ओटीपी शेयर करके पैन री-वेरीफाई करें।"
    }
}

def _sync_simulate_preset(preset: dict) -> dict:
    path = preset["audio_file"]
    if not os.path.isabs(path):
        path = os.path.join(PROJECT_ROOT, path)
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
    # Default to an inbound caller identity specific to this session, allowing client overrides
    caller_phone = (
        websocket.query_params.get("phone")
        or websocket.query_params.get("caller_id")
        or f"+91-0000000000 (Live Call {session_id[-6:]})"
    )

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

                # 4. Stream analysis back to client
                await websocket.send_json(eval_result)

            elif "text" in message:
                data = json.loads(message["text"])
                action = data.get("action")
                if action == "SET_CALLER":
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

if __name__ == "__main__":
    import uvicorn
    print("\n" + "=" * 70)
    print("VaniShield AI: Starting Unified FastAPI Backend & Web Dashboard")
    print("Dashboard URL: http://127.0.0.1:8000")
    print("WebSocket URL: ws://127.0.0.1:8000/ws/live-call")
    print("=" * 70 + "\n")
    uvicorn.run(app, host="127.0.0.1", port=8000, log_level="info")
