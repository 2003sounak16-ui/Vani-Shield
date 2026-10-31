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
import asyncio
import logging
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

from vanishield_engine import core_engine, JEV_SERVER_URL
from ephemeral_buffer import ram_ring_buffer, hash_caller_identity

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("VaniShieldAPI")

app = FastAPI(
    title="VaniShield AI — Real-Time Voice Cloning Defense & Banking Circuit Breaker",
    description="Full-stack AI Audio Forensics, Jev-Omni Brain, and UPI Lockdown Server",
    version="2.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount Static Files for 3-Panel Split-Screen Frontend
STATIC_DIR = os.path.join(os.path.dirname(__file__), "static")
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
# -------------------------------------------------------------
PRESET_CALLS = {
    "en_safe": {
        "audio_file": "data_real/real/en/en_real_000.wav",
        "phone": "+91-9830112233 (Bengaluru, Karnataka - Airtel)",
        "transcript": "Hi team, let's reschedule our design review meeting to Thursday afternoon at 3 PM. Let me know if that works."
    },
    "en_scam": {
        "audio_file": "test_multilingual_calls/scam_en.wav",
        "phone": "+91-1409988776 (VoIP Spoofed - International Gateway)",
        "transcript": "Urgent security notice from State Bank of India fraud team. Your account has an illegal debit of 65,000 rupees. Share your 6-digit OTP and CVV right now to reverse it."
    },
    "bn_safe": {
        "audio_file": "data_real/real/bn/bn_real_000_bin_9169_8109561360.wav",
        "phone": "+91-9433098765 (Kolkata, West Bengal - BSNL)",
        "transcript": "নমস্কার কাকিমা, আমি শুভঙ্কর বলছি। কাল বিকেলে আপনাদের বাড়ি আসব, কেমন আছেন সবাই?"
    },
    "bn_scam": {
        "audio_file": "test_multilingual_calls/scam_bn.wav",
        "phone": "+91-1401122334 (Spoofed Virtual PBX)",
        "transcript": "জরুরি ব্যাংক সতর্কতা: আপনার স্টেট ব্যাংক একাউন্ট স্থগিত করা হয়েছে। অবিলম্বে ৬ সংখ্যার ওটিপি ও ইউপিআই পিন প্রদান করুন অন্যথায় অ্যাকাউন্ট স্থায়ীভাবে বন্ধ হবে।"
    },
    "hi_safe": {
        "audio_file": "data_real/real/hi/hi_real_000_2131_098.wav",
        "phone": "+91-9811234567 (New Delhi - Jio)",
        "transcript": "नमस्ते भाई, कल कॉलेज की क्लास सुबह नौ बजे शुरू होगी। असाइनमेंट पूरा हो गया क्या?"
    },
    "hi_scam": {
        "audio_file": "test_multilingual_calls/scam_hi.wav",
        "phone": "+91-1407766554 (Unregistered VoIP Line)",
        "transcript": "नमस्कार, एचडीएफसी बैंक हेड ऑफिस। आपके पैन कार्ड में विसंगति पाई गई है। तुरंत अपना एटीएम पिन, सीवीवी और ओटीपी शेयर करके पैन री-वेरीफाई करें।"
    }
}

@app.get("/api/simulate/{preset_id}")
async def simulate_preset(preset_id: str):
    preset = PRESET_CALLS.get(preset_id)
    if not preset:
        raise HTTPException(status_code=404, detail="Unknown preset ID")

    # Load audio into memory
    path = preset["audio_file"]
    if not os.path.exists(path):
        # Fallback to any wav if specific path moved
        any_wavs = glob.glob("data_real/**/*.wav", recursive=True)
        path = any_wavs[0] if any_wavs else None

    wav, sr = torchaudio.load(path)
    if wav.shape[0] > 1:
        wav = wav.mean(dim=0, keepdim=True)
    if sr != 16000:
        wav = torchaudio.transforms.Resample(sr, 16000)(wav)
    audio_np = wav.squeeze(0).numpy()

    # Pass to unified core engine
    result = core_engine.evaluate_live_call(
        audio_chunk_16k=audio_np,
        phone_number=preset["phone"],
        known_transcript=preset["transcript"]
    )
    return result

def decode_audio_bytes(content: bytes) -> np.ndarray:
    """Decodes any audio container (m4a, mp3, wav, flac, ogg, webm) in volatile RAM."""
    try:
        import av
        container = av.open(io.BytesIO(content))
        resampler = av.AudioResampler(format="fltp", layout="mono", rate=16000)
        frames = []
        for frame in container.decode(audio=0):
            for rf in resampler.resample(frame):
                frames.append(rf.to_ndarray())
        if frames:
            arr = np.concatenate(frames, axis=1).squeeze()
            if arr.ndim > 1:
                arr = arr.mean(axis=0)
            arr = np.clip(arr, -1.0, 1.0)
            return arr.astype(np.float32)
    except Exception as e:
        logger.warning(f"PyAV decode fallback triggered: {e}")

    try:
        data, sr = sf.read(io.BytesIO(content))
        if data.ndim > 1:
            data = data.mean(axis=1)
        if sr != 16000:
            import librosa
            data = librosa.resample(data, orig_sr=sr, target_sr=16000)
        return data.astype(np.float32)
    except Exception as e:
        logger.error(f"Audio decode failed: {e}")
        raise HTTPException(status_code=400, detail=f"Unsupported or unreadable audio format: {e}")

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
    """Processes dropped/uploaded .m4a/.wav/.mp3 file in volatile RAM with zero disk writes."""
    logger.info(f"Received audio file upload: {file.filename} (content_type: {file.content_type})")
    content = await file.read()
    if not content:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    audio_np = decode_audio_bytes(content)

    clean_caller = phone_number.strip() if phone_number else "Direct Audio File Upload"
    result = core_engine.evaluate_live_call(
        audio_chunk_16k=audio_np,
        phone_number=clean_caller,
        known_transcript=transcript,
        is_upload=True
    )
    result["filename"] = file.filename
    result["duration_seconds"] = round(len(audio_np) / 16000.0, 2)
    return result

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
    session_id = f"call_{int(time.time()*1000)}"
    logger.info(f"WebSocket client connected. Session ID: {session_id}")

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

                # 3. Evaluate with unified core engine
                eval_result = core_engine.evaluate_live_call(
                    audio_chunk_16k=eval_audio,
                    phone_number="+91-1409988776 (VoIP Spoofed Gateway)"
                )

                # 4. Stream analysis back to client
                await websocket.send_json(eval_result)

            elif "text" in message:
                data = json.loads(message["text"])
                if data.get("action") == "END_CALL":
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
        "status": "healthy",
        "engine": "VaniShield AI Core v2.0",
        "device": core_engine.device,
        "cuda_available": torch.cuda.is_available(),
        "jev_server": JEV_SERVER_URL,
        "dpdp_privacy": "Ephemeral In-Process RAM Ring Buffer Active"
    }

@app.get("/api/blacklist")
async def get_blacklist():
    return core_engine.blacklist

if __name__ == "__main__":
    import uvicorn
    print("\n" + "=" * 70)
    print("VaniShield AI: Starting Unified FastAPI Backend & Web Dashboard")
    print("Dashboard URL: http://127.0.0.1:8000")
    print("WebSocket URL: ws://127.0.0.1:8000/ws/live-call")
    print("=" * 70 + "\n")
    uvicorn.run(app, host="127.0.0.1", port=8000, log_level="info")
