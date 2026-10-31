"""
VaniShield AI — Real-Time Streaming FastAPI Backend
Team CypherEye | Project VaniShield AI
Provides WebSocket streaming, dual-brain risk telemetry, preset simulations, upload analysis, and automated banking circuit breaker.
"""

import os
import sys
import time
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

# Setup paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from backend.api.evaluate import router as evaluate_router
from backend.api.ws_stream import router as ws_router
from backend.api.banking_hold import router as banking_router
from backend.engine.caller_tracker import load_blacklist

app = FastAPI(
    title="VaniShield AI Backend",
    description="Real-Time Voice Cloning Defense & Automated Banking Circuit Breaker",
    version="2.0.0"
)

# Enforce secure CORS allowlist (Issue #9)
configured_origins = os.environ.get("VANISHIELD_ALLOWED_ORIGINS", "").split(",")
allowed_origins = [orig.strip() for orig in configured_origins if orig.strip()] or [
    "http://localhost:8000",
    "http://127.0.0.1:8000",
    "http://localhost:5173",
    "http://127.0.0.1:5173",
    "http://localhost:3000",
    "http://127.0.0.1:3000",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)

# Include API Routers (Issue #3)
app.include_router(evaluate_router)
app.include_router(ws_router)
app.include_router(banking_router)

@app.get("/health")
async def health_check():
    """System health check and GPU acceleration status."""
    try:
        import torch
        cuda_available = torch.cuda.is_available()
        dev_name = torch.cuda.get_device_name(0) if cuda_available else "CPU Fallback"
    except Exception:
        cuda_available = False
        dev_name = "CPU Fallback"
    return {
        "status": "HEALTHY",
        "service": "VaniShield AI Streaming Backend",
        "cuda_active": cuda_available,
        "device": dev_name,
        "timestamp": time.time()
    }

@app.get("/api/scammers")
async def get_blacklisted_scammers():
    """Retrieve all flagged scammer records for the Threat Map with DPDP-compliant masking."""
    raw_data = load_blacklist()
    # Mask raw phone numbers to prevent privacy leakage
    sanitized = {}
    for phone, record in raw_data.items():
        if len(phone) >= 8:
            masked_key = phone[:4] + "****" + phone[-4:]
        else:
            masked_key = "ANONYMIZED_RECORD"
        entry = dict(record)
        entry["phone_masked"] = masked_key
        sanitized[masked_key] = entry
    return {"count": len(sanitized), "scammers": sanitized}

# Mount static web UI if available
STATIC_DIR = os.path.join(PROJECT_ROOT, "static")
if os.path.exists(STATIC_DIR):
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend.main:app", host="127.0.0.1", port=8000, reload=False)
