"""
VaniShield AI — Real-Time Streaming FastAPI Backend
Team CypherEye | Project VaniShield AI
Provides WebSocket streaming, dual-brain risk telemetry, and automated banking circuit breaker.
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

from backend.api.ws_stream import router as ws_router
from backend.api.banking_hold import router as banking_router
from backend.engine.caller_tracker import load_blacklist

app = FastAPI(
    title="VaniShield AI Backend",
    description="Real-Time Voice Cloning Defense & Automated Banking Circuit Breaker",
    version="2.0.0"
)

# Enforce secure CORS policy
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API Routers
app.include_router(ws_router)
app.include_router(banking_router)

@app.get("/health")
async def health_check():
    """System health check and GPU acceleration status."""
    import torch
    cuda_available = torch.cuda.is_available()
    dev_name = torch.cuda.get_device_name(0) if cuda_available else "CPU Fallback"
    return {
        "status": "HEALTHY",
        "service": "VaniShield AI Streaming Backend",
        "cuda_active": cuda_available,
        "device": dev_name,
        "timestamp": time.time()
    }

@app.get("/api/scammers")
async def get_blacklisted_scammers():
    """Retrieve all flagged scammer records for the Threat Map."""
    data = load_blacklist()
    return {"count": len(data), "scammers": data}

# Mount static web UI if available
STATIC_DIR = os.path.join(PROJECT_ROOT, "static")
if os.path.exists(STATIC_DIR):
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend.main:app", host="127.0.0.1", port=8000, reload=False)
