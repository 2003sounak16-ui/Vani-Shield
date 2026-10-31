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

# Load root .env before importing routers or reading environment variables
_root_env = os.path.join(PROJECT_ROOT, ".env")
try:
    import dotenv
    dotenv.load_dotenv(_root_env)
except Exception:
    if os.path.exists(_root_env):
        try:
            with open(_root_env, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith("#") and "=" in line:
                        k, v = line.split("=", 1)
                        k = k.strip()
                        # Safely strip inline comments and quotes
                        v = v.split("#")[0].strip().strip("'\"")
                        if k and k not in os.environ:
                            os.environ[k] = v
        except Exception:
            pass

from backend.api.evaluate import router as evaluate_router
from backend.api.ws_stream import router as ws_router
from backend.api.banking_hold import router as banking_router
from backend.engine.caller_tracker import load_blacklist
from backend.engine.mobile_edge_engine import mobile_engine

app = FastAPI(
    title="VaniShield AI Backend",
    description="Real-Time Voice Cloning Defense & Automated Banking Circuit Breaker",
    version="2.0.0"
)

# Enforce secure CORS allowlist (Issue #9 & Cloud Deployments)
configured_origins = os.environ.get("VANISHIELD_ALLOWED_ORIGINS", "").strip()
if configured_origins and configured_origins != "*":
    allowed_origins = [orig.strip() for orig in configured_origins.split(",") if orig.strip()]
    allow_credentials = True
else:
    # Allow all origins for seamless cloud PaaS deployments
    allowed_origins = ["*"]
    allow_credentials = False

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=allow_credentials,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API Routers (Issue #3)
app.include_router(evaluate_router)
app.include_router(ws_router)
app.include_router(banking_router)

# Mount audio evaluation, simulation, and speech routes from backend.app_demo
try:
    from backend.app_demo import app as demo_app
    existing_paths = {r.path for r in app.routes}
    for r in demo_app.routes:
        if r.path not in existing_paths and r.path not in ["/openapi.json", "/docs", "/redoc"]:
            app.routes.append(r)
except Exception:
    pass

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

@app.get("/api/engine/mobile-status")
async def get_mobile_engine_status():
    """Returns Mobile Edge Engine status, active ONNX provider, and memory footprint."""
    model_loaded = mobile_engine.ort_session is not None
    return {
        "mode": "mobile_edge_optimized",
        "onnx_model_loaded": model_loaded,
        "active_provider": mobile_engine.active_provider,
        "model_file": os.path.basename(mobile_engine.model_path) if mobile_engine.model_path else "None",
        "quantization": "INT8" if mobile_engine.model_path and "int8" in mobile_engine.model_path else "FP32",
        "estimated_ram_mb": 91.0 if model_loaded else 15.0,
        "target_platforms": ["Android (NNAPI / Qualcomm QNN)", "iOS (CoreML)", "Edge Linux / ARM64"]
    }

# Mount static web UI if available
STATIC_DIR = os.path.join(PROJECT_ROOT, "static")
if os.path.exists(STATIC_DIR):
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

# Mount React UI if built
FRONTEND_DIST = os.path.join(PROJECT_ROOT, "frontend", "dist")
if os.path.exists(FRONTEND_DIST):
    assets_dir = os.path.join(FRONTEND_DIST, "assets")
    if os.path.exists(assets_dir):
        app.mount("/assets", StaticFiles(directory=assets_dir), name="react-assets")

    @app.get("/react", response_class=HTMLResponse)
    async def serve_react_dashboard():
        react_index = os.path.join(FRONTEND_DIST, "index.html")
        if os.path.exists(react_index):
            with open(react_index, "r", encoding="utf-8") as f:
                return f.read()
        return "<h1>React Dashboard Not Found</h1>"

if __name__ == "__main__":
    import uvicorn
    server_port = int(os.environ.get("PORT", 8000))
    uvicorn.run("backend.main:app", host="0.0.0.0", port=server_port, reload=False)
