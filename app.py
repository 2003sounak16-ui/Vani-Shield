"""
VaniShield AI — Root Entry Point for Dashboard & API Server
Forwards to backend.app_demo.
"""
import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from backend.app_demo import app

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend.app_demo:app", host="127.0.0.1", port=8000, reload=False)
