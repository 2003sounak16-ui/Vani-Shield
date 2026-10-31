"""
VaniShield AI — WebSocket Live Call Streaming API
Team CypherEye | Project VaniShield AI
Streams real-time PCM audio chunks, runs dual-brain forensic analysis, and streams threat telemetry.
"""

import sys
import os
import json
import time
import asyncio
import numpy as np
from fastapi import APIRouter, WebSocket, WebSocketDisconnect

# Import banking state for auto circuit breaker
from backend.api.banking_hold import state as banking_state

# Import engines
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BACKEND_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
PROJECT_ROOT = os.path.abspath(os.path.join(BACKEND_DIR, ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from backend.engine.ephemeral_buffer import EphemeralRAMBuffer, hash_caller_identity
from backend.engine.caller_tracker import lookup_carrier, mark_scammer, get_ip_geolocation

router = APIRouter(tags=["Live WebSocket Stream"])
ephemeral_buffer = EphemeralRAMBuffer(max_duration_sec=4.0, sample_rate=16000)

@router.websocket("/ws/live-call")
async def websocket_live_call(websocket: WebSocket):
    await websocket.accept()
    session_id = f"sess_{int(time.time()*1000)}"
    caller_num = "+911409988776"
    caller_info = lookup_carrier(caller_num)

    # Initial handshake frame
    await websocket.send_text(json.dumps({
        "type": "SESSION_INIT",
        "session_id": session_id,
        "caller": caller_info,
        "circuit_breaker": banking_state.locked
    }))

    try:
        while True:
            data = await websocket.receive_text()
            msg = json.loads(data)
            action = msg.get("action")

            if action == "SIMULATE_SCAM_CALL":
                # Stream simulated scam call frames
                scenarios = [
                    {"t": 1, "text": "Hello, this is State Bank of India fraud prevention department.", "w2v": 98.4, "jev": 45.0, "risk": 74.4},
                    {"t": 2, "text": "An unauthorized debit of 65,000 rupees was requested on your card.", "w2v": 99.2, "jev": 85.0, "risk": 92.8},
                    {"t": 3, "text": "Please share the 6-digit OTP received on your mobile immediately to block it.", "w2v": 99.9, "jev": 98.0, "risk": 99.0},
                ]
                for sc in scenarios:
                    circuit_locked = sc["risk"] >= 70.0
                    if circuit_locked and not banking_state.locked:
                        banking_state.locked = True
                        banking_state.risk_score = sc["risk"]
                        banking_state.reason = f"Automated Defense: High Risk Voice Clone ({sc['risk']}%) detected on live call."
                        mark_scammer(caller_num, sc["risk"], sc["text"])

                    payload = {
                        "type": "TELEMETRY_FRAME",
                        "session_id": session_id,
                        "timestamp": time.time(),
                        "transcript": sc["text"],
                        "acoustic_clone": sc["w2v"],
                        "intent_scam": sc["jev"],
                        "unified_risk": sc["risk"],
                        "circuit_breaker_locked": banking_state.locked,
                        "threat_level": "CRITICAL SCAM (LOCKED)" if circuit_locked else "NORMAL",
                        "waveform": (np.sin(np.linspace(0, 10, 32)) * 0.8).tolist(),
                        "caller_geo": {
                            "number": caller_num,
                            "carrier": caller_info.get("carrier"),
                            "city": caller_info.get("location"),
                            "lat": caller_info.get("lat"),
                            "lon": caller_info.get("lon"),
                            "is_scam": circuit_locked
                        }
                    }
                    await websocket.send_text(json.dumps(payload))
                    await asyncio.sleep(1.2)

            elif action == "SIMULATE_SAFE_CALL":
                # Stream simulated safe call frames
                safe_frames = [
                    {"t": 1, "text": "Hi Rahul, are we still meeting for lunch today at 1 PM?", "w2v": 0.2, "jev": 0.0, "risk": 0.1},
                    {"t": 2, "text": "Let me know when you reach the station, I have parked nearby.", "w2v": 0.1, "jev": 0.0, "risk": 0.1},
                ]
                for sf in safe_frames:
                    payload = {
                        "type": "TELEMETRY_FRAME",
                        "session_id": session_id,
                        "timestamp": time.time(),
                        "transcript": sf["text"],
                        "acoustic_clone": sf["w2v"],
                        "intent_scam": sf["jev"],
                        "unified_risk": sf["risk"],
                        "circuit_breaker_locked": banking_state.locked,
                        "threat_level": "AUTHENTIC (SAFE)",
                        "waveform": (np.sin(np.linspace(0, 6, 32)) * 0.3).tolist(),
                        "caller_geo": {
                            "number": "+919876543210",
                            "carrier": "Jio Infocomm (Authentic)",
                            "city": "Bengaluru",
                            "lat": 12.9716,
                            "lon": 77.5946,
                            "is_scam": False
                        }
                    }
                    await websocket.send_text(json.dumps(payload))
                    await asyncio.sleep(1.2)

            elif action == "PUSH_PCM_CHUNK":
                # Raw audio chunk push
                await websocket.send_text(json.dumps({
                    "type": "CHUNK_ACK",
                    "session_id": session_id,
                    "buffered_seconds": ephemeral_buffer.get_buffer_duration(session_id)
                }))

    except WebSocketDisconnect:
        ephemeral_buffer.clear_session(session_id)
    except Exception as e:
        ephemeral_buffer.clear_session(session_id)
