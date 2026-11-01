"""
VaniShield AI — WebSocket Live Call Streaming API
Team CypherEye | Project VaniShield AI
Streams real-time PCM audio chunks, runs dual-brain forensic analysis, and streams threat telemetry.
"""

import sys
import os
import json
import time
import base64
import asyncio
import numpy as np
import logging
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

try:
    from vanishield_engine import core_engine
    _ENGINE_SOURCE = "vanishield_engine (root)"
except ImportError as _root_err:
    # Loud, not silent. These two modules have diverged: the root engine
    # carries the social-engineering stage pillar, the fusion and demo mode;
    # backend/engine/vanishield_engine.py carries none of them. Falling back
    # therefore does not degrade gracefully - it swaps the detector for one
    # that cannot detect a scam script at all. A deployment must never do
    # that quietly, so the reason is logged at error level and the active
    # source is reported on /health.
    logging.getLogger(__name__).error(
        "Root vanishield_engine failed to import (%s); falling back to "
        "backend.engine.vanishield_engine, which has NO scam-stage pillar. "
        "Scam-script detection is unavailable in this process.",
        _root_err,
    )
    try:
        from backend.engine.vanishield_engine import core_engine
        _ENGINE_SOURCE = "backend.engine.vanishield_engine (FALLBACK - no stage pillar)"
    except ImportError:
        core_engine = None
        _ENGINE_SOURCE = "unavailable"

router = APIRouter(tags=["Live WebSocket Stream"])
ephemeral_buffer = EphemeralRAMBuffer(max_duration_sec=4.0, sample_rate=16000)

@router.websocket("/ws/live-call")
async def websocket_live_call(websocket: WebSocket):
    await websocket.accept()
    session_id = f"sess_{int(time.time()*1000)}"
    caller_num = "+911409988776"
    caller_info = lookup_carrier(caller_num)
    loop = asyncio.get_running_loop()

    # Initial handshake frame
    await websocket.send_text(json.dumps({
        "type": "SESSION_INIT",
        "session_id": session_id,
        "caller": caller_info,
        "circuit_breaker": banking_state.locked,
        "circuit_breaker_locked": banking_state.locked,
        "locked": banking_state.locked,
        "reason": banking_state.reason,
        "risk_score": banking_state.risk_score
    }))

    try:
        while True:
            message = await websocket.receive()

            # 1. Binary audio frame handling
            if "bytes" in message and message["bytes"]:
                pcm_bytes = message["bytes"]
                ephemeral_buffer.push_chunk(session_id, pcm_bytes)
                eval_audio = ephemeral_buffer.get_eval_window(session_id)

                if core_engine is not None:
                    eval_result = await loop.run_in_executor(
                        None, core_engine.evaluate_live_call, eval_audio, caller_num
                    )
                    cb_engaged = bool(eval_result.get("circuit_breaker", {}).get("engaged", False))
                    if cb_engaged and not banking_state.locked:
                        banking_state.locked = True
                        banking_state.risk_score = eval_result.get("unified_risk_score", 95.0)
                        banking_state.reason = f"Automated Defense: {eval_result.get('threat_type', 'Scam detected')}"
                        mark_scammer(caller_num, banking_state.risk_score, "High-Risk AI Voice Clone & Social Engineering")

                    payload = {
                        "type": "TELEMETRY_FRAME",
                        "session_id": session_id,
                        "timestamp": time.time(),
                        "transcript": eval_result.get("transcript", ""),
                        "acoustic_clone": eval_result.get("w2v_fake_prob", 0.0),
                        "intent_scam": eval_result.get("jev_scam_prob", 0.0),
                        "unified_risk": eval_result.get("unified_risk_score", 0.0),
                        "circuit_breaker_locked": banking_state.locked,
                        "threat_level": "CRITICAL SCAM (LOCKED)" if banking_state.locked else "AUTHENTIC (SAFE)",
                        "waveform": (eval_audio[:32] if len(eval_audio) >= 32 else np.zeros(32)).tolist(),
                        "caller_geo": {
                            "number": caller_num,
                            "carrier": caller_info.get("carrier"),
                            "city": caller_info.get("location"),
                            "lat": caller_info.get("lat"),
                            "lon": caller_info.get("lon"),
                            "is_scam": banking_state.locked
                        }
                    }
                    await websocket.send_text(json.dumps(payload))
                else:
                    await websocket.send_text(json.dumps({
                        "type": "CHUNK_ACK",
                        "session_id": session_id,
                        "buffered_seconds": ephemeral_buffer.get_buffer_duration(session_id)
                    }))
                continue

            # 2. Text / JSON frame handling
            if "text" in message and message["text"]:
                data = json.loads(message["text"])
                action = data.get("action")

                if action == "SIMULATE_SCAM_CALL":
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
                            mark_scammer(caller_num, sc["risk"], "Simulated Banking / OTP Vishing Attack")

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
                    # Check if chunk includes raw base64 PCM
                    if "pcm_base64" in data:
                        raw_pcm = base64.b64decode(data["pcm_base64"])
                        ephemeral_buffer.push_chunk(session_id, raw_pcm)
                        if core_engine is not None:
                            eval_audio = ephemeral_buffer.get_eval_window(session_id)
                            eval_res = await loop.run_in_executor(
                                None, core_engine.evaluate_live_call, eval_audio, caller_num
                            )
                            if eval_res.get("circuit_breaker", {}).get("engaged"):
                                banking_state.locked = True
                                banking_state.risk_score = eval_res.get("unified_risk_score", 95.0)
                                banking_state.reason = eval_res.get("threat_type", "Scam detected")
                                mark_scammer(caller_num, banking_state.risk_score, eval_res.get("threat_type", "Scam detected"))

                    duration = ephemeral_buffer.get_buffer_duration(session_id)
                    await websocket.send_text(json.dumps({
                        "type": "CHUNK_ACK",
                        "session_id": session_id,
                        "buffered_seconds": duration
                    }))

    except WebSocketDisconnect:
        ephemeral_buffer.clear_session(session_id)
    except Exception as e:
        ephemeral_buffer.clear_session(session_id)
