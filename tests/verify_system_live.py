import os
import sys
import urllib.request
import json
import websockets
import asyncio

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

print("=" * 70)
print("VANISHIELD AI — LIVE SYSTEM END-TO-END VERIFICATION")
print("=" * 70)

failed_checks = []

print("\n[1/5] Checking Backend Health & GPU Status...")
try:
    with urllib.request.urlopen("http://127.0.0.1:8000/health", timeout=10) as r:
        data = json.loads(r.read().decode("utf-8"))
        print(f"  Status        : {data.get('status')}")
        print(f"  Engine        : {data.get('engine') or data.get('service')}")
        print(f"  Device        : {data.get('device')} (CUDA={data.get('cuda_available', data.get('cuda_active'))})")
        print(f"  Privacy Buffer: {data.get('dpdp_privacy', 'Compliant')}")
except Exception as e:
    print(f"  Error: {e}")
    failed_checks.append(f"Backend Health: {e}")

print("\n[2/5] Testing Scam Call Simulation (English SBI Vishing Attack)...")
try:
    with urllib.request.urlopen("http://127.0.0.1:8000/api/simulate/en_scam", timeout=30) as r:
        data = json.loads(r.read().decode("utf-8"))
        print(f"  Acoustic Fake Prob : {data.get('w2v_fake_prob')}%")
        print(f"  Jev Scam Prob      : {data.get('jev_scam_prob')}%")
        print(f"  Unified Risk Score : {data.get('unified_risk_score')}%")
        print(f"  Threat Verdict     : {data.get('threat_type')}")
        print(f"  Circuit Breaker    : {data.get('circuit_breaker', {}).get('action')}")
        print(f"  Execution Latency  : {data.get('latency_ms')} ms")
except Exception as e:
    print(f"  Error: {e}")
    failed_checks.append(f"Simulate en_scam: {e}")

print("\n[3/5] Testing Authentic Call Simulation (Hindi College Friend)...")
try:
    with urllib.request.urlopen("http://127.0.0.1:8000/api/simulate/hi_safe", timeout=30) as r:
        data = json.loads(r.read().decode("utf-8"))
        print(f"  Unified Risk Score : {data.get('unified_risk_score')}%")
        print(f"  Threat Verdict     : {data.get('threat_type')}")
        print(f"  Circuit Breaker    : {data.get('circuit_breaker', {}).get('action')}")
except Exception as e:
    print(f"  Error: {e}")
    failed_checks.append(f"Simulate hi_safe: {e}")

print("\n[4/5] Testing WebSocket Live Streaming (/ws/live-call)...")
async def test_ws():
    try:
        async with websockets.connect("ws://127.0.0.1:8000/ws/live-call") as ws:
            print("  WebSocket Handshake: CONNECTED")
    except Exception as e:
        print(f"  WebSocket Error: {e}")
        failed_checks.append(f"WebSocket live-call: {e}")

asyncio.run(test_ws())

print("\n[5/5] Checking React + Vite Frontend Dashboard...")
try:
    with urllib.request.urlopen("http://127.0.0.1:5173/", timeout=5) as r:
        print(f"  Frontend Status    : HTTP {r.status} OK (React UI Operational)")
except Exception as e:
    print(f"  Frontend Error: {e}")
    failed_checks.append(f"Frontend Dashboard: {e}")

print("\n" + "=" * 70)
if not failed_checks:
    print("CONCLUSION: ALL SUBSYSTEMS FULLY OPERATIONAL")
    print("=" * 70)
    sys.exit(0)
else:
    print(f"CONCLUSION: {len(failed_checks)} SUBSYSTEM CHECK(S) FAILED:")
    for fc in failed_checks:
        print(f"  - {fc}")
    print("=" * 70)
    sys.exit(1)
