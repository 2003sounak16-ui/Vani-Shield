import urllib.request
import json
import time

presets = [
    ("en_safe", "English Safe Meeting"),
    ("en_scam", "English SBI Scam"),
    ("bn_safe", "Bengali Safe Family"),
    ("bn_scam", "Bengali Bank Trap"),
    ("hi_safe", "Hindi Safe Friend"),
    ("hi_scam", "Hindi HDFC Fraud")
]

print("=" * 95)
print(f"{'CALL PRESET':<24} | {'WAV2VEC2':<12} | {'JEV-OMNI':<12} | {'UNIFIED RISK':<14} | {'CIRCUIT BREAKER':<20} | {'LATENCY'}")
print("=" * 95)

for p_id, label in presets:
    url = f"http://127.0.0.1:8000/api/simulate/{p_id}"
    req = urllib.request.urlopen(url, timeout=30)
    data = json.loads(req.read().decode("utf-8"))

    w2v = f"{data['w2v_fake_prob']:.1f}% Fake"
    jev = f"{data['jev_scam_prob']:.1f}% Scam"
    risk = f"{data['unified_risk_score']:.1f}%"
    breaker = data['circuit_breaker']['action']
    lat = f"{data['latency_ms']:.1f} ms"

    print(f"{label:<24} | {w2v:<12} | {jev:<12} | {risk:<14} | {breaker:<20} | {lat}")

print("=" * 95)
