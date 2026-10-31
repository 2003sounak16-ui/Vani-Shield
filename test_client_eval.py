import sys
import requests
import json

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

test_files = [
    ("Real Friend (REAL)", "data_demo/unseen_test/unseen_real_friend.wav"),
    ("Scam HDFC (FAKE)", "data_demo/unseen_test/unseen_scam_hdfc.wav"),
    ("Fake SBI OTP (FAKE)", "data_demo/fake/scam_sbi_otp.wav"),
    ("BN Safe (REAL)", "test_multilingual_calls/call_bn_safe.wav"),
    ("BN Scam (FAKE)", "test_multilingual_calls/call_bn_scam.wav"),
    ("EN Safe (REAL)", "test_multilingual_calls/call_en_safe.wav"),
    ("EN Scam (FAKE)", "test_multilingual_calls/call_en_scam.wav"),
    ("User Voice (REAL)", r"C:\Users\souna\.gemini\antigravity\brain\66a12257-e1a0-45be-b045-dd87c5fd01de\.user_uploaded\uploaded_media_1793365290300.m4a"),
]

print(f"{'Label':22} | {'W2V Fake':>9} | {'Jev Scam':>9} | {'Risk':>6} | {'Locked':>6} | {'Keywords':>15} | Transcript")
print("-" * 105)

for label, p in test_files:
    with open(p, "rb") as f:
        resp = requests.post(
            "http://127.0.0.1:8000/api/evaluate/upload",
            files={"file": f},
            data={"phone_number": "+91-9988776655"}
        )
    res = resp.json()
    w2v = res.get("w2v_fake_prob", 0)
    jev = res.get("jev_scam_prob", 0)
    risk = res.get("unified_risk_score", 0)
    breaker = res.get("circuit_breaker", {}).get("engaged", False)
    trans = res.get("transcript", "")[:35].replace("\n", " ")
    kw = res.get("flagged_keywords", [])
    print(f"{label:22} | {w2v:8.1f}% | {jev:8.1f}% | {risk:5.1f}% | {str(breaker):>6} | {str(kw):>15} | {trans}")
