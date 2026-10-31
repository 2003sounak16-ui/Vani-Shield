import os
import sys
import requests
import json

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
url = os.environ.get("VANISHIELD_EVAL_URL", "http://127.0.0.1:8000/api/evaluate/upload")
REQUEST_TIMEOUT = (5.0, 30.0)

test_files = [
    ("Real Friend (REAL)", os.path.join(PROJECT_ROOT, "data_demo/unseen_test/unseen_real_friend.wav")),
    ("Scam HDFC (FAKE)", os.path.join(PROJECT_ROOT, "data_demo/unseen_test/unseen_scam_hdfc.wav")),
    ("Fake SBI OTP (FAKE)", os.path.join(PROJECT_ROOT, "data_demo/fake/scam_sbi_otp.wav")),
    ("BN Safe (REAL)", os.path.join(PROJECT_ROOT, "demo_samples/real/call_bn_safe.wav")),
    ("BN Scam (FAKE)", os.path.join(PROJECT_ROOT, "demo_samples/fake/call_bn_scam.wav")),
    ("EN Safe (REAL)", os.path.join(PROJECT_ROOT, "demo_samples/real/call_en_safe.wav")),
    ("EN Scam (FAKE)", os.path.join(PROJECT_ROOT, "demo_samples/fake/call_en_scam.wav")),
]

print(f"{'Label':22} | {'W2V Fake':>9} | {'Jev Scam':>9} | {'Risk':>6} | {'Locked':>6} | {'Keywords':>15} | Transcript")
print("-" * 105)

for label, p in test_files:
    if not os.path.exists(p):
        print(f"{label:22} | (Skipped - file not found: {os.path.basename(p)})")
        continue

    try:
        with open(p, "rb") as f:
            resp = requests.post(
                url,
                files={"file": f},
                data={"phone_number": "+91-9988776655"},
                timeout=REQUEST_TIMEOUT
            )
        if resp.status_code != 200:
            print(f"{label:22} | HTTP {resp.status_code}: {resp.text[:50]}")
            continue
        res = resp.json()
    except requests.exceptions.RequestException as e:
        print(f"{label:22} | Request error/timeout: {e}")
        continue

    w2v = res.get("w2v_fake_prob", 0)
    jev = res.get("jev_scam_prob", 0)
    risk = res.get("unified_risk_score", 0)
    breaker = res.get("circuit_breaker", {}).get("engaged", False)
    trans = res.get("transcript", "")[:35].replace("\n", " ")
    kw = res.get("flagged_keywords", [])
    print(f"{label:22} | {w2v:8.1f}% | {jev:8.1f}% | {risk:5.1f}% | {str(breaker):>6} | {str(kw):>15} | {trans}")
