import os
import sys
import glob
import requests

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
audio_dir = (
    sys.argv[1]
    if len(sys.argv) > 1
    else os.environ.get("AUDIO_DIR", os.path.join(PROJECT_ROOT, "audio"))
)
files = sorted(glob.glob(os.path.join(audio_dir, "*")))
url = os.environ.get("VANISHIELD_EVAL_URL", "http://127.0.0.1:8000/api/evaluate/upload")

print(f"Scanning directory: {audio_dir}")
print(f"Total audio files found: {len(files)}\n")

REQUEST_TIMEOUT = (5.0, 30.0)

for i, fpath in enumerate(files, 1):
    fname = os.path.basename(fpath)
    try:
        with open(fpath, "rb") as f:
            resp = requests.post(
                url,
                files={"file": (fname, f, "audio/octet-stream")},
                timeout=REQUEST_TIMEOUT
            )
        if resp.status_code != 200:
            print(f"[{i}] {fname}: Request failed with HTTP {resp.status_code}")
            print("-" * 70)
            continue
        r = resp.json()
    except requests.exceptions.RequestException as e:
        print(f"[{i}] {fname}: Evaluation failed or timed out: {e}")
        print("-" * 70)
        continue

    dur = r.get("duration_seconds", 0)
    tx = r.get("transcript", "")
    w2v = r.get("w2v_fake_prob", 0.0)
    jev = r.get("jev_scam_prob", 0.0)
    risk = r.get("unified_risk_score", 0.0)
    threat = r.get("threat_type", "UNKNOWN")
    cb = r.get("circuit_breaker", {})

    # Verdict determination
    is_fake = w2v >= 60.0 or risk >= 60.0
    label = "AI VOICE CLONE (FAKE [X])" if is_fake else "AUTHENTIC HUMAN (REAL [OK])"

    print(f"[{i}] {fname}")
    print(f"    Format: {os.path.splitext(fname)[1]} | Duration: {dur:.2f} seconds")
    print(f"    Verdict: {label}")
    print(f"    AI Fake Probability: {w2v:.1f}%")
    print(f"    Overall Risk Score: {risk:.1f}%")
    print(f"    Circuit Breaker: {'ENGAGED (LOCKED)' if cb.get('engaged') else 'DISENGAGED (UNLOCKED)'}")
    print(f"    Transcript: {tx[:100]}..." if len(tx) > 100 else f"    Transcript: {tx}")
    print("-" * 70)
