import os
import sys
import glob
import requests

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# Allow configurable audio directory via CLI arg or environment variable
audio_dir = (
    sys.argv[1]
    if len(sys.argv) > 1
    else os.environ.get("AUDIO_DIR", os.path.join(os.path.dirname(os.path.abspath(__file__)), "audio"))
)
files = sorted(glob.glob(os.path.join(audio_dir, "*")))
url = os.environ.get("VANISHIELD_EVAL_URL", "http://127.0.0.1:8000/api/evaluate/upload")

print(f"Total audio files in {audio_dir}: {len(files)}\n")

# Bounded connection & read timeouts to prevent infinite stalling on failed samples
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
            print(f"[{i}] {fname}: Request failed with HTTP {resp.status_code} - {resp.text[:120]}")
            print("-" * 60)
            continue
        r = resp.json()
    except requests.exceptions.RequestException as e:
        print(f"[{i}] {fname}: Connection/evaluation error (skipped): {e}")
        print("-" * 60)
        continue

    dur = r.get("duration_seconds", 0)
    tx = r.get("transcript", "")
    threat = r.get("threat_type", "UNKNOWN")
    risk = r.get("unified_risk_score", 0.0)
    w2v = r.get("w2v_fake_prob", 0.0)
    jev = r.get("jev_scam_prob", 0.0)
    cb = r.get("circuit_breaker", {})

    print(f"[{i}] {fname}")
    print(f"    Duration: {dur:.2f} s")
    print(f"    Transcript: {tx}")
    print(f"    Threat: {threat}")
    print(f"    Unified Risk: {risk}% (AI Clone: {w2v}%, Scam: {jev}%)")
    print(f"    Circuit Breaker: Engaged={cb.get('engaged')}, Action={cb.get('action')}")
    print("-" * 60)
