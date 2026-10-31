import os, sys, glob, requests
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

audio_dir = r"C:\Users\souna\Downloads\audio"
files = glob.glob(os.path.join(audio_dir, "*"))
url = "http://127.0.0.1:8000/api/evaluate/upload"

print(f"Total audio files in {audio_dir}: {len(files)}\n")

for i, fpath in enumerate(files, 1):
    fname = os.path.basename(fpath)
    with open(fpath, "rb") as f:
        r = requests.post(url, files={"file": (fname, f, "audio/octet-stream")}).json()
    
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
