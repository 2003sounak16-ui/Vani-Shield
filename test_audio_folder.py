import os, sys, glob, requests
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

audio_dir = r"C:\AI\VoiceCloning\audio"
files = sorted(glob.glob(os.path.join(audio_dir, "*")))
url = "http://127.0.0.1:8000/api/evaluate/upload"

print(f"Scanning directory: {audio_dir}")
print(f"Total audio files found: {len(files)}\n")

for i, fpath in enumerate(files, 1):
    fname = os.path.basename(fpath)
    with open(fpath, "rb") as f:
        r = requests.post(url, files={"file": (fname, f, "audio/octet-stream")}).json()
    
    dur = r.get("duration_seconds", 0)
    tx = r.get("transcript", "")
    w2v = r.get("w2v_fake_prob", 0.0)
    jev = r.get("jev_scam_prob", 0.0)
    risk = r.get("unified_risk_score", 0.0)
    threat = r.get("threat_type", "UNKNOWN")
    cb = r.get("circuit_breaker", {})
    
    # Verdict determination
    is_fake = w2v >= 60.0 or risk >= 60.0
    label = "AI VOICE CLONE (FAKE ❌)" if is_fake else "AUTHENTIC HUMAN (REAL ✅)"
    
    print(f"[{i}] {fname}")
    print(f"    Format: {os.path.splitext(fname)[1]} | Duration: {dur:.2f} seconds")
    print(f"    Verdict: {label}")
    print(f"    AI Fake Probability: {w2v:.1f}%")
    print(f"    Overall Risk Score: {risk:.1f}%")
    print(f"    Circuit Breaker: {'ENGAGED (LOCKED)' if cb.get('engaged') else 'DISENGAGED (UNLOCKED)'}")
    print(f"    Transcript: {tx[:100]}..." if len(tx) > 100 else f"    Transcript: {tx}")
    print("-" * 70)
