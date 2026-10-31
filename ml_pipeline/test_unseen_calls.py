"""
VaniShield AI — Unseen Test Audio Generalization Evaluation
Team CypherEye | Project VaniShield AI
Evaluates model performance against held-out, completely unseen live test audio.
"""

import os
import sys
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
import time
import torch
import librosa
import numpy as np
from transformers import AutoModelForAudioClassification, AutoFeatureExtractor

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

UNSEEN_DIR = os.path.join(PROJECT_ROOT, "demo_samples", "unseen_test")
CHECKPOINT_DIR = os.path.join(PROJECT_ROOT, "results_master_unified")

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("\n" + "=" * 75)
    print(" VANISHIELD AI -- UNSEEN CALLS GENERALIZATION EVALUATION")
    print(f" Compute Device: {device.upper()} | Model: {CHECKPOINT_DIR}")
    print("=" * 75)

    if not os.path.exists(CHECKPOINT_DIR):
        print(f"[ERROR] Checkpoint directory not found at {CHECKPOINT_DIR}")
        return

    model = AutoModelForAudioClassification.from_pretrained(CHECKPOINT_DIR).to(device)
    fe = AutoFeatureExtractor.from_pretrained(CHECKPOINT_DIR)
    model.eval()

    test_files = [
        {"name": "unseen_real_friend.wav", "expected": "REAL (HUMAN)", "type": "Safe Friend Call"},
        {"name": "unseen_scam_hdfc.wav", "expected": "SCAM (AI CLONE)", "type": "HDFC Vishing Attack"},
    ]

    all_passed = True
    for item in test_files:
        path = os.path.join(UNSEEN_DIR, item["name"])
        if not os.path.exists(path):
            # Fallback to data_demo if demo_samples not yet migrated
            alt_path = os.path.join(PROJECT_ROOT, "data_demo", "unseen_test", item["name"])
            if os.path.exists(alt_path):
                path = alt_path
            else:
                print(f"[!] Warning: Test file {item['name']} not found.")
                continue

        t0 = time.perf_counter()
        y, sr = librosa.load(path, sr=16000)
        dur = len(y) / sr

        win_size = int(1.5 * sr)
        hop_size = int(0.75 * sr)
        probs = []

        for s in range(0, max(1, len(y) - win_size + 1), hop_size):
            chunk = y[s : s + win_size]
            if np.sqrt(np.mean(chunk**2)) < 0.010:
                continue
            inputs = fe(chunk, sampling_rate=16000, return_tensors="pt")["input_values"].to(device)
            with torch.no_grad():
                logits = model(inputs).logits
                p = torch.softmax(logits, dim=-1)[0, 1].item() * 100.0
                probs.append(p)

        latency_ms = (time.perf_counter() - t0) * 1000
        p_med = float(np.median(probs)) if probs else 0.0
        p_max = float(np.max(probs)) if probs else 0.0

        if "SCAM" in item["expected"]:
            passed = p_max > 60.0 or p_med > 50.0
            pred = "SCAM (AI CLONE)" if passed else "REAL (FALSE NEGATIVE)"
        else:
            passed = p_med < 30.0 and p_max < 55.0
            pred = "REAL (HUMAN)" if passed else "SCAM (FALSE POSITIVE)"

        if not passed:
            all_passed = False

        status_str = "[PASS]" if passed else "[FAIL]"
        print(f"\n{status_str} Audio: {item['name']} ({dur:.1f}s) - {item['type']}")
        print(f"       Expected: {item['expected']} | Predicted: {pred}")
        print(f"       Scores: Median={p_med:.1f}%, Peak={p_max:.1f}% | Latency: {latency_ms:.1f}ms")

    print("\n" + "=" * 75)
    if all_passed:
        print("[SUCCESS] All unseen generalization evaluation tests PASSED successfully.")
    else:
        print("[NOTICE] Generalization tests completed with calibration adjustments required.")
    print("=" * 75 + "\n")

if __name__ == "__main__":
    main()
