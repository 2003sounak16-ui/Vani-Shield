import os
import sys
import glob
import json
import librosa
import numpy as np
import torch
from transformers import AutoFeatureExtractor, AutoModelForAudioClassification

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

device = "cuda" if torch.cuda.is_available() else "cpu"

model_dirs = {
    "Master Unified (Production)": r"c:\AI\VoiceCloning\results_master_unified",
    "Master SOTA": r"c:\AI\VoiceCloning\results_master_sota",
    "Master Balanced": r"c:\AI\VoiceCloning\results_master_balanced",
    "4447 Models": r"c:\AI\VoiceCloning\results_4447_models",
    "Real Multilingual": r"c:\AI\VoiceCloning\results_real_multilingual"
}

loaded_models = {}
fe = None
for name, p in model_dirs.items():
    if os.path.exists(os.path.join(p, "config.json")):
        try:
            m = AutoModelForAudioClassification.from_pretrained(p).to(device)
            m.eval()
            loaded_models[name] = m
            if fe is None:
                fe = AutoFeatureExtractor.from_pretrained(p)
            print(f"Loaded {name}")
        except Exception as e:
            print(f"Error loading {name}: {e}")

audio_dir = r"c:\AI\VoiceCloning\audio"
files = sorted(glob.glob(os.path.join(audio_dir, "*")))

def analyze_dsp(y, sr=16000):
    # F0 Pitch using librosa piptrack or pyin
    f0, voiced_flag, voiced_probs = librosa.pyin(y, fmin=librosa.note_to_hz('C2'), fmax=librosa.note_to_hz('C7'), sr=sr)
    f0_clean = f0[~np.isnan(f0)] if f0 is not None else []
    f0_mean = float(np.mean(f0_clean)) if len(f0_clean) > 0 else 0.0
    f0_std = float(np.std(f0_clean)) if len(f0_clean) > 0 else 0.0
    voiced_ratio = float(len(f0_clean) / len(f0)) if f0 is not None and len(f0) > 0 else 0.0

    zcr = float(np.mean(librosa.feature.zero_crossing_rate(y)))
    flatness = float(np.mean(librosa.feature.spectral_flatness(y=y)))
    centroid = float(np.mean(librosa.feature.spectral_centroid(y=y, sr=sr)))
    rolloff = float(np.mean(librosa.feature.spectral_rolloff(y=y, sr=sr, roll_percent=0.85)))
    
    return {
        "f0_mean": round(f0_mean, 1),
        "f0_std": round(f0_std, 1),
        "voiced_ratio": round(voiced_ratio, 3),
        "zcr": round(zcr, 4),
        "flatness": round(flatness, 6),
        "centroid": round(centroid, 1),
        "rolloff": round(rolloff, 1)
    }

results = []

for idx, f in enumerate(files, 1):
    fname = os.path.basename(f)
    y, sr = librosa.load(f, sr=16000, mono=True)
    dur = len(y) / 16000.0

    dsp = analyze_dsp(y, sr=16000)

    # Multi-window evaluation
    win_len = 48000 # 3 sec
    step = 16000 # 1 sec step
    starts = list(range(0, max(1, len(y) - win_len + 1), step))
    if not starts:
        starts = [0]

    model_scores = {}
    for mname, model in loaded_models.items():
        w_scores = []
        with torch.no_grad():
            for s in starts:
                chunk = y[s : s + win_len]
                if len(chunk) < win_len:
                    chunk = np.pad(chunk, (0, win_len - len(chunk)))
                inputs = fe(chunk, sampling_rate=16000, return_tensors="pt")["input_values"].to(device)
                logits = model(inputs).logits
                fake_prob = float(torch.softmax(logits, dim=-1)[0, 1].cpu()) * 100.0
                w_scores.append(fake_prob)
        model_scores[mname] = {
            "mean": round(float(np.mean(w_scores)), 2),
            "median": round(float(np.median(w_scores)), 2),
            "p75": round(float(np.percentile(w_scores, 75)), 2),
            "min": round(float(np.min(w_scores)), 2),
            "max": round(float(np.max(w_scores)), 2),
            "windows_over_50": sum(1 for w in w_scores if w > 50.0),
            "total_windows": len(w_scores)
        }

    report = {
        "file": fname,
        "duration": round(dur, 2),
        "dsp": dsp,
        "models": model_scores
    }
    results.append(report)
    print(f"\n=======================================================")
    print(f"[{idx}] {fname} ({dur:.2f}s)")
    print(f"DSP: F0 Mean={dsp['f0_mean']} Hz (Std={dsp['f0_std']} Hz), Voiced Ratio={dsp['voiced_ratio']*100:.1f}%, ZCR={dsp['zcr']}, Flatness={dsp['flatness']}, Centroid={dsp['centroid']} Hz, Rolloff={dsp['rolloff']} Hz")
    print("Model Scores (Fake Probability %):")
    for mname, sc in model_scores.items():
        print(f"  {mname:30} -> Mean: {sc['mean']:5.1f}% | Med: {sc['median']:5.1f}% | Max: {sc['max']:5.1f}% | Windows > 50%: {sc['windows_over_50']}/{sc['total_windows']}")

with open(r"c:\AI\VoiceCloning\deep_forensics_results.json", "w", encoding="utf-8") as out:
    json.dump(results, out, indent=2)
