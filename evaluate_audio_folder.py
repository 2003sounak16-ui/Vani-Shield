import os
import sys
import glob
import time
import torch
import librosa
import numpy as np

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from vanishield_engine import core_engine
from transformers import AutoFeatureExtractor, AutoModelForAudioClassification

def evaluate():
    audio_dir = r"c:\AI\VoiceCloning\audio"
    files = sorted(glob.glob(os.path.join(audio_dir, "*")))
    print(f"=== EVALUATING {len(files)} AUDIO FILES ===\n")

    # Also load models directly for multi-model verification
    models = {}
    checkpoints = {
        "master_unified": r"c:\AI\VoiceCloning\results_master_unified",
        "master_sota": r"c:\AI\VoiceCloning\results_master_sota",
        "master_balanced": r"c:\AI\VoiceCloning\results_master_balanced"
    }
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    fe = None
    for name, path in checkpoints.items():
        if os.path.exists(os.path.join(path, "config.json")):
            try:
                m = AutoModelForAudioClassification.from_pretrained(path).to(device)
                m.eval()
                models[name] = m
                if fe is None:
                    fe = AutoFeatureExtractor.from_pretrained(path)
            except Exception as e:
                print(f"Note: Could not load {name}: {e}")

    for i, fpath in enumerate(files, 1):
        fname = os.path.basename(fpath)
        audio_np, sr = librosa.load(fpath, sr=16000, mono=True)
        dur = len(audio_np) / 16000.0

        # 1. Full VaniShield Core Evaluation
        res = core_engine.evaluate_live_call(
            audio_chunk_16k=audio_np,
            phone_number="+91-0000000000 (Local WhatsApp Audio)",
            is_upload=True
        )

        # 2. Raw Acoustic Model Scoring (multi-window averaging)
        raw_scores = {}
        window_size = 48000 # 3.0s
        if len(audio_np) <= window_size:
            starts = [0]
        else:
            num_w = min(10, max(3, len(audio_np) // 16000))
            starts = np.linspace(0, len(audio_np) - window_size, num=num_w, dtype=int).tolist()

        for m_name, model in models.items():
            probs = []
            with torch.no_grad():
                for s in starts:
                    clip = audio_np[s : s + window_size]
                    if len(clip) < window_size:
                        clip = np.pad(clip, (0, window_size - len(clip)))
                    inputs = fe(clip, sampling_rate=16000, return_tensors="pt")["input_values"].to(device)
                    logits = model(inputs).logits
                    p_fake = float(torch.softmax(logits, dim=-1)[0, 1].cpu()) * 100.0
                    probs.append(p_fake)
            raw_scores[m_name] = {
                "mean_fake_prob": round(float(np.mean(probs)), 2),
                "median_fake_prob": round(float(np.median(probs)), 2),
                "min_fake_prob": round(float(np.min(probs)), 2),
                "max_fake_prob": round(float(np.max(probs)), 2)
            }

        print(f"[{i}] {fname}")
        print(f"    Duration: {dur:.2f} seconds")
        print(f"    Verdict : {res['threat_type']}")
        print(f"    Unified Risk Score : {res['unified_risk_score']}%")
        print(f"    VaniShield W2V Fake: {res['w2v_fake_prob']}%")
        print(f"    Circuit Breaker    : {res['circuit_breaker']['action']}")
        print(f"    Transcript         : \"{res.get('transcript', '').strip()}\"")
        if res.get('flagged_keywords'):
            print(f"    Flagged Keywords   : {res.get('flagged_keywords')}")
        
        dsp = res.get('dsp_metrics', {})
        print(f"    DSP Features       : F0={dsp.get('pitch_f0_hz')} Hz | ZCR={dsp.get('zero_crossing_rate')} | Flatness={dsp.get('spectral_flatness')} | Centroid={dsp.get('spectral_centroid_hz')} Hz")
        print(f"    Raw Model Predictions:")
        for m_name, score in raw_scores.items():
            print(f"      - {m_name:16}: Fake Prob Mean={score['mean_fake_prob']}% (Med={score['median_fake_prob']}%, Min={score['min_fake_prob']}%, Max={score['max_fake_prob']}%)")
        print("-" * 80)

if __name__ == "__main__":
    evaluate()
