import os
import sys
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
import time
import glob
import numpy as np
import torch
import torchaudio
from transformers import AutoFeatureExtractor, AutoModelForAudioClassification
from predict import load_audio

BENCHMARK_SAMPLES = [
    ("Real: User Voice (Sounak)", r"C:\AI\VoiceCloning\user_test_voice_16k.wav", 0),
    ("Real: Friend Unseen", r"C:\AI\VoiceCloning\data_demo\unseen_test\unseen_real_friend.wav", 0),
    ("Real: Bengali CommonVoice", glob.glob("data_real/real/bn/*.wav")[0], 0),
    ("Real: Hindi CommonVoice", glob.glob("data_real/real/hi/*.wav")[0], 0),
    ("Real: Indian English", glob.glob("data_real/real/en/*.wav")[0], 0),
    ("Real: WhatsApp Voice Note", r"C:\Users\souna\Downloads\WhatsApp Audio 2026-10-30 at 7.10.45 PM.ogg", 0),
    ("Fake: ElevenLabs Rahul Vidya", r"C:\Users\souna\Downloads\WhatsApp Audio 2026-10-30 at 7.29.08 PM.ogg", 1),
    ("Fake: CVV Scam Call", r"C:\AI\VoiceCloning\data_demo\fake\scam_card_cvv.wav", 1),
    ("Fake: SBI OTP Call", r"C:\AI\VoiceCloning\data_demo\fake\scam_sbi_otp.wav", 1),
    ("Fake: HDFC Scam Call", r"C:\AI\VoiceCloning\data_demo\unseen_test\unseen_scam_hdfc.wav", 1),
    ("Fake: Bengali TTS Clone", glob.glob("data_real/fake/bn/*.wav")[0], 1),
    ("Fake: Hindi TTS Clone", glob.glob("data_real/fake/hi/*.wav")[0], 1)
]

def load_resampled_audio(path):
    wav, sr = load_audio(path)
    if wav.shape[0] > 1:
        wav = wav.mean(dim=0, keepdim=True)
    if sr != 16000:
        wav = torchaudio.transforms.Resample(sr, 16000)(wav)
    return wav.squeeze(0).numpy()

def evaluate_hf_model(model_dir, device="cuda"):
    print(f"\nEvaluating: {model_dir} on {device.upper()}...")
    fe = AutoFeatureExtractor.from_pretrained(model_dir)
    model = AutoModelForAudioClassification.from_pretrained(model_dir).to(device)
    model.eval()

    results = []
    latencies = []
    window_size = 48000

    for label, path, expected in BENCHMARK_SAMPLES:
        audio = load_resampled_audio(path)
        if len(audio) <= window_size:
            starts = [0]
        else:
            starts = np.linspace(0, len(audio) - window_size, num=5, dtype=int).tolist()

        t0 = time.perf_counter()
        window_probs = []
        with torch.no_grad():
            for s in starts:
                clip = audio[s : s + window_size]
                if len(clip) < window_size:
                    clip = np.pad(clip, (0, window_size - len(clip)))
                inputs = fe(clip, sampling_rate=16000, return_tensors="pt").to(device)
                logits = model(**inputs).logits
                p = torch.softmax(logits, dim=-1).squeeze(0).cpu().numpy()
                window_probs.append(p)
        latency = (time.perf_counter() - t0) * 1000
        latencies.append(latency)

        avg_prob = np.mean(window_probs, axis=0)
        pred = int(np.argmax(avg_prob))
        fake_prob = float(avg_prob[1]) * 100.0
        results.append({
            "sample": label,
            "expected": expected,
            "pred": pred,
            "fake_prob": fake_prob,
            "correct": (pred == expected)
        })

    return results, np.mean(latencies)

def evaluate_vanishield_core():
    print(f"\nEvaluating: VaniShield Unified Core Engine (Multimodal)...")
    from vanishield_engine import core_engine
    results = []
    latencies = []

    for label, path, expected in BENCHMARK_SAMPLES:
        audio = load_resampled_audio(path)
        t0 = time.perf_counter()
        res = core_engine.evaluate_live_call(audio, phone_number="+91-0000000000 (Bench)")
        latency = (time.perf_counter() - t0) * 1000
        latencies.append(latency)

        # Unified risk score threshold >= 70.0% or fake prob >= 50.0%
        pred = 1 if (res["circuit_breaker"]["engaged"] or res["w2v_fake_prob"] >= 50.0) else 0
        fake_prob = res["w2v_fake_prob"]
        results.append({
            "sample": label,
            "expected": expected,
            "pred": pred,
            "fake_prob": fake_prob,
            "correct": (pred == expected),
            "circuit_breaker": res["circuit_breaker"]["engaged"],
            "risk_score": res["unified_risk_score"]
        })

    return results, np.mean(latencies)

def print_summary(model_name, results, avg_latency):
    y_true = [r["expected"] for r in results]
    y_pred = [r["pred"] for r in results]
    
    tp = sum(1 for yt, yp in zip(y_true, y_pred) if yt == 1 and yp == 1)
    tn = sum(1 for yt, yp in zip(y_true, y_pred) if yt == 0 and yp == 0)
    fp = sum(1 for yt, yp in zip(y_true, y_pred) if yt == 0 and yp == 1)
    fn = sum(1 for yt, yp in zip(y_true, y_pred) if yt == 1 and yp == 0)

    acc = (tp + tn) / len(y_true) * 100.0
    prec = (tp / (tp + fp) * 100.0) if (tp + fp) > 0 else 0.0
    rec = (tp / (tp + fn) * 100.0) if (tp + fn) > 0 else 0.0
    f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0

    print("=" * 75)
    print(f"📊 MODEL: {model_name}")
    print("=" * 75)
    print(f"Accuracy  : {acc:.1f}% ({tp+tn}/{len(y_true)})")
    print(f"Precision : {prec:.1f}% | Recall: {rec:.1f}% | F1-Score: {f1:.1f}%")
    print(f"Confusion : TP={tp}, TN={tn}, FP={fp} (False Alarm), FN={fn} (Missed Fake)")
    print(f"Avg Time  : {avg_latency:.1f} ms / call")
    print("-" * 75)
    for r in results:
        status_icon = "✅" if r["correct"] else "❌"
        exp_txt = "FAKE" if r["expected"] == 1 else "REAL"
        pred_txt = "FAKE" if r["pred"] == 1 else "REAL"
        print(f"  {status_icon} {r['sample']:30s} | Exp: {exp_txt:4s} | Pred: {pred_txt:4s} ({r['fake_prob']:5.1f}% Fake)")
    print("=" * 75 + "\n")
    return {"name": model_name, "acc": acc, "prec": prec, "rec": rec, "f1": f1, "latency": avg_latency, "tp": tp, "tn": tn, "fp": fp, "fn": fn}

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    models_to_test = [
        "results_real_multilingual",
        "results_4447_models",
        "results_kaggle_all",
        "results_pilot"
    ]

    summaries = []

    for m in models_to_test:
        if os.path.exists(m):
            try:
                res, lat = evaluate_hf_model(m, device=device)
                summ = print_summary(m, res, lat)
                summaries.append(summ)
            except Exception as e:
                print(f"Error evaluating {m}: {e}")

    # Evaluate VaniShield Core
    try:
        core_res, core_lat = evaluate_vanishield_core()
        core_summ = print_summary("VaniShield Unified Core Engine (Multimodal)", core_res, core_lat)
        summaries.append(core_summ)
    except Exception as e:
        print(f"Error evaluating VaniShield Core: {e}")

    # Final Comparative Leaderboard
    print("\n" + "#" * 78)
    print("🏆 FINAL COMPARATIVE MODEL LEADERBOARD")
    print("#" * 78)
    header = f"{'Model Name':40s} | {'Accuracy':8s} | {'Precision':9s} | {'Recall':8s} | {'F1':6s} | {'Latency':8s}"
    print(header)
    print("-" * 78)
    for s in summaries:
        line = f"{s['name'][:40]:40s} | {s['acc']:7.1f}% | {s['prec']:8.1f}% | {s['rec']:7.1f}% | {s['f1']:5.1f}% | {s['latency']:6.1f}ms"
        print(line)
    print("#" * 78 + "\n")

if __name__ == "__main__":
    main()
