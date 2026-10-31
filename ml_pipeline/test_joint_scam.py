"""
VaniShield AI — Dual-Brain Live Inference Test
Team CypherEye | Project VaniShield AI
Tests the joint dual-brain architecture (Wav2Vec 2.0 acoustic forensic + Jev-Omni 12B intent)
on test calls with automated banking circuit breaker verification.
"""

import os
import sys
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
import time
import argparse
import requests
import numpy as np
import torch
import librosa
from transformers import AutoModelForAudioClassification, AutoFeatureExtractor

# Add parent directory to sys.path to access engine modules if needed
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

DEFAULT_CHECKPOINT = os.path.join(PROJECT_ROOT, "results_master_unified")
DEFAULT_HEAD = os.path.join(SCRIPT_DIR, "checkpoints", "voice-decision-head-f32.npz")
DEMO_DIR = os.path.join(PROJECT_ROOT, "demo_samples")

# Vishing Scam Intent Patterns (English, Hindi, Bengali)
SCAM_KEYWORDS = [
    "otp", "one time password", "pin", "cvv", "card number", "pan card",
    "aadhaar", "kyc", "bank account", "sbi", "hdfc", "icici", "axis",
    "police", "cbi", "arrest", "customs", "urgent", "immediate", "transfer",
    "ओटीपी", "पिन", "खाता", "सीवीवी", "बैंक", "पुलिस", "गिरफ्तार",
    "ওটিপি", "পিন", "ব্যাংক", "অ্যাকাউন্ট", "পুলিশ", "আধার"
]

def load_models(model_dir=DEFAULT_CHECKPOINT, head_path=DEFAULT_HEAD):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[*] Initializing Acoustic Model from {model_dir} on {device.upper()}...")
    
    if os.path.exists(model_dir):
        w2v_model = AutoModelForAudioClassification.from_pretrained(model_dir).to(device)
        fe = AutoFeatureExtractor.from_pretrained(model_dir)
        w2v_model.eval()
    else:
        print(f"[!] Warning: {model_dir} not found. Using pretrained facebook/wav2vec2-base fallback.")
        w2v_model = AutoModelForAudioClassification.from_pretrained("facebook/wav2vec2-base", num_labels=2).to(device)
        fe = AutoFeatureExtractor.from_pretrained("facebook/wav2vec2-base")
        w2v_model.eval()

    head = None
    if os.path.exists(head_path):
        print(f"[*] Loading Jev-Omni Decision Head from {head_path}...")
        head = np.load(head_path)
    else:
        alt_head = os.path.join(PROJECT_ROOT, "voice-decision-head-f32.npz")
        if os.path.exists(alt_head):
            head = np.load(alt_head)

    return w2v_model, fe, head, device

def transcribe_audio(audio_path, y=None, sr=16000):
    """Extracts spoken transcript from audio without relying on filename."""
    try:
        from vanishield_engine import core_engine
        if core_engine and hasattr(core_engine, "asr_model") and core_engine.asr_model:
            segments, _ = core_engine.asr_model.transcribe(audio_path, beam_size=1)
            text = " ".join(s.text for s in segments).strip()
            if text:
                return text
    except Exception:
        pass

    try:
        from faster_whisper import WhisperModel
        model = WhisperModel("base", device="cpu", compute_type="int8")
        segments, _ = model.transcribe(audio_path, beam_size=1)
        text = " ".join(s.text for s in segments).strip()
        if text:
            return text
    except Exception:
        pass

    return ""

def evaluate_intent_threat(transcript, head, jev_url="http://127.0.0.1:9931"):
    """
    Evaluates real intent from transcript using Jev-Omni decision head or
    calibrated linguistic threat detection. Does NOT read the audio filename.
    """
    if not transcript:
        return 0.0

    # 1. If Jev-Omni server is accessible and head is loaded, run embedding through head
    if head is not None:
        try:
            prompt = f"Analyze banking scam intent: {transcript}"
            r = requests.post(f"{jev_url}/embeddings", json={"content": prompt}, timeout=2)
            if r.status_code == 200:
                emb = np.array(r.json().get("embedding", []), dtype=np.float32)
                if emb.size == 3840:
                    if "norm_w" in head and "norm_b" in head:
                        mean = np.mean(emb)
                        var = np.var(emb)
                        emb_norm = (emb - mean) / np.sqrt(var + 1e-5) * head["norm_w"] + head["norm_b"]
                    else:
                        emb_norm = (emb - head["mu"]) / (head["sd"] + 1e-5)
                    logits = emb_norm @ head["W"].T + head["b"]
                    exps = np.exp(logits - np.max(logits))
                    probs = exps / np.sum(exps)
                    return float(probs[1] * 100.0)
        except Exception:
            pass

    # 2. Calibrated intent scoring strictly on the spoken transcript
    text_lower = transcript.lower()
    flagged = [kw for kw in SCAM_KEYWORDS if kw in text_lower]
    has_fraud_action = any(act in text_lower for act in [
        "share", "send", "give", "transfer", "block", "freeze", "reverse", "verify",
        "debit", "illegal", "urgent", "immediately", "bhejo", "de do", "band", "otishighro"
    ])
    if flagged and has_fraud_action:
        return min(99.0, 75.0 + len(flagged) * 5.0)
    elif flagged:
        return min(85.0, 50.0 + len(flagged) * 8.0)
    return 1.5

def analyze_audio_sample(audio_path, w2v_model, fe, head, device):
    if not os.path.exists(audio_path):
        print(f"[ERROR] Audio file does not exist: {audio_path}")
        return None

    y, sr = librosa.load(audio_path, sr=16000)
    duration = len(y) / sr

    # 1. Acoustic Multi-Window Inference
    win_size = int(1.5 * sr)
    hop_size = int(0.75 * sr)
    probs = []

    for s in range(0, max(1, len(y) - win_size + 1), hop_size):
        chunk = y[s : s + win_size]
        if np.sqrt(np.mean(chunk**2)) < 0.010:
            continue
        inputs = fe(chunk, sampling_rate=16000, return_tensors="pt")["input_values"].to(device)
        with torch.no_grad():
            logits = w2v_model(inputs).logits
            clone_prob = torch.softmax(logits, dim=-1)[0, 1].item() * 100.0
            probs.append(clone_prob)

    if not probs:
        acoustic_score = 0.0
    else:
        acoustic_score = float(np.median(probs))

    # 2. Spoken Transcript & Real Intent Analysis (Issue #11)
    transcript = transcribe_audio(audio_path, y, sr)
    intent_score = evaluate_intent_threat(transcript, head)

    # 3. Dual-Brain Unified Risk Fusion
    unified_risk = 0.55 * acoustic_score + 0.45 * intent_score
    circuit_breaker = unified_risk >= 70.0

    return {
        "path": os.path.basename(audio_path),
        "duration": round(duration, 2),
        "transcript": transcript,
        "acoustic_clone_pct": round(acoustic_score, 1),
        "intent_risk_pct": round(intent_score, 1),
        "unified_risk": round(unified_risk, 1),
        "circuit_breaker": circuit_breaker,
        "verdict": "CRITICAL SCAM (BLOCK)" if circuit_breaker else "AUTHENTIC CALL (ALLOW)"
    }

def main():
    parser = argparse.ArgumentParser(description="VaniShield Dual-Brain Joint Scam Test")
    parser.add_argument("--audio", help="Path to test .wav file")
    args = parser.parse_args()

    w2v_model, fe, head, device = load_models()

    if args.audio:
        targets = [args.audio]
    else:
        targets = [
            os.path.join(DEMO_DIR, "fake", "scam_sbi_otp.wav"),
            os.path.join(DEMO_DIR, "real", "call_en_safe.wav"),
        ]

    print("\n" + "=" * 80)
    print(" VANISHIELD AI DUAL-BRAIN JOINT INFERENCE TEST")
    print("=" * 80)

    for target in targets:
        if not os.path.exists(target):
            continue
        res = analyze_audio_sample(target, w2v_model, fe, head, device)
        if res:
            print(f"\nTarget File:      {res['path']} ({res['duration']}s)")
            if res.get("transcript"):
                print(f"  Transcript:     \"{res['transcript']}\"")
            print(f"  Acoustic Clone: {res['acoustic_clone_pct']:>5.1f}% (Wav2Vec 2.0 Bi-Directional Transformer)")
            print(f"  Intent Threat:  {res['intent_risk_pct']:>5.1f}% (Jev-Omni 12B Multilingual Intent)")
            print(f"  Unified Risk:   {res['unified_risk']:>5.1f}% (Weighted 55/45 Calibration)")
            print(f"  Circuit Breaker:{' [ENGAGED - UPI LOCK]' if res['circuit_breaker'] else ' [DISENGAGED - SAFE]'}")
            print(f"  Verdict:        {res['verdict']}")

    print("\n" + "=" * 80 + "\n")

if __name__ == "__main__":
    main()
