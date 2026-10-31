import os
import sys
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import json
import numpy as np
import soundfile as sf
import torch

from vanishield_engine import core_engine

WAV_PATH = r"C:\AI\VoiceCloning\user_test_voice_16k.wav"

def main():
    print("=" * 70)
    print("🔬 VANISHIELD AI — LIVE FORENSIC EVALUATION OF USER UPLOAD")
    print("=" * 70)
    
    if not os.path.exists(WAV_PATH):
        print(f"Error: {WAV_PATH} not found.")
        return

    # Load audio
    audio_np, sr = sf.read(WAV_PATH, dtype="float32")
    if audio_np.ndim > 1:
        audio_np = audio_np.mean(axis=1)

    duration_sec = len(audio_np) / sr
    print(f"📁 Audio File: user_test_voice_16k.wav")
    print(f"⏱️ Duration: {duration_sec:.2f} seconds ({len(audio_np)} samples @ {sr}Hz)")

    # Run full forensic engine
    phone_number = "+91-9830000000 (Sounak Patra, Dakshin Barasat, West Bengal)"
    print(f"📞 Simulated Caller: {phone_number}")
    print("\n⏳ Running end-to-end multimodal forensics across all 8 pillars...\n")

    result = core_engine.evaluate_live_call(
        audio_chunk_16k=audio_np,
        phone_number=phone_number
    )

    print("-" * 70)
    print("📊 COMPREHENSIVE FORENSIC REPORT")
    print("-" * 70)
    print(f"⚡ Processing Latency:        {result['latency_ms']} ms")
    print(f"🗣️ Transcribed Speech (ASR): \"{result['transcript']}\"")
    print(f"🚨 Flagged Scam Keywords:    {result['flagged_keywords']}")
    print(f"🎙️ Wav2Vec 2.0 Acoustic Fake: {result['w2v_fake_prob']}%")
    print(f"🧠 Jev-Omni Contextual Scam:  {result['jev_scam_prob']}%")
    print(f"🛡️ Unified Multi-Engine Risk: {result['unified_risk_score']}%")
    
    cb = result['circuit_breaker']
    status_label = "🚨 CRITICAL THREAT: FAKE / SCAM" if cb['engaged'] else "✅ SAFE: AUTHENTIC REAL HUMAN SPEECH"
    print(f"🎯 Final Classification:     {status_label}")
    print(f"🔒 Banking Circuit Breaker:   {'ENGAGED (LOCKED)' if cb['engaged'] else 'DISENGAGED (PAYMENTS UNLOCKED)'}")
    print(f"📋 Action Taken:              {cb['action']}")

    print("\n📈 Prosodic DSP Biomarkers:")
    for k, v in result['dsp_metrics'].items():
        print(f"   • {k}: {v}")

    print("\n📍 Telecom Intelligence:")
    for k, v in result['telecom'].items():
        print(f"   • {k}: {v}")
    print("=" * 70)

if __name__ == "__main__":
    main()
