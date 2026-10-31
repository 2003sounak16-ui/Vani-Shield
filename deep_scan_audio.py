import os, sys, glob, torch, torchaudio, av, numpy as np, librosa
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from train_master_unified import load_audio_robust
from vanishield_engine import core_engine

audio_dir = r"C:\AI\VoiceCloning\audio"
files = sorted(glob.glob(os.path.join(audio_dir, "*")))

print(f"=== DEEP FORENSIC SCAN OF {len(files)} AUDIO FILES ===\n")

for i, fpath in enumerate(files, 1):
    fname = os.path.basename(fpath)
    waveform = load_audio_robust(fpath, target_sr=16000)
    audio_np = waveform.squeeze(0).numpy()
    dur = len(audio_np) / 16000.0
    
    # Multi-window evaluation from engine
    res = core_engine.evaluate_live_call(audio_np, phone_number="WhatsApp Test", is_upload=True)
    
    # Dense sliding window analysis (3-second window, 1.5-second stride)
    window_size = 48000
    w_scores = []
    if len(audio_np) <= window_size:
        starts = [0]
    else:
        starts = list(range(0, len(audio_np) - window_size + 1, 24000))
        if starts[-1] != len(audio_np) - window_size:
            starts.append(len(audio_np) - window_size)
            
    for s in starts:
        clip = audio_np[s : s + window_size]
        if len(clip) < window_size:
            clip = np.pad(clip, (0, window_size - len(clip)))
        clip_t = core_engine.fe(clip, sampling_rate=16000, return_tensors="pt")["input_values"].to(core_engine.device)
        with torch.no_grad():
            p = float(torch.softmax(core_engine.model.wav2vec_cls(clip_t).logits, dim=-1)[0, 1].cpu()) * 100.0
        w_scores.append(round(p, 2))
    
    dsp = core_engine.extract_prosodic_dsp(audio_np)
    tx = res.get("transcript", "")
    
    print(f"[{i}] {fname}")
    print(f"    Duration: {dur:.2f} s | Samples: {len(audio_np)}")
    print(f"    Transcript: {tx}")
    print(f"    DSP: Pitch F0={dsp.get('pitch_f0_hz')} Hz, Centroid={dsp.get('spectral_centroid_hz')} Hz, ZCR={dsp.get('zero_crossing_rate')}, RMS={dsp.get('rms_energy')}")
    print(f"    Window-by-Window Fake Probabilities: {w_scores}")
    print(f"    Max Window Prob: {max(w_scores) if w_scores else 0:.1f}% | Median: {np.median(w_scores) if w_scores else 0:.1f}%")
    print(f"    Final W2V Prob: {res.get('w2v_fake_prob')}% | Jev Scam Prob: {res.get('jev_scam_prob')}%")
    print(f"    Final Threat: {res.get('threat_type')} (Risk: {res.get('unified_risk_score')}%)")
    print(f"    Circuit Breaker: {res.get('circuit_breaker', {}).get('engaged')} ({res.get('circuit_breaker', {}).get('action')})")
    print("-" * 70)
