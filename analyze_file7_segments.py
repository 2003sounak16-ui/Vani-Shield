import os
import sys
import librosa
import numpy as np
import torch
from transformers import AutoFeatureExtractor, AutoModelForAudioClassification
from faster_whisper import WhisperModel

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

device = "cuda" if torch.cuda.is_available() else "cpu"
model_dir = r"c:\AI\VoiceCloning\results_master_unified"
fe = AutoFeatureExtractor.from_pretrained(model_dir)
model = AutoModelForAudioClassification.from_pretrained(model_dir).to(device)
model.eval()

asr = WhisperModel("base", device="cpu", compute_type="int8")

fpath = r"c:\AI\VoiceCloning\audio\WhatsApp Audio 2026-10-31 at 12.39.30 AM.mpeg"
y, sr = librosa.load(fpath, sr=16000)

segments, info = asr.transcribe(y, beam_size=5)

print(f"=== DETAILED SEGMENT-BY-SEGMENT ANALYSIS OF FILE 7 ===")
for s in segments:
    chunk = y[int(s.start * 16000) : int(s.end * 16000)]
    if len(chunk) < 8000: # pad to at least 0.5s
        chunk = np.pad(chunk, (0, 8000 - len(chunk)))
    
    # DSP
    f0, _, _ = librosa.pyin(chunk, fmin=librosa.note_to_hz('C2'), fmax=librosa.note_to_hz('C7'), sr=16000)
    f0_c = f0[~np.isnan(f0)] if f0 is not None else []
    f0_m = float(np.mean(f0_c)) if len(f0_c) > 0 else 0.0
    zcr = float(np.mean(librosa.feature.zero_crossing_rate(chunk)))
    cent = float(np.mean(librosa.feature.spectral_centroid(y=chunk, sr=16000)))
    
    # Model score
    inp = fe(chunk, sampling_rate=16000, return_tensors="pt")["input_values"].to(device)
    with torch.no_grad():
        logits = model(inp).logits
        fake_p = float(torch.softmax(logits, dim=-1)[0, 1].cpu()) * 100.0
        
    print(f"[{s.start:05.2f} - {s.end:05.2f}] FakeProb: {fake_p:5.1f}% | F0: {f0_m:5.1f}Hz | Centroid: {cent:5.1f}Hz | Text: \"{s.text.strip()}\"")
