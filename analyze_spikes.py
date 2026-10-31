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

target_files = [
    r"c:\AI\VoiceCloning\audio\WhatsApp Audio 2026-10-30 at 11.30.22 PM.mpeg", # File 4
    r"c:\AI\VoiceCloning\audio\WhatsApp Audio 2026-10-30 at 11.38.38 PM.mpeg", # File 5
    r"c:\AI\VoiceCloning\audio\WhatsApp Audio 2026-10-30 at 11.41.33 PM.mpeg", # File 6
]

for fpath in target_files:
    fname = os.path.basename(fpath)
    y, sr = librosa.load(fpath, sr=16000)
    print(f"\n=======================================================")
    print(f"SPIKE ANALYSIS FOR: {fname}")
    win_len = 48000 # 3 sec
    step = 32000 # 2 sec
    for s in range(0, len(y) - win_len + 1, step):
        chunk = y[s : s + win_len]
        rms = float(np.sqrt(np.mean(chunk**2)))
        inp = fe(chunk, sampling_rate=16000, return_tensors="pt")["input_values"].to(device)
        with torch.no_grad():
            logits = model(inp).logits
            fake_p = float(torch.softmax(logits, dim=-1)[0, 1].cpu()) * 100.0
        
        # If fake_p > 50 or every few windows
        sec_start = s / 16000.0
        sec_end = (s + win_len) / 16000.0
        if fake_p > 50.0:
            segs, _ = asr.transcribe(chunk, beam_size=1)
            tx = " ".join([seg.text for seg in segs]).strip()
            print(f"  [HIGH FAKE SPIKE {fake_p:5.1f}%] {sec_start:4.1f}s - {sec_end:4.1f}s | RMS={rms:.4f} | Audio Text: \"{tx}\"")
        elif fake_p < 5.0 and (s // step) % 4 == 0:
            print(f"  [CLEAN REAL     {fake_p:5.1f}%] {sec_start:4.1f}s - {sec_end:4.1f}s | RMS={rms:.4f}")
