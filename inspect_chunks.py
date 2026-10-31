import os
import sys
import librosa
import numpy as np
from faster_whisper import WhisperModel

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

asr = WhisperModel("base", device="cpu", compute_type="int8")
f6 = r"c:\AI\VoiceCloning\audio\WhatsApp Audio 2026-10-30 at 11.41.33 PM.mpeg"
y, sr = librosa.load(f6, sr=16000)

print(f"Total duration: {len(y)/16000:.2f}s")
for start_sec in range(0, int(len(y)/16000), 15):
    end_sec = min(start_sec + 15, len(y) // 16000)
    chunk = y[start_sec*16000 : end_sec*16000]
    rms = float(np.sqrt(np.mean(chunk**2)))
    segments, info = asr.transcribe(chunk, beam_size=3)
    text = " ".join([s.text for s in segments]).strip()
    print(f"[{start_sec:02d}s - {end_sec:02d}s] (rms={rms:.4f}, lang={info.language}): \"{text}\"")
