import os
import sys
import librosa
import numpy as np

fpath = r"c:\AI\VoiceCloning\audio\WhatsApp Audio 2026-10-30 at 11.41.33 PM.mpeg"
y, sr = librosa.load(fpath, sr=16000)

print(f"File: {os.path.basename(fpath)}")
print(f"Sample count: {len(y)}, duration: {len(y)/sr:.2f}s")

# Let's check silence intervals / cuts / splices (abrupt RMS energy changes or silence gaps > 0.3s)
intervals = librosa.effects.split(y, top_db=25)
print(f"Found {len(intervals)} active voice segments (non-silent intervals):")
for idx, (st, en) in enumerate(intervals[:25]):
    dur = (en - st) / 16000.0
    print(f"  Segment {idx+1:02d}: {st/16000.0:5.2f}s to {en/16000.0:5.2f}s (duration: {dur:4.2f}s)")
