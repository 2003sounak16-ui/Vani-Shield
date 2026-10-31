import os
import numpy as np
import soundfile as sf
import pandas as pd

os.makedirs("data_pilot", exist_ok=True)
sr = 16000
duration = 3.0
samples = int(sr * duration)

records = []
for i in range(1, 5):
    t = np.linspace(0, duration, samples, endpoint=False)
    audio = (0.3 * np.sin(2 * np.pi * (150 + i * 20) * t) + 0.15 * np.sin(2 * np.pi * 300 * t)).astype(np.float32)
    path = f"data_pilot/pilot_real_{i}.wav"
    sf.write(path, audio, sr)
    records.append({"path": path, "label": "real", "group": f"speaker_{i}"})

for i in range(1, 5):
    t = np.linspace(0, duration, samples, endpoint=False)
    audio = (0.25 * np.sign(np.sin(2 * np.pi * (220 + i * 15) * t)) + 0.05 * np.random.normal(0, 0.1, samples)).astype(np.float32)
    path = f"data_pilot/pilot_fake_{i}.wav"
    sf.write(path, audio, sr)
    records.append({"path": path, "label": "fake", "group": f"engine_{i}"})

df = pd.DataFrame(records)
df.to_csv("pilot_manifest.csv", index=False)
print(f"Created pilot dataset with {len(df)} samples in C:\\AI\\VoiceCloning\\data_pilot")
