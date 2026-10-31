import os
import torch
import soundfile as sf
import numpy as np
from transformers import AutoFeatureExtractor, Wav2Vec2ForSequenceClassification
import av
import io

def decode_audio(path):
    if path.endswith(".wav"):
        data, sr = sf.read(path)
        if len(data.shape) > 1:
            data = data.mean(axis=1)
        if sr != 16000:
            import torchaudio
            t = torch.tensor(data, dtype=torch.float32).unsqueeze(0)
            t = torchaudio.transforms.Resample(sr, 16000)(t)
            data = t.squeeze(0).numpy()
        return data
    container = av.open(path)
    resampler = av.AudioResampler(format="flt", layout="mono", rate=16000)
    pcm = []
    for frame in container.decode(audio=0):
        rf = resampler.resample(frame)
        for sub_frame in (rf if isinstance(rf, list) else [rf]):
            arr = sub_frame.to_ndarray()
            pcm.append(arr.reshape(-1))
    return np.concatenate(pcm).astype(np.float32)

models_to_test = [
    ("Real Multilingual", "results_real_multilingual"),
    ("4447 Model", "results_4447_models"),
]

samples = [
    ("User Voice M4A (REAL)", r"C:\Users\souna\.gemini\antigravity\brain\66a12257-e1a0-45be-b045-dd87c5fd01de\.user_uploaded\uploaded_media_1793365290300.m4a"),
    ("User Voice WAV (REAL)", "user_test_voice_16k.wav"),
    ("Real Friend (REAL)", "data_demo/unseen_test/unseen_real_friend.wav"),
    ("Fake SBI OTP (FAKE)", "data_demo/fake/scam_sbi_otp.wav"),
    ("Fake Card CVV (FAKE)", "data_demo/fake/scam_card_cvv.wav"),
    ("Fake UPI PIN (FAKE)", "data_demo/fake/scam_upi_pin.wav"),
    ("Fake KYC Hindi (FAKE)", "data_demo/fake/scam_kyc_hindi.wav"),
]

fe = AutoFeatureExtractor.from_pretrained("facebook/wav2vec2-base")

for m_name, m_path in models_to_test:
    if not os.path.exists(m_path):
        continue
    print(f"\n=== Model: {m_name} ({m_path}) ===")
    model = Wav2Vec2ForSequenceClassification.from_pretrained(m_path).to("cuda")
    model.eval()

    for s_name, s_path in samples:
        audio = decode_audio(s_path)
        # 3.0s window
        w_size = 48000
        probs = []
        for start in range(0, max(1, len(audio) - w_size + 1), w_size):
            chunk = audio[start:start+w_size]
            if len(chunk) < w_size:
                chunk = np.pad(chunk, (0, w_size - len(chunk)))
            inp = fe(chunk, sampling_rate=16000, return_tensors="pt")["input_values"].to("cuda")
            with torch.no_grad():
                logits = model(inp).logits
                p_fake = torch.softmax(logits, dim=-1)[0, 1].item() * 100.0
                probs.append(p_fake)
        mean_fake = np.mean(probs) if probs else 0.0
        print(f"  {s_name:25} -> Fake Prob: {mean_fake:5.1f}% | Pred: {'FAKE' if mean_fake > 50 else 'REAL'}")
