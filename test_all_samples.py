import glob
import torchaudio
from vanishield_engine import core_engine
from predict import load_audio

files = [
    (r"C:\AI\VoiceCloning\user_test_voice_16k.wav", "Sounak Real Recording"),
    (r"C:\Users\souna\Downloads\WhatsApp Audio 2026-10-30 at 7.05.18 PM.mpeg", "Lottery Real Bengali Voice"),
    (r"C:\Users\souna\Downloads\WhatsApp Audio 2026-10-30 at 7.10.44 PM.ogg", "Real Bengali Voice 1"),
    (r"C:\Users\souna\Downloads\WhatsApp Audio 2026-10-30 at 7.10.45 PM.ogg", "Real Bengali Voice 2"),
    (r"C:\Users\souna\Downloads\WhatsApp Audio 2026-10-30 at 7.16.21 PM.ogg", "Real Bengali Voice 3"),
    (r"C:\Users\souna\Downloads\WhatsApp Audio 2026-10-30 at 7.34.16 PM.ogg", "Siantika Voice")
]

print(f"{'Sample Label':30s} | Fake % | Risk % | Breaker | Status")
print("-" * 70)
for path, label in files:
    wav, sr = load_audio(path)
    if wav.shape[0] > 1:
        wav = wav.mean(dim=0, keepdim=True)
    if sr != 16000:
        wav = torchaudio.transforms.Resample(sr, 16000)(wav)
    audio_np = wav.squeeze(0).numpy()
    res = core_engine.evaluate_live_call(audio_np, phone_number="+91-0000000000 (User Upload)")
    cb = "LOCKED" if res['circuit_breaker']['engaged'] else "SAFE"
    status = "RED THREAT" if res['circuit_breaker']['engaged'] else "GREEN SAFE"
    w2v = res['w2v_fake_prob']
    risk = res['unified_risk_score']
    print(f"{label:30s} | {w2v:6.1f} | {risk:6.1f} | {cb:7s} | {status}")
