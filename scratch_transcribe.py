import glob
import torchaudio
from vanishield_engine import core_engine
from predict import load_audio

files = sorted(glob.glob(r'C:\Users\souna\Downloads\WhatsApp Audio 2026-10-30*.ogg') + glob.glob(r'C:\Users\souna\Downloads\WhatsApp Audio 2026-10-30*.mpeg'))
for path in files:
    wav, sr = load_audio(path)
    if wav.shape[0] > 1:
        wav = wav.mean(dim=0, keepdim=True)
    if sr != 16000:
        wav = torchaudio.transforms.Resample(sr, 16000)(wav)
    audio_np = wav.squeeze(0).numpy()
    transcript, kw = core_engine.transcribe_and_spot_keywords(audio_np)
    fname = path.split('\\')[-1]
    print(f"{fname}:\n   {transcript}\n")
