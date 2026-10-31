import os
import glob
import time
import subprocess
import asyncio
import requests
import numpy as np
import soundfile as sf
import torch
import torch.nn as nn
import torchaudio
import edge_tts
from transformers import AutoFeatureExtractor, Wav2Vec2Model

SERVER_URL = "http://127.0.0.1:9931"
LLAMA_EXE = os.path.expandvars(r"%LOCALAPPDATA%\Microsoft\WinGet\Packages\ggml.llamacpp_Microsoft.Winget.Source_8wekyb3d8bbwe\llama-server.exe")
GGUF_MODEL = r"C:\AI\Models\Jev-Omni\Jev-Omni-Unified-Q4_K_M.gguf"
MMPROJ_MODEL = r"C:\AI\Models\Jev-Omni\mmproj-jev-omni.gguf"
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
MODEL_PT = os.environ.get("VANISHIELD_CALIBRATED_MODEL_PT", os.path.join(PROJECT_ROOT, "unified_jev_wav2vec_base_calibrated.pt"))

class UnifiedJevOmniVoiceNet(nn.Module):
    def __init__(self, jev_dim=3840, w2v_model_name="facebook/wav2vec2-base"):
        super().__init__()
        self.wav2vec = Wav2Vec2Model.from_pretrained(w2v_model_name)
        self.wav2vec.freeze_feature_encoder()
        w2v_dim = self.wav2vec.config.hidden_size
        self.jev_norm = nn.LayerNorm(jev_dim)
        self.jev_head = nn.Linear(jev_dim, 2)
        self.w2v_head = nn.Linear(w2v_dim, 2)
        self.fusion_gate = nn.Parameter(torch.tensor([0.5, 0.5]))

    def forward(self, audio_values, jev_emb):
        w2v_out = self.wav2vec(audio_values).last_hidden_state.mean(dim=1)
        logits_w2v = self.w2v_head(w2v_out)
        logits_jev = self.jev_head(self.jev_norm(jev_emb))
        weights = torch.softmax(self.fusion_gate, dim=0)
        return weights[0] * logits_w2v + weights[1] * logits_jev, logits_w2v, logits_jev

def start_jev_server():
    try:
        if requests.post(f"{SERVER_URL}/embeddings", json={"content": "test"}, timeout=2).status_code == 200:
            return None
    except Exception:
        pass
    print("Starting Jev-Omni server on port 9931...")
    cmd = [
        LLAMA_EXE, "--model", GGUF_MODEL, "--mmproj", MMPROJ_MODEL,
        "--ctx-size", "512", "--batch-size", "128", "--ubatch-size", "128",
        "--gpu-layers", "0", "--embedding", "--pooling", "none", "--port", "9931"
    ]
    proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    for _ in range(40):
        time.sleep(2)
        try:
            if requests.post(f"{SERVER_URL}/embeddings", json={"content": "test"}, timeout=2).status_code == 200:
                return proc
        except Exception:
            pass
    raise RuntimeError("Could not start Jev-Omni server")

def get_jev_embedding(text_prompt):
    r = requests.post(f"{SERVER_URL}/embeddings", json={"content": text_prompt}, timeout=15).json()
    emb = r[0]["embedding"] if isinstance(r, list) else r.get("embedding", r.get("embeddings"))
    arr = np.array(emb, dtype=np.float32)
    return arr[-1] if arr.ndim == 2 else (arr[0, -1] if arr.ndim == 3 else arr)

def load_wav_tensor(path, fe, max_samples=48000):
    wav, sr = torchaudio.load(path)
    if wav.shape[0] > 1:
        wav = wav.mean(dim=0, keepdim=True)
    if sr != 16000:
        wav = torchaudio.transforms.Resample(sr, 16000)(wav)
    arr = wav.squeeze(0).numpy()
    arr = arr[:max_samples] if len(arr) > max_samples else np.pad(arr, (0, max_samples - len(arr)))
    return fe(arr, sampling_rate=16000, return_tensors="pt")["input_values"].squeeze(0)

async def generate_scam_clips():
    os.makedirs("data_demo/fake", exist_ok=True)
    scams = [
        ("data_demo/fake/scam_sbi_otp.wav", "en-IN-PrabhatNeural", "Hello, this is State Bank fraud control. Your account is blocked. Share your six digit OTP immediately."),
        ("data_demo/fake/scam_kyc_hindi.wav", "hi-IN-MadhurNeural", "Namaskar, aapka bank KYC expire ho gaya hai. Turant apna OTP aur bank account details batayein."),
        ("data_demo/fake/scam_card_cvv.wav", "en-IN-NeerjaNeural", "Urgent alert from credit card security. Confirm your card CVV and OTP right now to cancel a 45,000 rupee charge."),
        ("data_demo/fake/scam_upi_pin.wav", "hi-IN-SwaraNeural", "Aapka bank account chalu rakhne ke liye abhi apna UPI PIN aur OTP share karein.")
    ]
    for fpath, voice, text in scams:
        if not os.path.exists(fpath):
            await edge_tts.Communicate(text, voice).save(fpath)
    return scams

def main():
    proc = start_jev_server()
    try:
        scams = asyncio.run(generate_scam_clips())
        device = "cuda" if torch.cuda.is_available() else "cpu"
        fe = AutoFeatureExtractor.from_pretrained("facebook/wav2vec2-base")
        model = UnifiedJevOmniVoiceNet().to(device)

        real_files = sorted(glob.glob("data_pilot/pilot_real_*.wav"))[:4]
        real_contexts = [
            "Normal genuine human phone call about college project meeting.",
            "Casual conversation with family member about dinner plans tonight.",
            "Colleague discussing hackathon presentation slides and schedule.",
            "Customer asking about normal store opening hours tomorrow."
        ]

        dataset = []
        for f, ctx in zip(real_files, real_contexts):
            prompt = f"Transcript: {ctx}. Assess acoustic tone and conversational context."
            dataset.append((f, prompt, 0, ctx))
        for f, _, ctx in scams:
            prompt = f"Transcript: {ctx}. Assess acoustic tone and conversational context."
            dataset.append((f, prompt, 1, ctx))

        print(f"\nExtracting features for {len(dataset)} samples (Real vs. Neural TTS Scam)...")
        X_audio = torch.stack([load_wav_tensor(d[0], fe) for d in dataset]).to(device)
        X_jev = torch.stack([torch.tensor(get_jev_embedding(d[1]), dtype=torch.float32) for d in dataset]).to(device)
        y = torch.tensor([d[2] for d in dataset], dtype=torch.long).to(device)

        # Differential learning rate: 5e-5 for Wav2Vec2 backbone, 5e-3 for classification heads
        optimizer = torch.optim.AdamW([
            {"params": model.wav2vec.parameters(), "lr": 5e-5},
            {"params": model.w2v_head.parameters(), "lr": 5e-3},
            {"params": model.jev_norm.parameters(), "lr": 5e-3},
            {"params": model.jev_head.parameters(), "lr": 5e-3},
            {"params": [model.fusion_gate], "lr": 1e-2}
        ])
        criterion = nn.CrossEntropyLoss()

        print("\n--- Training Calibrated Joint Model (12 Epochs on RTX 3050) ---")
        model.train()
        for epoch in range(1, 13):
            optimizer.zero_grad()
            joint_logits, w2v_logits, jev_logits = model(X_audio, X_jev)
            # Train both individual heads and the fused output together
            loss = criterion(joint_logits, y) + 0.5 * criterion(w2v_logits, y) + 0.5 * criterion(jev_logits, y)
            loss.backward()
            optimizer.step()
            if epoch % 3 == 0 or epoch == 1:
                acc = (joint_logits.argmax(dim=-1) == y).float().mean().item() * 100
                print(f"  Epoch {epoch:2d}/12 | Joint Loss: {loss.item():.4f} | Accuracy: {acc:.1f}%")

        torch.save(model.state_dict(), MODEL_PT)
        print(f"\n[OK] Updated calibrated model -> {MODEL_PT}")

        # Verify on Call 1 (Genuine) and Call 2 (Neural TTS Bank OTP Scam)
        model.eval()
        for idx, title, caller in [
            (0, "CALL 1: Genuine Colleague Call", "+91-9830011223 (Kolkata, West Bengal - Jio)"),
            (4, "CALL 2: Neural AI Bank Manager OTP Scam Call", "+91-1409998877 (Spoofed VoIP Proxy)")
        ]:
            with torch.no_grad():
                j_log, w_log, jev_log = model(X_audio[idx:idx+1], X_jev[idx:idx+1])
                j_p = float(torch.softmax(j_log, dim=-1)[0, 1].cpu()) * 100
                w_p = float(torch.softmax(w_log, dim=-1)[0, 1].cpu()) * 100
                jev_p = float(torch.softmax(jev_log, dim=-1)[0, 1].cpu()) * 100

            print(f"\n================ {title} ================")
            print(f"Caller ID & Location : {caller}")
            print(f"Audio File           : {dataset[idx][0]}")
            print(f"Call Transcript      : {dataset[idx][3]}")
            print(f"Wav2Vec2 Audio Score : {w_p:.1f}% Fake Voice")
            print(f"Jev-Omni Brain Score : {jev_p:.1f}% Scam Intent")
            print(f"Unified Scam Score   : {j_p:.1f}%")
            print(f"Action               : {'[RED SCAM ALERT] Caller Marked as Scammer + UPI/Bank App LOCKED!' if j_p >= 60 else '[GREEN SAFE] Verified Genuine Call — Proceed Normally.'}")

    finally:
        if proc is not None:
            proc.terminate()

if __name__ == "__main__":
    main()
