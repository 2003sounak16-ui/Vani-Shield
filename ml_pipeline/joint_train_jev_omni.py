import os
import glob
import time
import subprocess
import requests
import numpy as np
import torch
import torch.nn as nn
import torchaudio
from transformers import AutoFeatureExtractor, Wav2Vec2Model

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
SERVER_URL = "http://127.0.0.1:9931"
LLAMA_EXE = os.path.expandvars(r"%LOCALAPPDATA%\Microsoft\WinGet\Packages\ggml.llamacpp_Microsoft.Winget.Source_8wekyb3d8bbwe\llama-server.exe")
GGUF_MODEL = r"C:\AI\Models\Jev-Omni\Jev-Omni-Unified-Q4_K_M.gguf"
MMPROJ_MODEL = r"C:\AI\Models\Jev-Omni\mmproj-jev-omni.gguf"
ORIG_HEAD_NPZ = r"C:\AI\Models\Jev-Omni\decision-head-f32.npz"

# 1. Start Jev-Omni Server (with Multimodal Projector) if not running
def start_jev_server():
    try:
        if requests.post(f"{SERVER_URL}/embeddings", json={"content": "test"}, timeout=2).status_code == 200:
            print("[OK] Jev-Omni server already active on port 9931.")
            return None
    except Exception:
        pass

    print("Launching Jev-Omni + mmproj-jev-omni.gguf on port 9931...")
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
                print("[OK] Jev-Omni multimodal server ready!")
                return proc
        except Exception:
            pass
    raise RuntimeError("Could not start Jev-Omni llama-server.exe")

# 2. Extract 3,840-d Jev-Omni Embedding
def get_jev_embedding(text_prompt):
    r = requests.post(f"{SERVER_URL}/embeddings", json={"content": text_prompt}, timeout=15).json()
    emb = r[0]["embedding"] if isinstance(r, list) else r.get("embedding", r.get("embeddings"))
    arr = np.array(emb, dtype=np.float32)
    return arr[-1] if arr.ndim == 2 else (arr[0, -1] if arr.ndim == 3 else arr)

# 3. Single Unified PyTorch Architecture (Wav2Vec 2.0 + Jev-Omni Decision Head Together)
class UnifiedJevOmniVoiceNet(nn.Module):
    def __init__(self, jev_dim=3840, w2v_model_name="facebook/wav2vec2-base"):
        super().__init__()
        self.wav2vec = Wav2Vec2Model.from_pretrained(w2v_model_name)
        self.wav2vec.freeze_feature_encoder()  # Keep CNN frozen for 6GB RTX 3050 VRAM
        w2v_dim = self.wav2vec.config.hidden_size  # 768

        # Trainable Jev-Omni normalization & projection + Joint Fusion Head
        self.jev_norm = nn.LayerNorm(jev_dim)
        self.jev_head = nn.Linear(jev_dim, 2)
        self.w2v_head = nn.Linear(w2v_dim, 2)
        self.fusion_gate = nn.Parameter(torch.tensor([0.5, 0.5]))  # Learnable fusion weights

    def forward(self, audio_values, jev_emb):
        # Branch A: Wav2Vec 2.0 forward pass on raw audio
        w2v_out = self.wav2vec(audio_values).last_hidden_state.mean(dim=1)  # (B, 768)
        logits_w2v = self.w2v_head(w2v_out)

        # Branch B: Jev-Omni 3,840-d decision head forward pass
        logits_jev = self.jev_head(self.jev_norm(jev_emb))

        # Joint fused logits trained simultaneously
        weights = torch.softmax(self.fusion_gate, dim=0)
        return weights[0] * logits_w2v + weights[1] * logits_jev

def load_wav_tensor(path, fe, max_samples=48000):
    wav, sr = torchaudio.load(path)
    if wav.shape[0] > 1:
        wav = wav.mean(dim=0, keepdim=True)
    if sr != 16000:
        wav = torchaudio.transforms.Resample(sr, 16000)(wav)
    arr = wav.squeeze(0).numpy()
    arr = arr[:max_samples] if len(arr) > max_samples else np.pad(arr, (0, max_samples - len(arr)))
    return fe(arr, sampling_rate=16000, return_tensors="pt")["input_values"].squeeze(0)

def main():
    proc = start_jev_server()
    try:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"\n--- Building Unified Wav2Vec2 + Jev-Omni Model on {device.upper()} ---")

        fe = AutoFeatureExtractor.from_pretrained("facebook/wav2vec2-base")
        model = UnifiedJevOmniVoiceNet().to(device)

        # Load training audio files from demo_samples, data_pilot, or data_demo
        real_files = sorted(glob.glob("demo_samples/real/*.wav") + glob.glob("data_pilot/pilot_real_*.wav") + glob.glob("data_demo/real/*.wav"))
        fake_files = sorted(glob.glob("demo_samples/fake/*.wav") + glob.glob("data_pilot/pilot_fake_*.wav") + glob.glob("data_demo/fake/*.wav"))

        dataset = []
        for f in real_files:
            prompt = "Assess phone call audio and conversational context for potential spoofing."
            dataset.append((f, prompt, 0))
        for f in fake_files:
            prompt = "Assess phone call audio and conversational context for potential spoofing."
            dataset.append((f, prompt, 1))

        print(f"Preparing {len(dataset)} joint (Audio + Jev-Omni 3840-d) training samples...")
        audio_tensors, jev_tensors, labels = [], [], []
        for fpath, prompt, lbl in dataset:
            audio_tensors.append(load_wav_tensor(fpath, fe))
            jev_tensors.append(torch.tensor(get_jev_embedding(prompt), dtype=torch.float32))
            labels.append(lbl)

        X_audio = torch.stack(audio_tensors).to(device)
        X_jev = torch.stack(jev_tensors).to(device)
        y = torch.tensor(labels, dtype=torch.long).to(device)

        # Single Joint Optimizer & Loss Function training BOTH models together
        optimizer = torch.optim.AdamW(model.parameters(), lr=3e-5)
        criterion = nn.CrossEntropyLoss()

        print("\n--- Starting Joint End-to-End Training (Wav2Vec 2.0 + Jev-Omni Together) ---")
        model.train()
        for epoch in range(1, 6):
            optimizer.zero_grad()
            logits = model(X_audio, X_jev)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()

            preds = logits.argmax(dim=-1)
            acc = (preds == y).float().mean().item() * 100
            print(f"  Epoch {epoch}/5 | Joint Loss: {loss.item():.4f} | Joint Accuracy: {acc:.1f}%")

        # Save Joint Checkpoint + Jev-Omni .npz Decision Head to distinct paths
        model_out = os.environ.get("VANISHIELD_JOINT_MODEL_PT", os.path.join(PROJECT_ROOT, "unified_jev_omni_model.pt"))
        head_out = os.environ.get("VANISHIELD_JOINT_HEAD_NPZ", os.path.join(PROJECT_ROOT, "jev-omni-decision-head-f32.npz"))
        torch.save(model.state_dict(), model_out)
        mu = X_jev.mean(dim=0).detach().cpu().numpy()
        sd = (X_jev.std(dim=0).detach().cpu().numpy()) + 1e-6
        np.savez(
            head_out,
            mu=mu, sd=sd,
            W=model.jev_head.weight.detach().cpu().numpy(),
            b=model.jev_head.bias.detach().cpu().numpy(),
            norm_weight=model.jev_norm.weight.detach().cpu().numpy(),
            norm_bias=model.jev_norm.bias.detach().cpu().numpy()
        )
        print(f"\n[SUCCESS] Saved unified model -> {model_out}")
        print(f"[SUCCESS] Saved Jev-Omni head -> {head_out}")

    finally:
        if proc is not None:
            proc.terminate()

if __name__ == "__main__":
    main()
