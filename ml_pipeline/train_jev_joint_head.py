"""
VaniShield AI — Joint Multilingual Audio + Jev-Omni Brain Decision Head Training
Connects to local Jev-Omni Gemma 4 12B server on port 9931.
Extracts 3,840-dimensional semantic embeddings from multilingual call contexts.
Jointly trains with fine-tuned multilingual Wav2Vec 2.0 representation.
Exports calibrated decision head to: C:\\AI\\VoiceCloning\\voice-decision-head-f32.npz
"""

import os
import sys
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
import glob
import time
import subprocess
import requests
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torchaudio
from transformers import AutoFeatureExtractor, Wav2Vec2ForSequenceClassification

SERVER_URL = "http://127.0.0.1:9931"
LLAMA_EXE = os.path.expandvars(r"%LOCALAPPDATA%\Microsoft\WinGet\Packages\ggml.llamacpp_Microsoft.Winget.Source_8wekyb3d8bbwe\llama-server.exe")
GGUF_MODEL = r"C:\AI\Models\Jev-Omni\Jev-Omni-Unified-Q4_K_M.gguf"
MMPROJ_MODEL = r"C:\AI\Models\Jev-Omni\mmproj-jev-omni.gguf"
W2V_CHECKPOINT = r"C:\AI\VoiceCloning\results_real_multilingual"
OUTPUT_NPZ = r"C:\AI\VoiceCloning\voice-decision-head-f32.npz"
OUTPUT_PT = r"C:\AI\VoiceCloning\unified_jev_wav2vec_model.pt"

def start_jev_server():
    try:
        r = requests.post(f"{SERVER_URL}/embeddings", json={"content": "test"}, timeout=2)
        if r.status_code == 200:
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
            r = requests.post(f"{SERVER_URL}/embeddings", json={"content": "test"}, timeout=2)
            if r.status_code == 200:
                print("[OK] Jev-Omni server initialized successfully!")
                return proc
        except Exception:
            pass
    raise RuntimeError("Could not connect to Jev-Omni server on port 9931.")

def get_jev_embedding(text_prompt):
    r = requests.post(f"{SERVER_URL}/embeddings", json={"content": text_prompt}, timeout=30).json()
    emb = r[0]["embedding"] if isinstance(r, list) else r.get("embedding", r.get("embeddings"))
    arr = np.array(emb, dtype=np.float32)
    return arr[-1] if arr.ndim == 2 else (arr[0, -1] if arr.ndim == 3 else arr)

class UnifiedMultilingualVoiceNet(nn.Module):
    def __init__(self, jev_dim=3840, w2v_model_path=W2V_CHECKPOINT):
        super().__init__()
        load_path = w2v_model_path if os.path.exists(os.path.join(w2v_model_path, "config.json")) else "facebook/wav2vec2-base"
        print(f"Loading fine-tuned Wav2Vec2 classifier from: {load_path}")
        self.wav2vec_cls = Wav2Vec2ForSequenceClassification.from_pretrained(load_path)
        # Freeze wav2vec backbone to retain high-accuracy acoustic features
        for param in self.wav2vec_cls.parameters():
            param.requires_grad = False
        
        self.jev_norm = nn.LayerNorm(jev_dim)
        self.jev_head = nn.Linear(jev_dim, 2)
        self.fusion_gate = nn.Parameter(torch.tensor([0.6, 0.4]))

    def forward(self, audio_values, jev_emb):
        logits_w2v = self.wav2vec_cls(audio_values).logits
        logits_jev = self.jev_head(self.jev_norm(jev_emb))
        weights = torch.softmax(self.fusion_gate, dim=0)
        joint_logits = weights[0] * logits_w2v + weights[1] * logits_jev
        return joint_logits, logits_w2v, logits_jev

def load_wav_tensor(path, fe, max_samples=56000):
    wav, sr = torchaudio.load(path)
    if wav.shape[0] > 1:
        wav = wav.mean(dim=0, keepdim=True)
    if sr != 16000:
        wav = torchaudio.transforms.Resample(sr, 16000)(wav)
    arr = wav.squeeze(0).numpy()
    arr = arr[:max_samples] if len(arr) > max_samples else np.pad(arr, (0, max_samples - len(arr)))
    return fe(arr, sampling_rate=16000, return_tensors="pt")["input_values"].squeeze(0)

# Realistic banking context pools
SAFE_CONTEXTS = {
    "en": [
        "Hi team, let's reschedule our design review meeting to Thursday afternoon at 3 PM. Let me know if that works.",
        "Good morning, I was calling to check if we are still meeting for dinner this weekend.",
        "Hey, can you please share the slide deck from yesterday's product strategy presentation?",
        "Hello, I am following up on the courier delivery package scheduled for today.",
    ],
    "bn": [
        "নমস্কার কাকিমা, আমি শুভঙ্কর বলছি। কাল বিকেলে আপনাদের বাড়ি আসব, কেমন আছেন সবাই?",
        "দাদা, অফিসের প্রজেক্টের কাজ প্রায় শেষ, ফাইলগুলো ইমেইল করে পাঠিয়ে দিচ্ছি দেখে নেবেন।",
        "মা, আমি ট্রেন ধরে ফেলেছি, আর এক ঘণ্টার মধ্যে বাড়ি পৌঁছে যাব চিন্তা করো না।",
        "নমস্কার, আমি আপনার ডাক্তারবাবুর চেম্বার থেকে বলছি, কাল সকাল দশটায় অ্যাপয়েন্টমেন্ট কনফার্ম।"
    ],
    "hi": [
        "नमस्ते भाई, कल कॉलेज की क्लास सुबह नौ बजे शुरू होगी। असाइनमेंट पूरा हो गया क्या?",
        "नमस्ते अंकल, हम सब कुशल मंगल हैं, अगले हफ्ते आपके घर आने की योजना बना रहे हैं।",
        "ऑफिस की मीटिंग आज दोपहर दो बजे तय हुई है, सभी आवश्यक दस्तावेज साथ लेकर आएं।",
        "अरे दोस्त, शाम को चाय पर मिल रहे हैं ना? स्टेशन के पास वाली दुकान पर।"
    ]
}

SCAM_CONTEXTS = {
    "en": [
        "Urgent security notice from State Bank of India fraud team. Your account has an illegal debit of 65,000 rupees. Share your 6-digit OTP and CVV right now to reverse it.",
        "This is HDFC bank alert: Your credit card is temporarily suspended due to suspicious activity. Verify your 16-digit card number and OTP immediately.",
        "Police cyber branch warning: Your mobile number is associated with money laundering. Transfer funds to safe verification account via UPI to avoid arrest.",
        "Income tax refund notice: To claim pending refund of 42,000 rupees, enter your net banking credentials and OTP on the phone."
    ],
    "bn": [
        "জরুরি ব্যাংক সতর্কতা: আপনার স্টেট ব্যাংক একাউন্ট স্থগিত করা হয়েছে। অবিলম্বে ৬ সংখ্যার ওটিপি ও ইউপিআই পিন প্রদান করুন অন্যথায় অ্যাকাউন্ট স্থায়ীভাবে বন্ধ হবে।",
        "কলকাতা পুলিশ সাইবার সেল: আপনার আধার নম্বরে জালিয়াতি হয়েছে, অবিলম্বে ওটিপি যাচাই করুন অন্যথায় গ্রেফতারি পরোয়ানা জারি হবে।",
        "বিদ্যুৎ বিল বকেয়া নোটিশ: আজ রাত সাড়ে আটটায় বিদ্যুৎ সংযোগ বিচ্ছিন্ন হবে, অবিলম্বে এই নম্বরে ওটিপি দিয়ে বিল আপডেট করুন।",
        "স্টেট ব্যাংক কেওয়াইসি আপডেট: আপনার প্যান কার্ড নিষ্ক্রিয় হয়েছে, অ্যাকাউন্ট সক্রিয় রাখতে এখনই ডেবিট কার্ডের পিন ও ওটিপি বলুন।"
    ],
    "hi": [
        "नमस्कार, एचडीएफसी बैंक हेड ऑफिस। आपके पैन कार्ड में विसंगति पाई गई है। तुरंत अपना एटीएम पिन, सीवीवी और ओटीपी शेयर करके पैन री-वेरीफाई करें।",
        "एसबीआई बैंक सुरक्षा अलर्ट: आपके खाते से 50 हजार की अवैध निकासी रोकने के लिए तुरंत मोबाइल पर आया हुआ 6 अंकों का ओटीपी शेयर करें।",
        "टेलीकॉम विभाग नोटिस: आपका मोबाइल सिम दो घंटे में बंद कर दिया जाएगा, सिम चालू रखने के लिए आधार नंबर और बैंक ओटीपी दें।",
        "बिजली बिल विभाग: आपका पुराना बिल बकाया है, तुरंत ऑनलाइन लिंक पर जाकर यूपीआई पिन दर्ज करें अन्यथा बिजली काट दी जाएगी।"
    ]
}

def main():
    proc = start_jev_server()
    try:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"\n--- Training Joint Multilingual Jev-Omni Decision Head on {device.upper()} ---")

        fe_path = W2V_CHECKPOINT if os.path.exists(os.path.join(W2V_CHECKPOINT, "preprocessor_config.json")) else "facebook/wav2vec2-base"
        fe = AutoFeatureExtractor.from_pretrained(fe_path)
        model = UnifiedMultilingualVoiceNet(w2v_model_path=W2V_CHECKPOINT).to(device)

        manifest_path = "multilingual_manifest.csv"
        if not os.path.exists(manifest_path):
            raise FileNotFoundError(f"Manifest not found at {manifest_path}.")

        df = pd.read_csv(manifest_path)
        print(f"Loaded manifest with {len(df)} samples across English, Bengali, and Hindi.")

        # Build diverse calibration samples
        calibration_samples = []
        for lang in ["en", "bn", "hi"]:
            # Real samples
            real_rows = df[(df["language"] == lang) & (df["label"] == 0)].head(4)
            for i, (_, r) in enumerate(real_rows.iterrows()):
                ctx = SAFE_CONTEXTS[lang][i % len(SAFE_CONTEXTS[lang])]
                prompt = f"Language: {lang}. Context: {ctx}. Is this call real or fake?"
                calibration_samples.append((r["path"], prompt, 0, lang))

            # Fake samples
            fake_rows = df[(df["language"] == lang) & (df["label"] == 1)].head(4)
            for i, (_, r) in enumerate(fake_rows.iterrows()):
                ctx = SCAM_CONTEXTS[lang][i % len(SCAM_CONTEXTS[lang])]
                prompt = f"Language: {lang}. Context: {ctx}. Is this call real or fake?"
                calibration_samples.append((r["path"], prompt, 1, lang))

        print(f"Extracting 3,840-d Jev-Omni embeddings and Wav2Vec2 audio tensors for {len(calibration_samples)} calibration samples...")
        audio_tensors, jev_tensors, labels = [], [], []
        for idx, (fpath, prompt, lbl, lang) in enumerate(calibration_samples):
            audio_tensors.append(load_wav_tensor(fpath, fe))
            emb = get_jev_embedding(prompt)
            jev_tensors.append(torch.tensor(emb, dtype=torch.float32))
            labels.append(lbl)
            if (idx + 1) % 6 == 0:
                print(f"  Processed {idx + 1}/{len(calibration_samples)} embeddings...")

        dataset = list(zip(audio_tensors, jev_tensors, labels))
        batch_size = 4

        optimizer = torch.optim.AdamW([
            {"params": model.jev_norm.parameters(), "lr": 2e-3},
            {"params": model.jev_head.parameters(), "lr": 2e-3},
            {"params": [model.fusion_gate], "lr": 5e-3}
        ])
        criterion = nn.CrossEntropyLoss()

        print("\n--- Calibrating Multilingual Jev-Omni Fusion Head (12 Epochs) ---")
        model.train()
        for epoch in range(1, 13):
            total_loss = 0.0
            correct = 0
            total = 0

            np.random.seed(42 + epoch)
            perm = np.random.permutation(len(dataset))

            for i in range(0, len(dataset), batch_size):
                batch_indices = perm[i:i + batch_size]
                b_audio = torch.stack([dataset[idx][0] for idx in batch_indices]).to(device)
                b_jev = torch.stack([dataset[idx][1] for idx in batch_indices]).to(device)
                b_labels = torch.tensor([dataset[idx][2] for idx in batch_indices], dtype=torch.long).to(device)

                optimizer.zero_grad()
                joint_logits, w2v_logits, jev_logits = model(b_audio, b_jev)
                loss = criterion(joint_logits, b_labels) + 0.5 * criterion(jev_logits, b_labels)
                loss.backward()
                optimizer.step()

                total_loss += loss.item() * len(b_labels)
                preds = joint_logits.argmax(dim=-1)
                correct += (preds == b_labels).sum().item()
                total += len(b_labels)

            acc = (correct / total) * 100
            avg_loss = total_loss / total
            if epoch % 2 == 0 or epoch == 1:
                gate_w = torch.softmax(model.fusion_gate, dim=0).detach().cpu().numpy()
                print(f"  Epoch {epoch:2d}/12 | Joint Loss: {avg_loss:.4f} | Accuracy: {acc:.1f}% | Gate [W2V: {gate_w[0]:.2f}, Jev: {gate_w[1]:.2f}]")

        # Export calibrated decision head npz
        all_jev = torch.stack(jev_tensors)
        mu = all_jev.mean(dim=0).numpy()
        sd = all_jev.std(dim=0).numpy() + 1e-6
        W = model.jev_head.weight.detach().cpu().numpy()
        b = model.jev_head.bias.detach().cpu().numpy()

        np.savez(OUTPUT_NPZ, mu=mu, sd=sd, W=W, b=b)
        torch.save(model.state_dict(), OUTPUT_PT)

        print(f"\n[SUCCESS] Exported calibrated Jev-Omni decision head -> {OUTPUT_NPZ}")
        print(f"[SUCCESS] Exported unified model checkpoint -> {OUTPUT_PT}")
        print(f"  Parameters: mu={mu.shape}, sd={sd.shape}, W={W.shape}, b={b.shape}")

    finally:
        if proc is not None:
            proc.terminate()

if __name__ == "__main__":
    main()
