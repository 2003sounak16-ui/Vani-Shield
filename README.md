# 🛡️ VaniShield AI (वाणीशील्ड)
### Real-Time Multi-Modal AI Voice Clone Defense & Automated Banking Circuit Breaker

[![Python 3.11](https://img.shields.io/badge/Python-3.11-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-2.0-009688.svg)](https://fastapi.tiangolo.com/)
[![React 18](https://img.shields.io/badge/React-18-61DAFB.svg)](https://react.dev/)
[![PyTorch CUDA](https://img.shields.io/badge/PyTorch-CUDA_11.8-EE4C2C.svg)](https://pytorch.org/)
[![Wav2Vec 2.0](https://img.shields.io/badge/Acoustic_Ear-Wav2Vec_2.0-FF6F00.svg)](https://huggingface.co/facebook/wav2vec2-base)
[![Jev-Omni 12B](https://img.shields.io/badge/Brain-Jev--Omni_12B_GGUF-4CAF50.svg)](https://github.com/ggerganov/llama.cpp)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Compliance](https://img.shields.io/badge/DPDP_Act_2023-Compliant_(Zero_Disk_Audio)-brightgreen.svg)]()

> **Built by Team CypherEye**  
> Protecting citizens from AI-synthesized impersonation calls, deepfake kidnapping scams, and banking vishing across Indic languages (Hindi, Bengali, Indian English).

---

## 🏛️ Project Directory Architecture

```text
C:\AI\VoiceCloning\
├── README.md                              # Root project overview & startup commands
├── requirements.txt                       # Verified Python dependencies
├── check_env.py                           # Hardware diagnostic script (CUDA & packages)
├── scammer_blacklist.json                 # Persistent scammer tracking records
├── caller_tracker.py                      # Carrier lookup, IP-API geolocation & scam marking
│
├── docs\                                  # Technical Documentation
│   ├── KNOWLEDGE_BASE.md                  # Comprehensive AI handover & technical specs
│   ├── TEAM_INSTALLATION_AND_USER_MANUAL.md # Setup & troubleshooting manual
│   └── PROJECT_PROGRESS_AND_ROADMAP.md    # Chronological progress log & milestones
│
├── demo_samples\                          # Labeled Audio Dataset for Demos & Pitches
│   ├── real\                              # Authentic human voice clips (.wav)
│   ├── fake\                              # Neural AI-cloned scam audio (.wav via edge-tts)
│   └── unseen_test\                       # Held-out evaluation audio for live judging
│
├── ml_pipeline\                           # ML Training, Calibration & Inference
│   ├── training.py                        # Standalone Wav2Vec2 trainer (FP16)
│   ├── predict.py                         # Single-file CUDA inference script
│   ├── joint_train_jev_omni.py            # Simultaneous PyTorch joint trainer
│   ├── calibrate_joint_model.py           # Calibration script with differential learning rates
│   ├── test_joint_scam.py                 # Live inference test for the dual-brain model
│   ├── test_unseen_calls.py               # Generalization test on unseen test audio
│   ├── create_pilot_data.py               # Pilot audio generator
│   ├── generate_scam_data.py              # Neural scam call generator
│   ├── record_mic.py                      # 4-second microphone voice recorder
│   ├── manifests\                         # CSV manifests for training data loaders
│   └── checkpoints\                       # Saved model checkpoints & weights
│       ├── unified_jev_wav2vec_model.pt   # Trained PyTorch unified dual-brain model
│       └── voice-decision-head-f32.npz    # Custom Jev-Omni FP32 voice decision head
│
├── backend\                               # Real-Time Streaming Server (FastAPI)
│   ├── main.py                            # FastAPI app with CORS & WebSocket /ws/live-call
│   ├── app_demo.py                        # Standalone self-contained backend + HTML dashboard
│   ├── api\                               # Modular route handlers (ws_stream.py, banking_hold.py)
│   └── engine\                            # In-memory buffer & risk fusion logic
│
└── frontend\                              # Interactive User Dashboard (React + Vite)
    ├── package.json                       # Dependencies (react, tailwindcss, lucide-react, leaflet)
    ├── vite.config.js                     # Vite build configuration & proxy settings
    ├── index.html                         # HTML entry mount with Tailwind & Leaflet styles
    └── src\
        ├── main.jsx                       # React root mount
        ├── App.jsx                        # 3-Panel Split-Screen dashboard (Call + Map + Banking)
        └── components\                    # Reusable React components
            ├── CallScreen.jsx             # Live call card, waveform visualizer & risk meter
            ├── ScammerMap.jsx             # Interactive Leaflet map with green/red scammer pins
            └── MockBankingApp.jsx         # UPI/Bank payment screen with auto-locking button
```

---

## 🚀 Quick Startup Guide

### 0. Install Dependencies
```powershell
# Install backend Python dependencies
pip install -r requirements.txt

# Install frontend Node packages
cd frontend
npm install
cd ..
```

### 1. Run Hardware Diagnostic
```powershell
python check_env.py
```
*Checks Python 64-bit, NVIDIA RTX 3050 CUDA acceleration, FP16 support, and dependencies.*

### 2. Launch Jev-Omni 12B Intent Server (Port 9931)
```powershell
& "$env:LOCALAPPDATA\Microsoft\WinGet\Packages\ggml.llamacpp_Microsoft.Winget.Source_8wekyb3d8bbwe\llama-server.exe" `
  --model "C:\AI\Models\Jev-Omni\Jev-Omni-Unified-Q4_K_M.gguf" `
  --mmproj "C:\AI\Models\Jev-Omni\mmproj-jev-omni.gguf" `
  --ctx-size 512 `
  --batch-size 128 `
  --ubatch-size 128 `
  --gpu-layers 16 `
  --embedding `
  --pooling none `
  --port 9931
```

### 3. Launch FastAPI Streaming Backend (Port 8000)
```powershell
python -m uvicorn backend.main:app --host 127.0.0.1 --port 8000 --reload
```
*Interactive Swagger Documentation*: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)  
*Health Check*: [http://127.0.0.1:8000/health](http://127.0.0.1:8000/health)  
*Architecture Note*: `backend.main:app` provides the modular streaming architecture for live demos and interactive judging, featuring real-time WebSockets, in-memory circuit breakers, and DPDP-compliant telemetry. For standalone monolithic telephony evaluation, use `python app.py`.

### 4. Launch React + Vite Dashboard (Port 5173)
```powershell
cd frontend
npm run dev
```
*Live 3-Panel Dashboard*: [http://localhost:5173](http://localhost:5173)

---

## 🧪 Verification & Testing Commands

### Live Dual-Brain Joint Scam Test
```powershell
python ml_pipeline\test_joint_scam.py
```
*Verifies joint inference across Wav2Vec 2.0 acoustic forensic model and Jev-Omni 12B decision head, evaluating audio frames and transcribed conversational intent.*

### Unseen Calls Generalization Test
```powershell
python test_unseen_calls.py
```
*Evaluates model accuracy on held-out test clips in `demo_samples/unseen_test/`.*

### Telecom Carrier & Geolocation Lookup
```powershell
python caller_tracker.py --lookup "+911409988776"
python caller_tracker.py --list
```

---

## 🔒 India DPDP Act 2023 Compliance
- **Zero Raw Audio on Disk**: Audio payloads stream entirely inside volatile in-memory ring buffers (`EphemeralRAMBuffer`).
- **Configurable Ephemeral Eviction Window**: Unprocessed or evaluated speech segments automatically expire from volatile memory (configured 4.0s retention window with background eviction).
- **HMAC Pseudonymization**: Caller identities are cryptographically pseudonymized using keyed HMAC-SHA256 (`hash_caller_identity`). Raw phone numbers and unredacted call transcripts are never persisted in blacklist records.

---

## 👥 Team CypherEye
- **Lead AI Systems Engineer**: Sounak Banerjee ([@2003sounak16-ui](https://github.com/2003sounak16-ui))
- **Documentation**: Comprehensive technical references are available in the [`docs/`](docs/) directory.

## 📄 License
Licensed under the [MIT License](LICENSE).
