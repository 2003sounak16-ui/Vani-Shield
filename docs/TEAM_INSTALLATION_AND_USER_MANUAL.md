# VaniShield AI — Team Installation & User Manual
**Team CypherEye | Project VaniShield AI**

---

## 1. System Requirements & Prerequisites
- **Operating System**: Windows 11 (64-bit) or Windows 10 (Build 19045+)
- **GPU**: NVIDIA RTX 3050 (4GB or 6GB VRAM) with NVIDIA Game Ready/Studio Driver 530+
- **CUDA Toolkit**: CUDA 11.8 or CUDA 12.x
- **Python**: Version 3.10 or 3.11 (64-bit required)
- **Node.js**: Version 18.x or 20.x LTS (with `npm`)

---

## 2. Installation & Environment Setup

### A. Clone and Install Python Dependencies
Open Windows PowerShell inside `C:\AI\VoiceCloning\`:
```powershell
cd C:\AI\VoiceCloning
pip install -r requirements.txt
```

### B. Install Frontend Dependencies
```powershell
cd C:\AI\VoiceCloning\frontend
npm install
cd ..
```

### C. Run System Hardware Diagnostic
Verify CUDA acceleration, GPU memory, and dependencies:
```powershell
python check_env.py
```
*Expected output*: `Environment: READY`, `GPU Accel: ACTIVE (RTX 3050)`, `Dependencies: ALL SATISFIED`.

---

## 3. Starting the Platform Services

### Terminal 1: Launch Jev-Omni 12B Intent Server (Port 9931)

#### Prerequisites & Setup:
1. **llama-server.exe**:
   - Install via Windows Package Manager:
     ```powershell
     winget install ggml.llamacpp
     ```
   - Or download prebuilt release binaries from [llama.cpp releases](https://github.com/ggerganov/llama.cpp/releases) and place `llama-server.exe` in your PATH or tools directory.
2. **Model GGUF Files**:
   - Download `Jev-Omni-Unified-Q4_K_M.gguf` and `mmproj-jev-omni.gguf` to a local folder (e.g. `C:\AI\Models\Jev-Omni\`).
   - If stored elsewhere, update `--model` and `--mmproj` paths below to point to your files.

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

### Terminal 2: Launch FastAPI Streaming Backend (Port 8000)
```powershell
cd C:\AI\VoiceCloning
python -m uvicorn backend.main:app --host 127.0.0.1 --port 8000 --reload
```
*Backend Swagger Docs*: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)  
*Health Check*: [http://127.0.0.1:8000/health](http://127.0.0.1:8000/health)

### Terminal 3: Launch React Vite Frontend (Port 5173)
```powershell
cd C:\AI\VoiceCloning\frontend
npm run dev
```
*Live Dashboard*: [http://localhost:5173](http://localhost:5173)

---

## 4. Verification & Testing Commands

### Run Dual-Brain Live Inference Test
```powershell
python ml_pipeline\test_joint_scam.py
```
*Evaluates both `scam_sbi_otp.wav` and `call_en_safe.wav` with full forensic arbitration.*

### Run Unseen Audio Generalization Test
```powershell
python test_unseen_calls.py
```
*Evaluates held-out evaluation clips in `demo_samples/unseen_test/`.*

### Telecom Carrier & Geolocation Lookup
```powershell
python caller_tracker.py --lookup "+911409988776"
python caller_tracker.py --list
```

### 4-Second Live Voice Recorder
```powershell
python ml_pipeline\record_mic.py
```

---

## 5. Troubleshooting & FAQ

### Issue: "CUDA out of memory" on RTX 3050
- **Fix**: Reduce `--gpu-layers` on Jev-Omni from 16 to 8 or 0 (CPU offload). Wav2Vec2 only requires ~380MB VRAM, so keeping Jev-Omni layers around 8-12 preserves optimal performance on 4GB/6GB GPUs.

### Issue: "Port 8000 or 9931 already in use"
- Find PID: `netstat -ano | findstr 8000`
- Terminate process: `Stop-Process -Id <PID> -Force`

### Issue: "Warning: HF Hub unauthenticated requests"
- Set local environment token:
  `$env:HF_HUB_DISABLE_SYMLINKS_WARNING = "1"`
  `$env:TOKENIZERS_PARALLELISM = "false"`
