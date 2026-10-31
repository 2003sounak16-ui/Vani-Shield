# VaniShield AI — Technical Knowledge Base & AI Handover
**Team CypherEye | Project VaniShield AI**  
*Document Version: 2.0.0 | Date: October 2026*

---

## 1. Executive Summary & Vision
**Project VaniShield AI** is an ultra-low-latency, real-time voice cloning defense and fraud mitigation system designed to detect and neutralize deepfake vishing (voice phishing) attacks in the Indian telecommunications and banking ecosystem. By fusing acoustic neural speech forensics with contextual semantic intent reasoning, VaniShield AI protects citizens from financial fraud, unauthorized UPI transfers, and cyber extortion.

---

## 2. Core Architecture: Dual-Brain Forensic Arbitration

VaniShield AI operates on a synchronized **Dual-Brain** topology:

```
[ Incoming Telephony / VoIP Audio Stream ]
                   │
                   ▼
       [ Ephemeral Volatile RAM Ring Buffer ]
       (DPDP Act 2023 Compliant | Max 4.0s SLA)
                   │
         ┌─────────┴─────────┐
         ▼                   ▼
   [ BRAIN 1 ]          [ BRAIN 2 ]
Acoustic Forensics    Intent Reasoning
(Wav2Vec 2.0 FP16)     (Jev-Omni 12B)
   • 1.5s Windows      • Faster-Whisper ASR
   • 0.75s Hop Size    • 3840-dim Embeddings
   • Glottal Physics   • Trilingual Context
   • Codec Invariance    (en, hi, bn)
         │                   │
         └─────────┬─────────┘
                   ▼
   [ Weighted Risk Fusion Engine ]
   Risk_unified = 0.55 * Acoustic + 0.45 * Intent
                   │
                   ▼
  [ Automated Banking Circuit Breaker ]
      (Triggers when Risk >= 70.0%)
   ──► Instant UPI / Net Banking Hold
   ──► HLR Telecom Carrier Flagging
   ──► Real-Time React Telemetry Stream
```

### Brain 1: Acoustic Forensic Branch (Wav2Vec 2.0)
- **Model Checkpoint**: Fine-tuned Wav2Vec 2.0 sequence classifier (`results_master_unified/`).
- **Precision**: Half-precision FP16 on NVIDIA GeForce RTX 3050 Laptop GPU (6GB VRAM).
- **Inference Strategy**: Sliding multi-window analysis using 1.5-second evaluation slices with 0.75-second hop sizes.
- **Physical Signal Verification**:
  - Glottal fundamental frequency (F0) perturbation.
  - Spectral centroid, flatness, and high-frequency roll-off.
  - Energy threshold gating ($\text{RMS} \ge 0.010$) to eliminate ambient room noise and WhatsApp PTT microphone edge clicks.

### Brain 2: Semantic Intent & Fraud Reasoning Branch (Jev-Omni 12B)
- **Engine**: Jev-Omni 12B Unified model (`Jev-Omni-Unified-Q4_K_M.gguf` + `mmproj-jev-omni.gguf`) hosted on a local, hardware-accelerated `llama-server.exe` instance on port 9931.
- **ASR Transcription**: Fast CTranslate2 Whisper model with conditioned rep-penalty to prevent hallucination loops.
- **Language Coverage**: Seamless contextual understanding in English, Hindi, and Bengali.
- **Intent Vectors**: 3840-dimensional text embeddings evaluated against fine-tuned decision head weights (`voice-decision-head-f32.npz`).

---

## 3. Threat Fusion Formula & Circuit Breaker Logic

The Unified Scam Risk Score is calculated as:
$$\text{Risk}_{\text{unified}} = 0.55 \cdot \text{Acoustic}_{\text{clone}} + 0.45 \cdot \text{Intent}_{\text{scam}}$$

### Action Thresholds:
| Score Range | Threat Classification | Automated Response |
| :--- | :--- | :--- |
| **0.0% – 34.9%** | `AUTHENTIC_SAFE_CALL` | Green status; normal UPI/banking active. |
| **35.0% – 69.9%** | `SUSPICIOUS_CALL` | Amber warning banner on user screen; biometric confirmation required. |
| **70.0% – 100.0%** | `CRITICAL_SCAM` | **Banking Circuit Breaker Engaged**: Immediate lock of UPI & Net Banking. Caller recorded in `scammer_blacklist.json`. |

---

## 4. Privacy & Regulatory Compliance (DPDP Act 2023)
VaniShield AI strictly enforces India's **Digital Personal Data Protection (DPDP) Act 2023**:
1. **Zero Raw Audio Persistence**: Raw voice recordings are processed strictly inside volatile memory (`EphemeralRAMBuffer`). No temporary audio files are written to disk during live call monitoring.
2. **Auto-Eviction SLA**: In-memory ring buffer chunks automatically expire and are purged from volatile RAM within 4.0 seconds.
3. **Caller Pseudonymization**: Caller phone numbers are anonymized using HMAC-SHA256 (`hash_caller_identity`). Raw transcripts are stripped before persistent threat records are saved.
4. **Thread-Safe Atomic Storage**: Blacklist operations utilize mutual exclusion locks and atomic file replacement to prevent data corruption.

---

## 5. Hardware Specifications & Deployment Boundaries
- **Primary GPU**: NVIDIA GeForce RTX 3050 Laptop GPU (6GB GDDR6 VRAM, Compute Capability 8.6).
- **CUDA Runtime**: PyTorch 2.7.1+cu118 with cuDNN 9.1 acceleration.
- **Memory Footprint**:
  - Wav2Vec 2.0 Acoustic Model: ~380 MB VRAM.
  - Jev-Omni 12B (Q4_K_M): Offloaded to 16 GPU layers + system RAM via `llama-server.exe` (~4.8 GB RAM).
  - Faster-Whisper ASR: Int8 quantization on CPU (~150 MB RAM).
- **Latency SLA**: Total end-to-end inference turnaround under 400ms per audio evaluation window.
