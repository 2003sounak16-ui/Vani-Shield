"""
VaniShield AI — Mobile Edge Engine (Fix 1 & Fix 3)
Team CypherEye | Project VaniShield AI

Mobile-optimized, ultra-low latency Dual-Brain inference engine.
Replaces the heavy desktop PyTorch + 12B LLM pipeline with:
  1. Quantized INT8 Wav2Vec 2.0 ONNX running via ONNX Runtime Mobile / NNAPI / QNN (~45-90MB RAM).
  2. Mobile Distilled Multilingual Intent Scorer (<5MB RAM, <1ms latency).
  3. Integrated DPDP Act 2023 Ephemeral RAM Ring Buffer.
  4. Real-time Multi-Pillar Risk Fusion & Automated Banking Circuit Breaker.
"""

import os
import sys
import time
import json
import logging
import numpy as np
from typing import Dict, List, Optional, Tuple, Any

# Ensure project root is in sys.path
_CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(_CURRENT_DIR, "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

logger = logging.getLogger("mobile_edge_engine")

try:
    from backend.engine.caller_tracker import (
        hash_caller_identity,
        lookup_carrier,
        is_blacklisted,
        mark_scammer,
        BLACKLIST_PATH
    )
except ImportError:
    from caller_tracker import (
        hash_caller_identity,
        lookup_carrier,
        is_blacklisted,
        mark_scammer,
        BLACKLIST_PATH
    )

try:
    from ml_pipeline.mobile_distilled_intent import distilled_intent_scorer
except ImportError:
    from mobile_distilled_intent import distilled_intent_scorer


class MobileVaniShieldEngine:
    """
    Mobile Edge Optimized VaniShield Engine.
    Executes in under 60ms with under 250MB RAM footprint on edge/mobile devices.
    """

    def __init__(
        self,
        onnx_model_path: Optional[str] = None,
        use_int8: bool = True,
        preferred_provider: Optional[str] = None
    ):
        self.sample_rate = 16000
        self.window_size_samples = 24000  # 1.5 seconds at 16kHz
        self.hop_size_samples = 12000     # 0.75 seconds hop
        self.intent_scorer = distilled_intent_scorer

        # Resolve ONNX checkpoint paths
        default_dir = os.path.join(PROJECT_ROOT, "ml_pipeline", "checkpoints", "mobile_edge")
        int8_candidate = os.path.join(default_dir, "vanishield_acoustic_int8.onnx")
        fp32_candidate = os.path.join(default_dir, "vanishield_acoustic_fp32.onnx")

        if onnx_model_path and os.path.exists(onnx_model_path):
            self.model_path = onnx_model_path
        elif use_int8 and os.path.exists(int8_candidate):
            self.model_path = int8_candidate
        elif os.path.exists(fp32_candidate):
            self.model_path = fp32_candidate
        else:
            self.model_path = None

        self.ort_session = None
        self.active_provider = "UNINITIALIZED"
        self._init_onnx_session(preferred_provider)

    def _init_onnx_session(self, preferred_provider: Optional[str] = None):
        """Initializes ONNX Runtime session with Mobile NPU/CPU providers."""
        if not self.model_path:
            logger.warning("[MobileEdge] No ONNX model checkpoint found. Will use acoustic heuristics.")
            return

        try:
            import onnxruntime as ort

            session_options = ort.SessionOptions()
            session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            session_options.intra_op_num_threads = 2
            session_options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL

            available = ort.get_available_providers()
            providers = []

            # Priority order for mobile & edge acceleration
            priority = [
                "NNAPIExecutionProvider",    # Android Neural Networks API (NPU / DSP)
                "QNNExecutionProvider",      # Qualcomm Snapdragon NPU
                "CoreMLExecutionProvider",   # Apple Neural Engine
                "CUDAExecutionProvider",     # Desktop CUDA (if available)
                "CPUExecutionProvider"       # Universal fallback
            ]

            if preferred_provider and preferred_provider in available:
                providers.append(preferred_provider)

            for p in priority:
                if p in available and p not in providers:
                    providers.append(p)

            if not providers:
                providers = ["CPUExecutionProvider"]

            self.ort_session = ort.InferenceSession(
                self.model_path,
                sess_options=session_options,
                providers=providers
            )
            self.active_provider = self.ort_session.get_providers()[0]
            logger.info(f"[MobileEdge] Loaded ONNX model '{os.path.basename(self.model_path)}' via {self.active_provider}")
        except Exception as e:
            logger.error(f"[MobileEdge] Failed to initialize ONNX session: {e}")
            self.ort_session = None
            self.active_provider = f"FAILED: {e}"

    def compute_dsp_metrics(self, audio_chunk: np.ndarray) -> Dict[str, float]:
        """
        Lightweight audio physical signal checks (energy, spectral flatness, centroid).
        Fast NumPy implementation that runs in <2ms on mobile CPU without librosa.
        """
        if len(audio_chunk) == 0:
            return {"rms_energy": 0.0, "spectral_flatness": 0.0, "spectral_centroid": 0.0}

        audio = audio_chunk.astype(np.float32)
        rms = float(np.sqrt(np.mean(audio ** 2)))

        # FFT for spectral features
        n_fft = 512
        if len(audio) < n_fft:
            audio = np.pad(audio, (0, n_fft - len(audio)))

        # Take first 512-sample frame for fast physical check
        frame = audio[:n_fft]
        spectrum = np.abs(np.fft.rfft(frame)) + 1e-10

        # Spectral Centroid
        freqs = np.linspace(0, self.sample_rate / 2, len(spectrum))
        centroid = float(np.sum(freqs * spectrum) / np.sum(spectrum))

        # Spectral Flatness (Geometric Mean / Arithmetic Mean)
        log_spec = np.log(spectrum)
        geom_mean = np.exp(np.mean(log_spec))
        arith_mean = np.mean(spectrum)
        flatness = float(geom_mean / (arith_mean + 1e-10))

        return {
            "rms_energy": round(rms, 4),
            "spectral_centroid": round(centroid, 1),
            "spectral_flatness": round(flatness, 5)
        }

    def infer_acoustic_window(self, window_16k: np.ndarray) -> float:
        """
        Runs acoustic voice clone detection on a 1.5s (24,000 samples) window.
        Returns clone probability percentage (0.0 to 100.0%).
        """
        if self.ort_session is None:
            # Acoustic heuristic fallback if model not loaded
            dsp = self.compute_dsp_metrics(window_16k)
            if dsp["spectral_flatness"] > 0.015 and dsp["rms_energy"] > 0.02:
                return 40.0
            return 5.0

        # Ensure correct shape: (1, sequence_length)
        if len(window_16k) != self.window_size_samples:
            if len(window_16k) < self.window_size_samples:
                window_16k = np.pad(window_16k, (0, self.window_size_samples - len(window_16k)))
            else:
                window_16k = window_16k[:self.window_size_samples]

        input_tensor = np.expand_dims(window_16k.astype(np.float32), axis=0)

        # Run ONNX inference
        outputs = self.ort_session.run(None, {"input_values": input_tensor})
        logits = outputs[0][0]  # Shape: (2,)

        # Softmax: index 0 = real, index 1 = fake/clone
        exp_logits = np.exp(logits - np.max(logits))
        probs = exp_logits / np.sum(exp_logits)
        fake_prob = float(probs[1]) * 100.0

        return round(fake_prob, 1)

    def evaluate_live_call(
        self,
        audio_16k: np.ndarray,
        phone_number: str = "+919876543210",
        transcript: Optional[str] = None,
        language: str = "en"
    ) -> Dict[str, Any]:
        """
        Mobile Edge Real-Time Forensic Evaluation.
        Fuses acoustic neural inference (ONNX INT8) with distilled intent scoring.
        """
        t_start = time.perf_counter()

        # 1. DSP Physical Pre-filtering
        dsp = self.compute_dsp_metrics(audio_16k)
        is_silence = dsp["rms_energy"] < 0.010

        # 2. Sliding Window Acoustic Analysis
        window_scores = []
        if is_silence or len(audio_16k) < 8000:
            w2v_p = 0.0
            is_ai_clone = False
        else:
            step = self.hop_size_samples
            win_size = self.window_size_samples
            n_samples = len(audio_16k)

            # Extract 1.5s windows
            for start in range(0, max(1, n_samples - win_size + 1), step):
                chunk = audio_16k[start:start + win_size]
                if len(chunk) < win_size:
                    chunk = np.pad(chunk, (0, win_size - len(chunk)))
                w_score = self.infer_acoustic_window(chunk)
                window_scores.append(w_score)

            if not window_scores:
                window_scores = [self.infer_acoustic_window(audio_16k[:win_size])]

            w2v_p = float(np.median(window_scores))
            high_conf_windows = sum(1 for s in window_scores if s >= 70.0)
            high_ratio = high_conf_windows / len(window_scores)
            is_ai_clone = (w2v_p >= 50.0 or high_ratio >= 0.35)

        # 3. Distilled Semantic Intent Scoring (<1ms)
        intent_res = self.intent_scorer.score_intent(transcript or "", language=language)
        intent_p = intent_res["intent_risk"]
        is_awareness = intent_res.get("is_awareness_warning", False)

        # 4. Telecom & Scammer Blacklist Verification
        caller_hash = hash_caller_identity(phone_number)
        telecom_info = lookup_carrier(phone_number)
        is_blacklisted_caller, blacklist_entry = is_blacklisted(phone_number)
        telecom_info["is_blacklisted"] = is_blacklisted_caller

        # 5. Risk Fusion & Arbitration
        if is_blacklisted_caller:
            final_risk = 99.0
            threat_type = "BLACKLISTED_SCAMMER"
            action = "IMMEDIATE_CIRCUIT_BREAKER_ENGAGED"
        elif is_awareness:
            final_risk = 0.0
            threat_type = "AUTHENTIC_SAFE_CALL"
            action = "NORMAL_OPERATION"
        elif is_ai_clone and intent_p >= 70.0:
            final_risk = round(max(w2v_p, intent_p), 1)
            threat_type = "AI_CLONE_AND_FINANCIAL_SCAM"
            action = "IMMEDIATE_CIRCUIT_BREAKER_ENGAGED"
        elif is_ai_clone:
            final_risk = round(w2v_p, 1)
            threat_type = "AI_VOICE_CLONE" if final_risk >= 70.0 else "SUSPICIOUS_UNVERIFIED_CALL"
            action = "IMMEDIATE_CIRCUIT_BREAKER_ENGAGED" if final_risk >= 70.0 else "HOLD_VERIFICATION_REQUIRED"
        elif intent_p >= 70.0:
            final_risk = round(intent_p, 1)
            threat_type = "FINANCIAL_SCAM"
            action = "IMMEDIATE_CIRCUIT_BREAKER_ENGAGED"
        else:
            final_risk = round(0.55 * w2v_p + 0.45 * intent_p, 1)
            if final_risk >= 70.0:
                threat_type = "CRITICAL_SCAM"
                action = "IMMEDIATE_CIRCUIT_BREAKER_ENGAGED"
            elif final_risk >= 35.0:
                threat_type = "SUSPICIOUS_CALL"
                action = "HOLD_VERIFICATION_REQUIRED"
            else:
                threat_type = "AUTHENTIC_SAFE_CALL"
                action = "NORMAL_OPERATION"

        total_latency_ms = (time.perf_counter() - t_start) * 1000.0

        # Auto-blacklist if critical scam
        if final_risk >= 70.0 and not is_blacklisted_caller:
            mark_scammer(
                phone_number=phone_number,
                risk_score=final_risk,
                reason=f"Mobile Edge Detection: Risk={final_risk}% ({threat_type})"
            )

        return {
            "status": "success",
            "mode": "mobile_edge_optimized",
            "risk_score": final_risk,
            "threat_type": threat_type,
            "acoustic": {
                "clone_probability": w2v_p,
                "is_ai_clone": is_ai_clone,
                "window_count": len(window_scores),
                "peak_window_score": max(window_scores) if window_scores else 0.0,
                "dsp": dsp
            },
            "intent": {
                "intent_risk": intent_p,
                "threat_category": intent_res.get("threat_type"),
                "urgency": intent_res.get("urgency_level"),
                "flagged_keywords": intent_res.get("flagged_keywords", [])
            },
            "circuit_breaker": {
                "action": action,
                "upi_hold": action == "IMMEDIATE_CIRCUIT_BREAKER_ENGAGED",
                "net_banking_locked": action == "IMMEDIATE_CIRCUIT_BREAKER_ENGAGED"
            },
            "telecom": telecom_info,
            "caller_hash": caller_hash,
            "telemetry": {
                "inference_provider": self.active_provider,
                "latency_ms": round(total_latency_ms, 2),
                "model_type": "Wav2Vec2_INT8_ONNX + Distilled_Intent"
            }
        }


# Singleton Mobile Engine instance
mobile_engine = MobileVaniShieldEngine()
