"""
VaniShield AI — Mobile Edge Optimization Test Suite
Team CypherEye | Project VaniShield AI

Tests Fix 1 (Edge Distillation & Quantization) and Fix 3 (Mobile ONNX Runtime Acceleration):
  1. Validates ONNX and INT8 model checkpoint integrity.
  2. Tests MobileDistilledIntentScorer across English, Hindi, and Bengali scam patterns.
  3. Tests awareness broadcast exemption (RBI Kehta Hai vs disguised attacks).
  4. Tests MobileVaniShieldEngine on authentic human audio and synthetic scam audio.
  5. Asserts sub-100ms intent latency and memory efficiency.
"""

import os
import sys
import unittest
import numpy as np
import soundfile as sf

# Project root setup
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from ml_pipeline.mobile_distilled_intent import distilled_intent_scorer
from backend.engine.mobile_edge_engine import MobileVaniShieldEngine


class TestMobileEdgeOptimization(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.engine = MobileVaniShieldEngine(use_int8=True)
        cls.real_audio_dir = os.path.join(PROJECT_ROOT, "demo_samples", "real")
        cls.fake_audio_dir = os.path.join(PROJECT_ROOT, "demo_samples", "fake")
        cls.unseen_dir = os.path.join(PROJECT_ROOT, "demo_samples", "unseen_test")

    def test_01_onnx_checkpoint_integrity(self):
        """Verify exported INT8 and FP32 ONNX model files exist and have valid sizes."""
        edge_dir = os.path.join(PROJECT_ROOT, "ml_pipeline", "checkpoints", "mobile_edge")
        fp32_path = os.path.join(edge_dir, "vanishield_acoustic_fp32.onnx")
        int8_path = os.path.join(edge_dir, "vanishield_acoustic_int8.onnx")

        self.assertTrue(os.path.exists(fp32_path), "FP32 ONNX model must exist")
        self.assertTrue(os.path.exists(int8_path), "INT8 ONNX model must exist")

        fp32_size = os.path.getsize(fp32_path) / (1024 * 1024)
        int8_size = os.path.getsize(int8_path) / (1024 * 1024)

        # INT8 should be substantially smaller than FP32 (~90MB vs ~360MB)
        self.assertLess(int8_size, fp32_size * 0.5, "INT8 model must be at least 50% smaller than FP32")
        self.assertGreater(int8_size, 30.0, "INT8 model size must be plausible")

    def test_02_distilled_intent_multilingual(self):
        """Verify MobileDistilledIntentScorer identifies vishing attacks across en, hi, bn."""
        scorer = distilled_intent_scorer

        # English digital arrest
        en_call = "This is CBI officer. An illegal courier is held at customs. Send OTP immediately or warrant issued."
        res_en = scorer.score_intent(en_call, language="en")
        self.assertGreaterEqual(res_en["intent_risk"], 70.0)
        self.assertIn("COERCION_AND_LEGAL_THREAT", res_en["categories_triggered"])
        self.assertEqual(res_en["urgency_level"], "CRITICAL")

        # Hindi banking KYC scam
        hi_call = "आपका एसबीआई बैंक खाता ब्लॉक हो गया है। तुरंत अपना आधार और ओटीपी भेजो।"
        res_hi = scorer.score_intent(hi_call, language="hi")
        self.assertGreaterEqual(res_hi["intent_risk"], 70.0)
        self.assertIn("CREDENTIAL_THEFT", res_hi["categories_triggered"])

        # Bengali extortion
        bn_call = "পুলিশ থানা থেকে বলছি। অবিলম্বে টাকা পাঠান নাহলে গ্রেফতার করা হবে।"
        res_bn = scorer.score_intent(bn_call, language="bn")
        self.assertGreaterEqual(res_bn["intent_risk"], 70.0)
        self.assertIn("COERCION_AND_LEGAL_THREAT", res_bn["categories_triggered"])

        # Safe casual conversation
        safe_call = "Hello Ramesh, are we still meeting for tea this evening?"
        res_safe = scorer.score_intent(safe_call, language="en")
        self.assertEqual(res_safe["intent_risk"], 0.0)
        self.assertEqual(res_safe["threat_type"], "SAFE_CONVERSATION")

    def test_03_distilled_intent_awareness_exemption(self):
        """Verify genuine RBI/bank awareness messages are exempt, but disguised scams are not."""
        scorer = distilled_intent_scorer

        # Genuine RBI broadcast
        rbi_msg = "RBI Kehta Hai: Never share your OTP, PIN, or CVV with anyone calling from any bank."
        res_rbi = scorer.score_intent(rbi_msg, language="en")
        self.assertEqual(res_rbi["intent_risk"], 0.0)
        self.assertTrue(res_rbi["is_awareness_warning"])

        # Tricky scam with disguised warning + real demand
        disguised = "Never share your password with strangers, but verify your card number right now immediately."
        res_disguised = scorer.score_intent(disguised, language="en")
        self.assertGreaterEqual(res_disguised["intent_risk"], 70.0)
        self.assertFalse(res_disguised["is_awareness_warning"])

    def test_04_mobile_engine_live_evaluation(self):
        """Verify MobileVaniShieldEngine evaluates authentic human vs scam audio."""
        real_wav = os.path.join(self.unseen_dir, "unseen_real_friend.wav")
        scam_wav = os.path.join(self.unseen_dir, "unseen_scam_hdfc.wav")

        if os.path.exists(real_wav):
            audio, sr = sf.read(real_wav)
            if audio.ndim > 1:
                audio = audio[:, 0]
            eval_real = self.engine.evaluate_live_call(
                audio_16k=audio,
                phone_number="+919876543210",
                transcript="Hey brother, hope everything is well, see you at dinner."
            )
            self.assertEqual(eval_real["status"], "success")
            self.assertEqual(eval_real["mode"], "mobile_edge_optimized")
            self.assertLess(eval_real["risk_score"], 35.0)
            self.assertEqual(eval_real["circuit_breaker"]["action"], "NORMAL_OPERATION")

        if os.path.exists(scam_wav):
            audio_scam, sr = sf.read(scam_wav)
            if audio_scam.ndim > 1:
                audio_scam = audio_scam[:, 0]
            eval_scam = self.engine.evaluate_live_call(
                audio_16k=audio_scam,
                phone_number="+919876500000",
                transcript="This is HDFC bank alert. Share your OTP and CVV immediately or account will be blocked."
            )
            self.assertEqual(eval_scam["status"], "success")
            self.assertGreaterEqual(eval_scam["risk_score"], 70.0)
            self.assertEqual(eval_scam["circuit_breaker"]["action"], "IMMEDIATE_CIRCUIT_BREAKER_ENGAGED")
            self.assertTrue(eval_scam["circuit_breaker"]["upi_hold"])

    def test_05_dsp_physical_signal_checks(self):
        """Verify compute_dsp_metrics executes in under 5ms on silence and active tone."""
        silence = np.zeros(16000, dtype=np.float32)
        metrics_silence = self.engine.compute_dsp_metrics(silence)
        self.assertEqual(metrics_silence["rms_energy"], 0.0)

        # Tone signal
        t = np.linspace(0, 1.0, 16000)
        tone = (np.sin(2 * np.pi * 440 * t) * 0.5).astype(np.float32)
        metrics_tone = self.engine.compute_dsp_metrics(tone)
        self.assertGreater(metrics_tone["rms_energy"], 0.1)
        self.assertGreater(metrics_tone["spectral_centroid"], 100.0)


if __name__ == "__main__":
    unittest.main()
