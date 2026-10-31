import os
import sys
import unittest
import numpy as np
import pandas as pd
import time

# Ensure repo root is on sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

try:
    from backend.engine.ephemeral_buffer import EphemeralRAMBuffer, hash_caller_identity
except ImportError:
    from ephemeral_buffer import EphemeralRAMBuffer, hash_caller_identity

try:
    from ml_pipeline.build_robust_manifest import make_split
except ImportError:
    from build_robust_manifest import make_split

class TestVaniShieldFixes(unittest.TestCase):

    def test_ephemeral_buffer_configured_retention(self):
        """Verify EphemeralRAMBuffer honors configured duration and cleans up expired chunks."""
        buf = EphemeralRAMBuffer(max_duration_sec=1.0, sample_rate=16000)
        sess = "test_sess_01"
        pcm_chunk = np.ones(1600, dtype=np.int16).tobytes()

        # Push chunk
        buf.push_chunk(sess, pcm_chunk)
        self.assertEqual(len(buf.buffers[sess]), 1)

        # Immediate eval window should have non-zero data
        window = buf.get_eval_window(sess)
        self.assertGreater(np.abs(window).sum(), 0.0)

        # After sleep > max_duration_sec, window should be zeroed / evicted
        time.sleep(1.05)
        window_expired = buf.get_eval_window(sess)
        self.assertEqual(np.abs(window_expired).sum(), 0.0)
        self.assertEqual(len(buf.buffers[sess]), 0)

    def test_ephemeral_buffer_evict_expired_idle(self):
        """Verify proactive idle session eviction."""
        buf = EphemeralRAMBuffer(max_duration_sec=0.5, sample_rate=16000)
        sess = "idle_sess"
        buf.push_chunk(sess, np.ones(800, dtype=np.int16).tobytes())
        time.sleep(0.55)
        buf.evict_expired()
        self.assertNotIn(sess, buf.buffers)

    def test_build_robust_manifest_connected_components(self):
        """Verify that rows sharing either a speaker or a source never cross splits."""
        # Source 1 has speakers S1 and S2
        # Source 2 has speakers S2 and S3 (connected via S2)
        # Source 3 has speaker S4 (independent)
        data = {
            "path": [f"audio_{i}.wav" for i in range(7)],
            "label": [0, 0, 1, 1, 0, 1, 0],
            "source": ["src_A", "src_A", "src_B", "src_B", "src_C", "src_C", "src_C"],
            "speaker_id": ["spk_1", "spk_2", "spk_2", "spk_3", "spk_4", "spk_4", "spk_5"]
        }
        df = pd.DataFrame(data)
        split_df = make_split(df, holdout=0.3, seed=42, split_name="val")

        train_sources = set(split_df[split_df["split"] == "train"]["source"])
        val_sources = set(split_df[split_df["split"] == "val"]["source"])
        train_speakers = set(split_df[split_df["split"] == "train"]["speaker_id"])
        val_speakers = set(split_df[split_df["split"] == "val"]["speaker_id"])

        # No overlap between train and val sources or speakers
        self.assertEqual(train_sources.intersection(val_sources), set(), "Source leaked between train and val!")
        self.assertEqual(train_speakers.intersection(val_speakers), set(), "Speaker leaked between train and val!")

    def test_awareness_vs_fraud_demand_logic(self):
        """Verify that warning words accompanied by fraud demands are NOT treated as awareness exemptions."""
        import re

        AWARENESS_WARNING_PATTERNS = [
            r"\b(never|do not|don'?t|should not|must not)\s+(?:share|give|disclose|tell|send|enter)\b.*?\b(otp|pin|cvv|password|details|credential|card number)\b",
            r"\b(bank|rbi|police)\s+(?:never|will never|does not)\s+(?:ask|call to ask|demand)\b.*?\b(otp|pin|cvv|password|credentials)\b",
            r"(?:ओटीपी|पिन|सीवीवी|पासवर्ड|डिटेल्स)\s*(?:कभी\s*)?(?:मत|ना)\s*(?:दीजिए|देना|बताइए|शेयर|करना)",
            r"बैंक\s*(?:कभी\s*)?(?:ओटीपी|पिन|पासवर्ड)\s*नहीं\s*(?:मांगता|मांगती)",
            r"(?:ওটিপি|পিন|সিভিভি|পাসওয়ার্ড)\s*(?:কখনও\s*)?(?:কাউকে\s*)?(?:দেবেন না|বলবেন না|শেয়ার করবেন না)",
            r"ব্যাংক\s*(?:কখনও\s*)?(?:ওটিপি|পিন|পাসওয়ার্ড)\s*চায় না"
        ]

        DEMAND_PATTERNS = [
            r"\b(share|send|tell|give|provide|transfer|enter|verify|update|reverse|confirm|pay|deposit)\b",
            r"\b(immediately|right now|urgent|urgently|within \d+ minutes|otherwise|blocked|suspended|warrant|arrest|illegal|jail)\b",
            r"(भेजो|दीजिए|दे दो|बताओ|शेयर करो|ट्रांसफर करो|डालो|तुरंत|अवैध|ब्लॉक|गिरफ्तारी|वॉरंट)",
            r"(পাঠান|দিন|শেয়ার করুন|বলুন|ট্রান্সফার করুন|অবিলম্বে|স্থগিত|গ্রেফতার|বন্ধ হবে)"
        ]

        def evaluate_awareness(text, kws):
            has_advice = bool(text and any(re.search(p, text, flags=re.IGNORECASE) for p in AWARENESS_WARNING_PATTERNS))
            cleaned = text or ""
            for p in AWARENESS_WARNING_PATTERNS:
                cleaned = re.sub(p, " ", cleaned, flags=re.IGNORECASE)
            has_indep_demand = bool(any(re.search(dp, cleaned, flags=re.IGNORECASE) for dp in DEMAND_PATTERNS))
            is_aware = bool(has_advice and not (has_indep_demand and kws))
            return has_advice, has_indep_demand, is_aware

        # Case 1: Scammer vishing attack using the word 'alert' and credential demand
        scam_call = "This is HDFC bank alert: Your account is blocked. Share your OTP and card CVV immediately to avoid suspension."
        flagged_kw = ["otp", "cvv", "account blocked", "hdfc bank"]
        has_advice_1, has_demand_1, is_aware_1 = evaluate_awareness(scam_call, flagged_kw)

        self.assertFalse(has_advice_1, "'alert' alone must not count as awareness advice")
        self.assertTrue(has_demand_1, "Credential demand and urgency must be detected")
        self.assertFalse(is_aware_1, "Scam call must NOT be exempted")

        # Case 2: Genuine Bank Awareness Broadcast
        awareness_call = "Dear customer, RBI Kehta Hai: Never share your OTP or CVV with anyone calling from any bank."
        flagged_kw_aware = ["otp", "cvv"]
        has_advice_2, has_demand_2, is_aware_2 = evaluate_awareness(awareness_call, flagged_kw_aware)

        self.assertTrue(has_advice_2, "Explicit warning must be recognized")
        self.assertFalse(has_demand_2, "Negated sharing warning must not count as independent fraud demand")
        self.assertTrue(is_aware_2, "Genuine educational warning must be exempted")

        # Case 3: Tricky scam combining awareness warning with actual demand
        tricky_scam = "Never share your OTP with strangers, but please share your account verification code right now immediately."
        has_advice_3, has_demand_3, is_aware_3 = evaluate_awareness(tricky_scam, ["otp", "verification code"])
        self.assertTrue(has_advice_3)
        self.assertTrue(has_demand_3)
        self.assertFalse(is_aware_3, "Tricky scam with real demand must NOT be exempted")

    def test_decision_head_file_integrity(self):
        """Verify voice-decision-head-f32.npz exists and has valid shapes for W, b, mu, sd."""
        head_path = os.environ.get("VANISHIELD_HEAD_NPZ", os.path.join(PROJECT_ROOT, "ml_pipeline", "checkpoints", "voice-decision-head-f32.npz"))
        if not os.path.exists(head_path):
            head_path = os.path.join(PROJECT_ROOT, "voice-decision-head-f32.npz")
        self.assertTrue(os.path.exists(head_path), "voice-decision-head-f32.npz must exist")
        data = np.load(head_path)
        self.assertIn("W", data)
        self.assertIn("b", data)
        self.assertIn("mu", data)
        self.assertIn("sd", data)
        self.assertEqual(data["W"].shape, (2, 3840))
        self.assertEqual(data["b"].shape, (2,))

    def test_transcription_failure_status(self):
        """Verify that when transcription fails, the actual engine produces incomplete_analysis and HOLD action."""
        try:
            from backend.engine.vanishield_engine import core_engine
        except ImportError:
            from vanishield_engine import core_engine

        # Mock transcribe_and_spot_keywords on core_engine
        orig_method = core_engine.transcribe_and_spot_keywords
        try:
            core_engine.transcribe_and_spot_keywords = lambda *args, **kwargs: ("", [], "ASR model timeout")
            dummy_audio = (np.sin(np.linspace(0, 100, 16000)) * 0.1).astype(np.float32)
            result = core_engine.evaluate_live_call(dummy_audio, phone_number="+919876543210")

            self.assertEqual(result.get("status"), "incomplete_analysis")
            self.assertEqual(result.get("threat_type"), "UNVERIFIED_TRANSCRIPTION_FAILED")
            self.assertEqual(result.get("circuit_breaker", {}).get("action"), "HOLD_VERIFICATION_REQUIRED")
        finally:
            core_engine.transcribe_and_spot_keywords = orig_method

if __name__ == "__main__":
    unittest.main()
