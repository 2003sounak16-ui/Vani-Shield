import os
import sys
import unittest
import numpy as np
import pandas as pd
import time
from unittest.mock import MagicMock

# Ensure repo root is on sys.path
_curr_dir = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(_curr_dir) if os.path.basename(_curr_dir) == "tests" else _curr_dir
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
_ml_dir = os.path.join(PROJECT_ROOT, "ml_pipeline")
if _ml_dir not in sys.path:
    sys.path.insert(0, _ml_dir)

from ephemeral_buffer import EphemeralRAMBuffer, hash_caller_identity
try:
    from build_robust_manifest import make_split
except ImportError:
    from ml_pipeline.build_robust_manifest import make_split

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
        """Verify voice-decision-head-f32.npz exists and has valid shapes for W, b, mu, sd, norm_w, norm_b."""
        head_path = os.environ.get("VANISHIELD_HEAD_NPZ", os.path.join(PROJECT_ROOT, "ml_pipeline", "checkpoints", "voice-decision-head-f32.npz"))
        if not os.path.exists(head_path):
            head_path = os.path.join(PROJECT_ROOT, "voice-decision-head-f32.npz")
        self.assertTrue(os.path.exists(head_path), "voice-decision-head-f32.npz must exist")
        data = np.load(head_path)
        self.assertIn("W", data)
        self.assertIn("b", data)
        self.assertIn("mu", data)
        self.assertIn("sd", data)
        self.assertIn("norm_w", data)
        self.assertIn("norm_b", data)
        self.assertEqual(data["W"].shape, (2, 3840))
        self.assertEqual(data["b"].shape, (2,))
        self.assertEqual(data["norm_w"].shape, (3840,))
        self.assertEqual(data["norm_b"].shape, (3840,))

    def test_transcription_failure_status(self):
        """Verify that when transcription fails, evaluate_live_call produces incomplete_analysis rather than safe."""
        from vanishield_engine import VaniShieldCore

        # Instantiate engine object without triggering heavyweight CUDA/HF model loads
        engine = VaniShieldCore.__new__(VaniShieldCore)
        engine.device = "cpu"
        engine.asr_name = "Faster-Whisper Mock"
        engine.blacklist = {}
        engine.jev_head_trained = False
        engine.w2v_model_path = None
        engine.demo_scorer = None

        # Simulate ASR failure (e.g. timeout or exception during speech recognition)
        engine.transcribe_and_spot_keywords = MagicMock(
            return_value=("", [], "ASR model timeout / inference crash", "en")
        )
        engine.analyze_telecom = MagicMock(
            return_value={"is_blacklisted": False, "is_voip_spoofed": False, "location": "Bengaluru"}
        )
        engine.extract_prosodic_dsp = MagicMock(
            return_value={"rms_energy": 0.05, "zcr": 0.02, "spectral_flatness": 0.01}
        )
        engine.run_acoustic_ear = MagicMock(return_value=10.0)

        # 2 seconds of 16kHz sine wave simulating audible speech
        sr = 16000
        dummy_audio = (np.sin(2 * np.pi * 440 * np.linspace(0, 2, 2 * sr)) * 0.2).astype(np.float32)

        result = engine.evaluate_live_call(dummy_audio, phone_number="+91-9876543210")

        self.assertEqual(result["status"], "incomplete_analysis")
        self.assertEqual(result["threat_type"], "UNVERIFIED_TRANSCRIPTION_FAILED")
        self.assertNotEqual(result["threat_type"], "AUTHENTIC_SAFE_CALL")
        self.assertEqual(result["circuit_breaker"]["action"], "HOLD_VERIFICATION_REQUIRED")
        self.assertIn("transcription_error", result)
        self.assertEqual(result["transcription_error"], "ASR model timeout / inference crash")

    def test_caller_tracker_pseudonymization(self):
        """Verify caller_tracker uses HMAC-SHA256, masks phone numbers, and keys records by hash."""
        from backend.engine.caller_tracker import hash_caller_identity, mask_phone, mark_scammer, is_blacklisted, load_blacklist
        import tempfile

        test_phone = "+919988776655"
        h = hash_caller_identity(test_phone)
        self.assertEqual(len(h), 16, "HMAC-SHA256 hash must be 16 hex chars")
        self.assertEqual(mask_phone(test_phone), "+919****6655")

        # Isolated blacklist file
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tf:
            temp_bl = tf.name

        try:
            import backend.engine.caller_tracker as ct
            old_bl = ct.BLACKLIST_PATH
            ct.BLACKLIST_PATH = temp_bl

            rec = ct.mark_scammer(test_phone, 95.0, "Critical Bank Fraud Scam Demand")
            self.assertEqual(rec["caller_hash"], h)
            self.assertEqual(rec["phone_masked"], "+919****6655")
            self.assertNotIn(test_phone, rec, "Raw phone number must NOT be in persisted record")

            # Verify saved JSON is keyed by hash, not raw phone
            saved_bl = ct.load_blacklist()
            self.assertIn(h, saved_bl, "Blacklist must be keyed by caller_hash")
            self.assertNotIn(test_phone, saved_bl, "Raw phone number must NOT be a top-level key")

            # Verify is_blacklisted resolves via HMAC hash
            is_bl, bl_data = ct.is_blacklisted(test_phone)
            self.assertTrue(is_bl)
            self.assertEqual(bl_data["caller_hash"], h)

            # Verify backward compatibility with legacy-hashed records.
            # Use a SEPARATE temp file with ONLY the legacy record so the canonical
            # hash key (h) is absent and is_blacklisted must reach the legacy fallback
            # path to succeed — without this isolation the test always hits the
            # canonical record first and the legacy path is never exercised.
            h_legacy = ct.hash_caller_identity_legacy(test_phone)
            import json, tempfile as _tmpmod
            with _tmpmod.NamedTemporaryFile(suffix=".json", delete=False) as tf2:
                temp_bl_legacy = tf2.name
            try:
                legacy_only = {h_legacy: {"caller_hash": h_legacy, "phone_masked": "+919****6655", "risk_score": 90.0}}
                with open(temp_bl_legacy, "w", encoding="utf-8") as f:
                    json.dump(legacy_only, f)
                ct.BLACKLIST_PATH = temp_bl_legacy
                is_bl_legacy, bl_data_legacy = ct.is_blacklisted(test_phone)
                self.assertTrue(is_bl_legacy, "Legacy hash backward-compat lookup must succeed")
                self.assertEqual(bl_data_legacy["caller_hash"], h_legacy,
                                 "Returned record must be the legacy-keyed entry")

                # Verify that even when VANISHIELD_HMAC_SECRET is set and differs from VANISHIELD_SALT,
                # the legacy record saved with VANISHIELD_SALT is still correctly matched (Issue #1)
                old_hmac_secret = os.environ.get("VANISHIELD_HMAC_SECRET")
                try:
                    os.environ["VANISHIELD_HMAC_SECRET"] = "new-different-hmac-secret-999"
                    is_bl_migrated, bl_data_migrated = ct.is_blacklisted(test_phone)
                    self.assertTrue(is_bl_migrated, "Must match legacy record even when VANISHIELD_HMAC_SECRET differs from VANISHIELD_SALT")
                    self.assertEqual(bl_data_migrated["caller_hash"], h_legacy)
                finally:
                    if old_hmac_secret is not None:
                        os.environ["VANISHIELD_HMAC_SECRET"] = old_hmac_secret
                    else:
                        os.environ.pop("VANISHIELD_HMAC_SECRET", None)
            finally:
                ct.BLACKLIST_PATH = temp_bl  # restore to canonical temp file
                if os.path.exists(temp_bl_legacy):
                    os.remove(temp_bl_legacy)

        finally:
            ct.BLACKLIST_PATH = old_bl
            if os.path.exists(temp_bl):
                os.remove(temp_bl)

    def test_banking_unlock_pin_validation(self):
        """Verify banking circuit breaker requires valid PIN and rejects missing/invalid PIN."""
        import asyncio
        from backend.api.banking_hold import unlock_banking, UnlockRequest, state
        from fastapi import HTTPException
        from pydantic import ValidationError

        # Set locked state
        state.locked = True

        # 1. Missing pin must fail Pydantic validation (cannot pass empty {})
        with self.assertRaises(ValidationError):
            UnlockRequest()

        # 2. Invalid PIN must raise HTTPException(403)
        loop = asyncio.new_event_loop()
        try:
            with self.assertRaises(HTTPException) as ctx:
                loop.run_until_complete(unlock_banking(UnlockRequest(pin="0000")))
            self.assertEqual(ctx.exception.status_code, 403)
            self.assertTrue(state.locked)

            # 3. Correct PIN releases hold
            res = loop.run_until_complete(unlock_banking(UnlockRequest(pin="1234")))
            self.assertEqual(res["status"], "ACTIVE")
            self.assertFalse(state.locked)
        finally:
            loop.close()


    def test_tts_authorization_and_transfer_lock(self):
        """Verify TTS routes require authorization when configured and transfer is blocked when locked."""
        import asyncio
        from fastapi import HTTPException
        from backend.api.evaluate import check_tts_access, check_tts_budget_and_length
        from backend.api.banking_hold import simulate_transfer, TransferRequest, state

        class MockRequest:
            def __init__(self, headers=None, host="127.0.0.1"):
                self.headers = headers or {}
                self.client = type("Client", (), {"host": host})()

        # 1. Test TTS authorization enforcement
        old_token = os.environ.get("TTS_ACCESS_TOKEN")
        try:
            os.environ["TTS_ACCESS_TOKEN"] = "secure-secret-token"
            req_no_auth = MockRequest()
            with self.assertRaises(HTTPException) as ctx:
                check_tts_access(req_no_auth)
            self.assertEqual(ctx.exception.status_code, 401)

            req_wrong = MockRequest({"X-TTS-Token": "wrong-token"})
            with self.assertRaises(HTTPException) as ctx:
                check_tts_access(req_wrong)
            self.assertEqual(ctx.exception.status_code, 401)

            req_valid = MockRequest({"X-TTS-Token": "secure-secret-token"})
            check_tts_access(req_valid)

            check_tts_access(req_no_auth, body_token="secure-secret-token")

            with self.assertRaises(HTTPException) as ctx:
                check_tts_budget_and_length("a" * 501, req_valid)
            self.assertEqual(ctx.exception.status_code, 400)
        finally:
            if old_token is not None:
                os.environ["TTS_ACCESS_TOKEN"] = old_token
            else:
                os.environ.pop("TTS_ACCESS_TOKEN", None)

        # 2. Test transfer lock enforcement
        state.locked = True
        loop = asyncio.new_event_loop()
        try:
            with self.assertRaises(HTTPException) as ctx:
                loop.run_until_complete(simulate_transfer(TransferRequest(payee="Electricity", amount=1500.0)))
            self.assertEqual(ctx.exception.status_code, 423)

            state.locked = False
            tx_res = loop.run_until_complete(simulate_transfer(TransferRequest(payee="Electricity", amount=1500.0)))
            self.assertTrue(tx_res["success"])
        finally:
            loop.close()

        # 3. Test real audio evaluation trips banking_state.locked (Issue #5)
        from backend.api.evaluate import _sync_evaluate_audio
        import backend.api.evaluate as ev_mod
        from unittest.mock import MagicMock
        state.locked = False
        saved_engine = ev_mod.core_engine
        try:
            mock_eng = MagicMock()
            mock_eng.evaluate_live_call = MagicMock(return_value={
                "circuit_breaker": {"engaged": True},
                "unified_risk_score": 98.0,
                "threat_type": "AI_CLONE_AND_FINANCIAL_SCAM"
            })
            ev_mod.core_engine = mock_eng
            saved_loader = ev_mod.load_audio_as_16k_np
            ev_mod.load_audio_as_16k_np = MagicMock(return_value=np.zeros(16000, dtype=np.float32))
            try:
                res = _sync_evaluate_audio(b"dummy_wav", "+91-1409988776", None)
            finally:
                ev_mod.load_audio_as_16k_np = saved_loader
            self.assertTrue(state.locked, "Real audio evaluation tripping circuit breaker must set banking state locked to True")
            self.assertEqual(state.risk_score, 98.0)
        finally:
            ev_mod.core_engine = saved_engine
            state.locked = False

if __name__ == "__main__":
    unittest.main()
