"""
VaniShield AI — Tests for the social-engineering stage classifier and fusion.

Runnable either under pytest or directly (`python test_scam_stages.py`), since
the repo's other test_*.py files are plain scripts and pytest is not a declared
dependency.

The cases here are deliberately weighted towards the two failure modes that
matter in a fraud detector, because they are not symmetric:

  * A false negative lets a scam through — tested via the six shipped call
    fixtures and the full cloned-relative script.
  * A false positive teaches the user to ignore the warning, which disarms the
    product permanently. Tested via the bank's own awareness recording, a
    benign mention of an OTP, and a genuine bank call that identifies itself
    but asks for nothing.
"""

import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import scam_stages as S
from scam_stages import (
    AUTHORITY,
    CREDENTIAL,
    REMOTE,
    SECRECY,
    URGENCY,
    band_for,
    band_label,
    classify_stages,
    event_text,
    explain,
    fuse_risk,
    normalize_lang,
)

# The six fixtures shipped in test_multilingual_calls/, with the transcripts
# app.py's PRESET_CALLS pairs with them.
PRESET_TRANSCRIPTS = {
    "en_safe": "Hi team, let's reschedule our design review meeting to Thursday afternoon at 3 PM. Let me know if that works.",
    "en_scam": "Urgent security notice from State Bank of India fraud team. Your account has an illegal debit of 65,000 rupees. Share your 6-digit OTP and CVV right now to reverse it.",
    "bn_safe": "নমস্কার কাকিমা, আমি শুভঙ্কর বলছি। কাল বিকেলে আপনাদের বাড়ি আসব, কেমন আছেন সবাই?",
    "bn_scam": "জরুরি ব্যাংক সতর্কতা: আপনার স্টেট ব্যাংক একাউন্ট স্থগিত করা হয়েছে। অবিলম্বে ৬ সংখ্যার ওটিপি ও ইউপিআই পিন প্রদান করুন অন্যথায় অ্যাকাউন্ট স্থায়ীভাবে বন্ধ হবে।",
    "hi_safe": "नमस्ते भाई, कल कॉलेज की क्लास सुबह नौ बजे शुरू होगी। असाइनमेंट पूरा हो गया क्या?",
    "hi_scam": "नमस्कार, एचडीएफसी बैंक हेड ऑफिस। आपके पैन कार्ड में विसंगति पाई गई है। तुरंत अपना एटीएम पिन, सीवीवी और ओटीपी शेयर करके पैन री-वेरीफाई करें।",
}

CLONED_RELATIVE_SCRIPT = (
    "দিদা, আমি রাহুল বলছি, তোমার নাতি। "
    "I had an accident, the hospital needs money right now. "
    "Please send ₹50,000 to this UPI ID, quickly. "
    "মা-কে কিছু বলো না, প্লিজ"
)


def _risk(text, acoustic_prob=0.0):
    analysis = classify_stages(text)
    fusion = fuse_risk(
        analysis.stages,
        acoustic_prob=acoustic_prob,
        payment_attempt=analysis.payment_attempt,
        credential_high_sensitivity=analysis.high_sensitivity_credential,
    )
    return analysis, fusion


# ---------------------------------------------------------------------------
# False negatives: the scams must be caught
# ---------------------------------------------------------------------------


def test_scam_presets_are_red_on_transcript_alone():
    """All three scam fixtures must convict without any acoustic evidence.

    These are human-fraudster phishing scripts. If they need the deepfake
    pillar to reach red, then a scammer who simply uses their own voice walks
    straight through, which is the most common attack of all.
    """
    for preset in ("en_scam", "bn_scam", "hi_scam"):
        analysis, fusion = _risk(PRESET_TRANSCRIPTS[preset])
        assert CREDENTIAL in analysis.stages, f"{preset}: no credential demand found"
        assert URGENCY in analysis.stages, f"{preset}: no urgency found"
        assert fusion["band"] == S.BAND_RED, f"{preset}: band was {fusion['band']} at {fusion['risk_score']}"


def test_cloned_relative_script_is_red_and_finds_four_stages():
    analysis, fusion = _risk(CLONED_RELATIVE_SCRIPT)
    assert analysis.stages == [AUTHORITY, URGENCY, CREDENTIAL, SECRECY]
    assert analysis.payment_attempt is True
    assert analysis.amount_inr == 50000.0
    assert fusion["band"] == S.BAND_RED


def test_remote_access_demand_is_at_least_amber():
    analysis, fusion = _risk("Please install AnyDesk so I can help you fix the refund")
    assert REMOTE in analysis.stages
    assert fusion["band"] in (S.BAND_AMBER, S.BAND_RED)


def test_romanised_indic_from_the_asr_is_caught():
    """Faster-Whisper returns Indic speech in Latin script as often as not.

    The shipped Hindi fixture transcribes live as
    "Namaskar, HDFC Bank Head Office. Aapke pan kard me visangati..." — so a
    classifier that only knows Devanagari scores the production path far
    lower than the fixture-transcript path, which is exactly backwards.
    """
    romanised_hi = (
        "Namaskar, HDFC Bank Head Office. Aapke pan kard me visangati payi gayi hai. "
        "Turant apna ATM pin, CVV aur OTP share karke pan re-verify karen."
    )
    analysis, fusion = _risk(romanised_hi)
    assert AUTHORITY in analysis.stages, "romanised institution claim missed"
    assert URGENCY in analysis.stages, "romanised 'turant' missed"
    assert CREDENTIAL in analysis.stages, "romanised credential demand missed"
    assert fusion["band"] == S.BAND_RED, f"romanised scam only reached {fusion['risk_score']}"

    romanised_bn = (
        "Joruri bank sotorkota. Apnar account sthogit kora hoyeche. "
        "Obilombe OTP ebong UPI pin pathan."
    )
    analysis_bn, fusion_bn = _risk(romanised_bn)
    assert URGENCY in analysis_bn.stages, "romanised Bengali urgency missed"
    assert CREDENTIAL in analysis_bn.stages, "romanised Bengali credential demand missed"
    assert fusion_bn["band"] == S.BAND_RED


def test_romanised_secrecy_is_caught():
    for text in (
        "Kisi ko kuch mat batana, hamare bich hi rahe",
        "Maa-ke kichu bolo na, gopon rakho",
    ):
        assert SECRECY in classify_stages(text).stages, f"romanised secrecy missed: {text}"


def test_romanised_patterns_do_not_fire_on_plain_english():
    """The romanised rows must not catch ordinary English conversation."""
    for text in (
        "Hi team, let's reschedule our design review meeting to Thursday afternoon at 3 PM.",
        "I'll send you the report after the standup, no rush at all.",
        "Can you confirm whether the parcel arrived at the office today?",
    ):
        analysis, fusion = _risk(text)
        assert analysis.stages == [], f"romanised rows misfired on English: {text} -> {analysis.stages}"
        assert fusion["band"] == S.BAND_GREEN


def test_secrecy_detected_in_all_three_languages():
    cases = [
        "Don't tell anyone about this, keep it a secret",
        "किसी को मत बताना, यह गुप्त रखना",
        "কাউকে বলো না, গোপন রাখো",
    ]
    for text in cases:
        assert SECRECY in classify_stages(text).stages, f"secrecy missed in: {text}"


# ---------------------------------------------------------------------------
# False positives: the safe calls must stay quiet
# ---------------------------------------------------------------------------


def test_safe_presets_are_green():
    for preset in ("en_safe", "bn_safe", "hi_safe"):
        _, fusion = _risk(PRESET_TRANSCRIPTS[preset])
        assert fusion["band"] == S.BAND_GREEN, f"{preset}: band was {fusion['band']} at {fusion['risk_score']}"


def test_bank_awareness_recording_scores_zero_stages():
    """The bank's own warning contains every high-signal credential word.

    This is the single most dangerous false positive in the product: the
    phrases a fraud-awareness message must use are exactly the phrases a
    fraudster uses, so an unguarded keyword matcher flags the warning itself.
    """
    for text in (
        "Remember, never share your OTP or PIN with anyone",
        "Bank never asks for your OTP or PIN. Beware of fraud calls.",
        "ओटीपी किसी को मत बताइए",
        "ওটিপি কাউকে বলবেন না",
    ):
        analysis = classify_stages(text)
        assert analysis.is_awareness_advice is True, f"not recognised as advice: {text}"
        assert CREDENTIAL not in analysis.stages, f"advice fired CREDENTIAL: {text}"
        assert band_for(fuse_risk(analysis.stages)["risk_score"]) == S.BAND_GREEN


def test_mentioning_an_otp_without_a_demand_is_not_credential():
    for text in (
        "I never received the OTP for my transaction, can you check?",
        "The OTP expired before I could read it",
    ):
        analysis = classify_stages(text)
        assert CREDENTIAL not in analysis.stages, f"bare mention fired CREDENTIAL: {text}"


def test_genuine_bank_call_identifying_itself_stays_green():
    analysis, fusion = _risk(
        "Hello, this is SBI customer care about your new card. "
        "Your card will arrive by Friday. Have a good day."
    )
    assert analysis.stages == [AUTHORITY], f"unexpected stages: {analysis.stages}"
    assert fusion["band"] == S.BAND_GREEN


def test_empty_and_whitespace_input_is_inert():
    for text in (None, "", "   ", "\n\t"):
        analysis = classify_stages(text)
        assert analysis.stages == []
        assert analysis.payment_attempt is False
        assert explain(analysis, fuse_risk([]))


# ---------------------------------------------------------------------------
# Spans, for the transcript highlighting
# ---------------------------------------------------------------------------


def test_match_spans_are_valid_and_non_overlapping_per_stage():
    text = CLONED_RELATIVE_SCRIPT
    analysis = classify_stages(text)
    assert analysis.matches, "expected matches to highlight"

    by_stage = {}
    for m in analysis.matches:
        assert 0 <= m.start < m.end <= len(text), f"span out of range: {m}"
        assert text[m.start:m.end] == m.text, f"span text mismatch: {m!r}"
        by_stage.setdefault(m.stage, []).append(m)

    for stage, items in by_stage.items():
        items.sort(key=lambda m: m.start)
        for a, b in zip(items, items[1:]):
            assert a.end <= b.start, f"{stage} spans overlap: {a} / {b}"


def test_amount_span_is_tight():
    analysis = classify_stages("Please send ₹50,000 to this UPI ID")
    amounts = [m for m in analysis.matches if "50,000" in m.text]
    assert amounts, "amount not matched"
    assert amounts[0].text == amounts[0].text.strip(), f"span has padding: {amounts[0].text!r}"


# ---------------------------------------------------------------------------
# Fusion behaviour
# ---------------------------------------------------------------------------


def test_adding_evidence_never_lowers_risk():
    """Noisy-OR must be monotonic — more evidence cannot mean less risk."""
    chains = [
        [],
        [AUTHORITY],
        [AUTHORITY, URGENCY],
        [AUTHORITY, URGENCY, CREDENTIAL],
        [AUTHORITY, URGENCY, CREDENTIAL, SECRECY],
        [AUTHORITY, URGENCY, CREDENTIAL, SECRECY, REMOTE],
    ]
    for acoustic in (0.0, 0.4, 0.95):
        scores = [fuse_risk(c, acoustic_prob=acoustic)["risk_score"] for c in chains]
        assert scores == sorted(scores), f"non-monotonic at acoustic={acoustic}: {scores}"


def test_acoustic_evidence_alone_cannot_reach_red():
    """A synthetic voice is not by itself a scam.

    Plenty of legitimate calls are synthesised — IVR menus, accessibility
    tools, automated appointment reminders. Convicting on the vocoder alone is
    how the existing threshold ladder produces its false positives.
    """
    fusion = fuse_risk([], acoustic_prob=1.0)
    assert fusion["band"] != S.BAND_RED, f"acoustic alone reached red: {fusion}"


def test_credential_demand_in_synthetic_voice_floors_at_80():
    fusion = fuse_risk([CREDENTIAL], acoustic_prob=0.9)
    assert fusion["risk_score"] >= 80.0
    assert fusion["hard_rule"] == "CREDENTIAL_DEMAND_IN_SYNTHETIC_VOICE"


def test_credential_under_pressure_floors_at_70_without_acoustic():
    fusion = fuse_risk(
        [URGENCY, CREDENTIAL], acoustic_prob=0.0, credential_high_sensitivity=True
    )
    assert fusion["risk_score"] >= 70.0
    assert fusion["band"] == S.BAND_RED
    assert fusion["hard_rule"] == "CREDENTIAL_DEMAND_UNDER_PRESSURE"


def test_low_sensitivity_credential_under_pressure_does_not_floor():
    """Asking to confirm an account number is not asking for an OTP.

    A real bank may reasonably confirm an account number, so that demand must
    not auto-convict the way a PIN or CVV request does.
    """
    fusion = fuse_risk(
        [URGENCY, CREDENTIAL], acoustic_prob=0.0, credential_high_sensitivity=False
    )
    assert fusion["hard_rule"] is None, f"unexpected hard rule: {fusion}"


def test_blacklisted_caller_short_circuits_to_99():
    fusion = fuse_risk([], acoustic_prob=0.0, blacklisted=True)
    assert fusion["risk_score"] == 99.0
    assert fusion["band"] == S.BAND_RED
    assert fusion["hard_rule"] == "KNOWN_SCAMMER_NUMBER"


def test_voip_bump_only_applies_above_amber():
    low = fuse_risk([], acoustic_prob=0.0, voip_spoofed=True)
    high = fuse_risk([AUTHORITY, URGENCY], acoustic_prob=0.0, voip_spoofed=True)
    plain = fuse_risk([AUTHORITY, URGENCY], acoustic_prob=0.0, voip_spoofed=False)
    assert low["risk_score"] == fuse_risk([])["risk_score"], "VoIP bumped a green call"
    assert high["risk_score"] > plain["risk_score"], "VoIP bump not applied above amber"


def test_risk_score_stays_in_range():
    fusion = fuse_risk(
        list(S.STAGE_ORDER),
        acoustic_prob=1.0,
        payment_attempt=True,
        voip_spoofed=True,
        credential_high_sensitivity=True,
    )
    assert 0.0 <= fusion["risk_score"] <= 100.0


def test_acoustic_prob_is_clamped():
    for bad in (-5.0, 1.5, 99.0):
        fusion = fuse_risk([AUTHORITY], acoustic_prob=bad)
        assert 0.0 <= fusion["acoustic_prob"] <= 1.0


# ---------------------------------------------------------------------------
# Localisation
# ---------------------------------------------------------------------------


def test_detect_language_on_native_script():
    assert S.detect_language("নমস্কার কাকিমা, আমি শুভঙ্কর বলছি") == "bn"
    assert S.detect_language("नमस्ते भाई, कल कॉलेज की क्लास") == "hi"
    assert S.detect_language("Hi team, let's reschedule the meeting") == "en"
    assert S.detect_language(None) == "en"
    assert S.detect_language("") == "en"


def test_detect_language_on_what_the_asr_actually_emits():
    """The ASR returns romanised and Shahmukhi text, not clean native script.

    These are real Faster-Whisper outputs for the shipped fixtures. Tagging
    them 'en' would hand a Hindi or Bengali caller an English warning screen,
    which is the whole failure this localisation was meant to prevent.
    """
    romanised_hi = (
        "Namaskar, HDFC Bank Head Office  Aapke pan kard me visangati pai gayi hai  "
        "Turant aapna ATM pin share karke verify kariye"
    )
    assert S.detect_language(romanised_hi) == "hi"

    romanised_bn = (
        "Joruri Bank Shotorkota, apnar State Bank Account Sthogit Kora Hoyeche. "
        "Obilombe OTP pathan."
    )
    assert S.detect_language(romanised_bn) == "bn"

    shahmukhi_hi = "نمستے بھائی قل کی پریچشا کے نوڑس تیار ہو گئے ہیں"
    assert S.detect_language(shahmukhi_hi) == "hi"


def test_detect_language_does_not_tag_english_as_indic():
    """A stray collision must not flip an English call to Hindi or Bengali."""
    for text in (
        "Hi team, let's reschedule our design review meeting to Thursday afternoon at 3 PM.",
        "Your card will arrive by Friday. Have a good day.",
        "I'll send the report after standup, no rush.",
        "Yes that is fine, thanks",
    ):
        assert S.detect_language(text) == "en", f"English misdetected: {text}"


def test_normalize_lang_handles_junk_and_regional_tags():
    assert normalize_lang(None) == "en"
    assert normalize_lang("") == "en"
    assert normalize_lang("bn") == "bn"
    assert normalize_lang("BN") == "bn"
    assert normalize_lang("bn-IN") == "bn"
    assert normalize_lang("hi_IN") == "hi"
    assert normalize_lang("ta") == "en", "unsupported language must fall back, not raise"
    assert normalize_lang("en-GB") == "en"


def test_every_supported_language_has_a_complete_table():
    for lang in S.SUPPORTED_LANGS:
        assert set(S.STAGE_LABELS_BY_LANG[lang]) == set(S.STAGE_ORDER), lang
        assert set(S.STAGE_EXPLANATIONS_BY_LANG[lang]) == set(S.STAGE_ORDER), lang
        assert set(S.STAGE_CLAUSES_BY_LANG[lang]) == set(S.STAGE_ORDER), lang
        assert set(S.BAND_LABELS_BY_LANG[lang]) == {S.BAND_RED, S.BAND_AMBER, S.BAND_GREEN}, lang
        # Every language must answer every event key, or a live call would
        # silently fall back to English mid-log.
        assert set(S.EVENT_STRINGS[lang]) == set(S.EVENT_STRINGS["en"]), lang
        assert set(S.VERDICT_STRINGS[lang]) == set(S.VERDICT_STRINGS["en"]), lang


def test_verdicts_differ_per_language_and_are_not_english_leakage():
    analysis, fusion = _risk(CLONED_RELATIVE_SCRIPT, acoustic_prob=0.85)
    verdicts = {lang: explain(analysis, fusion, lang) for lang in S.SUPPORTED_LANGS}
    assert len(set(verdicts.values())) == 3, f"verdicts not distinct: {verdicts}"
    assert any("ঀ" <= c <= "৿" for c in verdicts["bn"]), "Bengali verdict has no Bengali"
    assert any("ऀ" <= c <= "ॿ" for c in verdicts["hi"]), "Hindi verdict has no Devanagari"


def test_indic_sentences_end_with_a_danda():
    analysis, fusion = _risk(CLONED_RELATIVE_SCRIPT)
    for lang in ("hi", "bn"):
        assert explain(analysis, fusion, lang).endswith("।"), f"{lang} verdict not terminated with danda"
    assert explain(analysis, fusion, "en").endswith("."), "English verdict not terminated with a full stop"


def test_band_and_stage_labels_localise():
    assert band_label(S.BAND_RED, "en") != band_label(S.BAND_RED, "bn")
    assert S.stage_label(CREDENTIAL, "hi") != S.stage_label(CREDENTIAL, "en")
    assert S.stage_label("NOT_A_STAGE", "en") == "NOT_A_STAGE", "unknown stage must pass through"


def test_event_text_formats_and_degrades_gracefully():
    assert "Authority" in event_text("stage_detected", "en", label="Authority")
    assert event_text("stage_detected", "bn", label="পরিচয়ের দাবি")
    # A missing format field must not raise mid-call.
    assert event_text("stage_detected", "en") == S.EVENT_STRINGS["en"]["stage_detected"]
    # An unknown key degrades to the key itself rather than a KeyError.
    assert event_text("no_such_key", "en") == "no_such_key"


def test_as_dict_is_json_shaped_and_localised():
    analysis = classify_stages(CLONED_RELATIVE_SCRIPT)
    payload = analysis.as_dict("bn")
    assert payload["lang"] == "bn"
    assert payload["stage_count"] == len(payload["stages"])
    assert payload["stage_total"] == len(S.STAGE_ORDER)
    assert set(payload["labels"]) == set(S.STAGE_ORDER)
    assert len(payload["explanations"]) == len(payload["stages"])
    for match in payload["matches"]:
        assert set(match) == {"stage", "label", "start", "end", "text"}

    import json

    json.dumps(payload)  # must be serialisable as-is


def _main():
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    failures = []
    for fn in tests:
        try:
            fn()
            print(f"  PASS  {fn.__name__}")
        except AssertionError as e:
            failures.append((fn.__name__, e))
            print(f"  FAIL  {fn.__name__}: {e}")
        except Exception as e:
            failures.append((fn.__name__, e))
            print(f"  ERROR {fn.__name__}: {type(e).__name__}: {e}")
    print(f"\n{len(tests) - len(failures)}/{len(tests)} passed")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(_main())
