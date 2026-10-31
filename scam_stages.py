"""
VaniShield AI — Social-Engineering Stage Classifier & Noisy-OR Risk Fusion

The acoustic pillar in `vanishield_engine` answers "is this voice synthetic?".
This module answers the orthogonal question "is this conversation a scam script?"
by decomposing the transcript into the five stages of a vishing playbook:

    AUTHORITY  — caller claims an identity (bank / police / relative)
    URGENCY    — manufactured panic or a deadline
    CREDENTIAL — an actual demand for money, OTP, PIN or card data
    SECRECY    — instruction to keep the call hidden from family
    REMOTE     — instruction to install a remote-access or screen-share app

Why stages rather than a flat keyword list: a single keyword is close to
worthless as evidence ("my OTP never arrived" is not a scam), whereas the
*combination* of an identity claim, a deadline and a credential demand is
close to conclusive. Stages also give the operator an explanation —
"caller claimed to be your grandson, then asked for money, then asked you to
keep it secret" — which a bag of matched words cannot.

Every stage carries character offsets back into the original transcript so the
UI can highlight the exact phrase that fired, instead of re-searching the text
client-side and highlighting the wrong occurrence.

Awareness advice is masked before stage matching. A bank's own recorded warning
("never share your OTP with anyone") contains the highest-signal credential
words in the language and must not fire CREDENTIAL. This mirrors, and shares
intent with, the AWARENESS_WARNING_PATTERNS guard in `vanishield_engine`.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

# ---------------------------------------------------------------------------
# Stage identifiers
# ---------------------------------------------------------------------------

AUTHORITY = "AUTH"
URGENCY = "URG"
CREDENTIAL = "CRED"
SECRECY = "SECR"
REMOTE = "REMOTE"

STAGE_ORDER: Tuple[str, ...] = (AUTHORITY, URGENCY, CREDENTIAL, SECRECY, REMOTE)

# ---------------------------------------------------------------------------
# Localisation
#
# The person this product protects is, very often, an elderly Bengali or Hindi
# speaker being told their grandson is in hospital. A verdict they cannot read
# is not a verdict. So every user-visible string lives here, keyed by language,
# and the API serves it in the language the caller is actually speaking — as
# reported by VaniShieldCore.detect_language() on the live transcript.
#
# English is the fallback for any language we have no table for, never a
# hard-coded default in the UI.
# ---------------------------------------------------------------------------

SUPPORTED_LANGS: Tuple[str, ...] = ("en", "hi", "bn")
DEFAULT_LANG = "en"

# Risk banding, shared by the dashboard and the on-call alert screen.
# Declared here because the localised band labels below key off these values.
BAND_RED = "red"
BAND_AMBER = "amber"
BAND_GREEN = "green"

RED_THRESHOLD = 65.0
AMBER_THRESHOLD = 35.0

# Short chip labels. Kept deliberately terse — these sit in a 5-across strip
# on a 380px phone screen.
STAGE_LABELS_BY_LANG: Dict[str, Dict[str, str]] = {
    "en": {
        AUTHORITY: "Authority",
        URGENCY: "Urgency",
        CREDENTIAL: "Money / OTP",
        SECRECY: "Secrecy",
        REMOTE: "Remote app",
    },
    "hi": {
        AUTHORITY: "पहचान का दावा",
        URGENCY: "जल्दबाज़ी",
        CREDENTIAL: "पैसा / ओटीपी",
        SECRECY: "छिपाना",
        REMOTE: "रिमोट ऐप",
    },
    "bn": {
        AUTHORITY: "পরিচয়ের দাবি",
        URGENCY: "তাড়াহুড়ো",
        CREDENTIAL: "টাকা / ওটিপি",
        SECRECY: "গোপনীয়তা",
        REMOTE: "রিমোট অ্যাপ",
    },
}

# Full sentences, used as chip tooltips and in the forensic report.
STAGE_EXPLANATIONS_BY_LANG: Dict[str, Dict[str, str]] = {
    "en": {
        AUTHORITY: "Caller claims an identity you are expected to trust",
        URGENCY: "Caller manufactures panic or a deadline",
        CREDENTIAL: "Caller asks you to hand over money or a secret code",
        SECRECY: "Caller asks you to keep the call hidden",
        REMOTE: "Caller asks you to install a remote-access app",
    },
    "hi": {
        AUTHORITY: "कॉल करने वाला ऐसी पहचान बता रहा है जिस पर आप भरोसा करें",
        URGENCY: "कॉल करने वाला डर या जल्दबाज़ी पैदा कर रहा है",
        CREDENTIAL: "कॉल करने वाला पैसे या गुप्त कोड मांग रहा है",
        SECRECY: "कॉल करने वाला कह रहा है कि किसी को न बताएं",
        REMOTE: "कॉल करने वाला रिमोट ऐप इंस्टॉल करने को कह रहा है",
    },
    "bn": {
        AUTHORITY: "কলকারী এমন পরিচয় দিচ্ছে যাতে আপনি বিশ্বাস করেন",
        URGENCY: "কলকারী ভয় বা তাড়াহুড়ো তৈরি করছে",
        CREDENTIAL: "কলকারী টাকা বা গোপন কোড চাইছে",
        SECRECY: "কলকারী বলছে কাউকে না জানাতে",
        REMOTE: "কলকারী রিমোট অ্যাপ ইনস্টল করতে বলছে",
    },
}

# Bare verb phrases, chained into the one-line verdict so it reads as prose
# ("The caller claims an identity you are expected to trust, then manufactures
# panic, then asks you to hand over money") rather than as a list of sentences
# stapled together.
STAGE_CLAUSES_BY_LANG: Dict[str, Dict[str, str]] = {
    "en": {
        AUTHORITY: "claims an identity you are expected to trust",
        URGENCY: "manufactures panic or a deadline",
        CREDENTIAL: "asks you to hand over money or a secret code",
        SECRECY: "asks you to keep the call hidden",
        REMOTE: "asks you to install a remote-access app",
    },
    "hi": {
        AUTHORITY: "भरोसे वाली पहचान बता रहा है",
        URGENCY: "डर या जल्दबाज़ी पैदा कर रहा है",
        CREDENTIAL: "पैसे या गुप्त कोड मांग रहा है",
        SECRECY: "किसी को न बताने को कह रहा है",
        REMOTE: "रिमोट ऐप इंस्टॉल करने को कह रहा है",
    },
    "bn": {
        AUTHORITY: "বিশ্বাসযোগ্য পরিচয় দিচ্ছে",
        URGENCY: "ভয় বা তাড়াহুড়ো তৈরি করছে",
        CREDENTIAL: "টাকা বা গোপন কোড চাইছে",
        SECRECY: "কাউকে না জানাতে বলছে",
        REMOTE: "রিমোট অ্যাপ ইনস্টল করতে বলছে",
    },
}

BAND_LABELS_BY_LANG: Dict[str, Dict[str, str]] = {
    "en": {
        BAND_RED: "Red · likely scam",
        BAND_AMBER: "Amber · caution",
        BAND_GREEN: "Green · safe",
    },
    "hi": {
        BAND_RED: "लाल · धोखाधड़ी की आशंका",
        BAND_AMBER: "पीला · सावधान",
        BAND_GREEN: "हरा · सुरक्षित",
    },
    "bn": {
        BAND_RED: "লাল · প্রতারণা সম্ভব",
        BAND_AMBER: "হলুদ · সতর্ক থাকুন",
        BAND_GREEN: "সবুজ · নিরাপদ",
    },
}

# Sentence frames for the one-line verdict.
VERDICT_STRINGS: Dict[str, Dict[str, str]] = {
    "en": {
        "blacklisted": "This number has already been reported for fraud on this device.",
        "synthetic_only": "This voice appears to be AI-generated, though nothing has been asked for yet.",
        "awareness": "This call is giving safety advice, not asking for anything.",
        "nothing": "Nothing suspicious has been said so far.",
        "subject": "The caller",
        "synthetic_prefix": "This voice appears to be AI-generated, and the caller",
        "then": ", then ",
        "terminator": ".",
        "silent": "No voice on the line right now. Monitoring continues.",
    },
    "hi": {
        "blacklisted": "इस नंबर की पहले ही धोखाधड़ी की शिकायत दर्ज है।",
        "synthetic_only": "यह आवाज़ एआई से बनी लगती है, पर अभी कुछ मांगा नहीं गया है।",
        "awareness": "यह कॉल सुरक्षा सलाह दे रही है, कुछ मांग नहीं रही।",
        "nothing": "अब तक कुछ संदिग्ध नहीं कहा गया है।",
        "subject": "कॉल करने वाला",
        "synthetic_prefix": "यह आवाज़ एआई से बनी लगती है, और कॉल करने वाला",
        "then": ", फिर ",
        # Devanagari and Bengali end a sentence with the danda, not a full stop.
        "terminator": "।",
        "silent": "अभी लाइन पर कोई आवाज़ नहीं। निगरानी जारी है।",
    },
    "bn": {
        "blacklisted": "এই নম্বরটি আগেই প্রতারণার জন্য রিপোর্ট করা হয়েছে।",
        "synthetic_only": "এই কণ্ঠস্বর এআই-তৈরি মনে হচ্ছে, তবে এখনও কিছু চাওয়া হয়নি।",
        "awareness": "এই কল নিরাপত্তা পরামর্শ দিচ্ছে, কিছু চাইছে না।",
        "nothing": "এখনও পর্যন্ত সন্দেহজনক কিছু বলা হয়নি।",
        "subject": "কলকারী",
        "synthetic_prefix": "এই কণ্ঠস্বর এআই-তৈরি মনে হচ্ছে, এবং কলকারী",
        "then": ", তারপর ",
        "terminator": "।",
        "silent": "এখন লাইনে কোনো আওয়াজ নেই। নজরদারি চালু আছে।",
    },
}

# Event-log lines. These are user-visible on both frontends, and both the
# one-shot REST path (`vanishield_engine._synth_events`) and the streaming path
# (`call_session.CallSession`) render them — so they live here, once, rather
# than in two tables that would quietly drift apart.
EVENT_STRINGS: Dict[str, Dict[str, str]] = {
    "en": {
        "call_started": "Call started",
        "call_ended": "Call ended · volatile memory purged",
        "stage_detected": "Stage detected: {label}",
        "synthetic": "Synthetic voice above threshold · {pct}%",
        "payment": "Payment attempt detected",
        "payment_amount": "Payment attempt detected · ₹{amount}",
        "breaker": "Circuit breaker engaged · banking and UPI locked",
        "risk_red": "Risk turned RED ({score})",
        "risk_amber": "Risk turned AMBER ({score})",
        "risk_green": "Risk returned to GREEN ({score})",
        "risk_state_red": "Risk is RED ({score})",
        "risk_state_amber": "Risk is AMBER ({score})",
        "risk_state_green": "Risk is GREEN ({score})",
        "awareness_ignored": "Safety advice recognised · credential mention ignored",
        "hard_rule": "Hard rule applied · {rule}",
        "mic_silent": "Line silent · monitoring on standby",
    },
    "hi": {
        "call_started": "कॉल शुरू",
        "call_ended": "कॉल समाप्त · मेमोरी साफ़",
        "stage_detected": "चरण मिला: {label}",
        "synthetic": "आवाज़ एआई से बनी लगती है · {pct}%",
        "payment": "पैसे भेजने की कोशिश पकड़ी गई",
        "payment_amount": "पैसे भेजने की कोशिश · ₹{amount}",
        "breaker": "सुरक्षा लॉक लगा · बैंकिंग और यूपीआई बंद",
        "risk_red": "जोखिम लाल हुआ ({score})",
        "risk_amber": "जोखिम पीला हुआ ({score})",
        "risk_green": "जोखिम हरा हुआ ({score})",
        "risk_state_red": "जोखिम लाल है ({score})",
        "risk_state_amber": "जोखिम पीला है ({score})",
        "risk_state_green": "जोखिम हरा है ({score})",
        "awareness_ignored": "सुरक्षा सलाह पहचानी गई · ओटीपी का ज़िक्र नज़रअंदाज़",
        "hard_rule": "सख़्त नियम लागू · {rule}",
        "mic_silent": "लाइन शांत · निगरानी जारी",
    },
    "bn": {
        "call_started": "কল শুরু",
        "call_ended": "কল শেষ · মেমোরি মুছে ফেলা হয়েছে",
        "stage_detected": "ধাপ শনাক্ত: {label}",
        "synthetic": "কণ্ঠস্বর এআই-তৈরি মনে হচ্ছে · {pct}%",
        "payment": "টাকা পাঠানোর চেষ্টা ধরা পড়েছে",
        "payment_amount": "টাকা পাঠানোর চেষ্টা · ₹{amount}",
        "breaker": "সুরক্ষা লক · ব্যাংকিং ও ইউপিআই বন্ধ",
        "risk_red": "ঝুঁকি লাল হল ({score})",
        "risk_amber": "ঝুঁকি হলুদ হল ({score})",
        "risk_green": "ঝুঁকি সবুজ হল ({score})",
        "risk_state_red": "ঝুঁকি লাল ({score})",
        "risk_state_amber": "ঝুঁকি হলুদ ({score})",
        "risk_state_green": "ঝুঁকি সবুজ ({score})",
        "awareness_ignored": "নিরাপত্তা পরামর্শ শনাক্ত · ওটিপি উল্লেখ উপেক্ষা করা হয়েছে",
        "hard_rule": "কঠোর নিয়ম প্রয়োগ · {rule}",
        "mic_silent": "লাইন নীরব · নজরদারি চালু",
    },
}


def event_text(key: str, lang: str = DEFAULT_LANG, **fields) -> str:
    """Formats one localised event-log line, falling back to English."""
    table = EVENT_STRINGS[normalize_lang(lang)]
    template = table.get(key) or EVENT_STRINGS[DEFAULT_LANG].get(key, key)
    try:
        return template.format(**fields)
    except (KeyError, IndexError):
        return template


# Function words and high-frequency content words that identify romanised
# Hindi or Bengali. Chosen to be rare or absent in English so that ordinary
# English conversation scores zero: "ami"/"apni"/"hobe" are not English words,
# and the few that collide ("hai", "na", "kya") need company before they count.
ROMANISED_MARKERS: Dict[str, Tuple[str, ...]] = {
    "hi": (
        "namaskar", "namaste", "aap", "aapka", "aapke", "aapki", "apna", "apne",
        "kripya", "dhanyavad", "hain", "karen", "kariye", "kijiye", "karke",
        "bhai", "behen", "paisa", "paise", "khata", "turant", "jaldi", "nahi",
        "mera", "meri", "tumhara", "tumhari", "yeh", "woh", "kuch", "bahut",
        "abhi", "batana", "batao", "bhejo", "rupaye", "rupay", "ghar", "beta",
        "sahab", "ji", "haan", "theek", "accha", "kaise", "kyun", "matlab",
    ),
    "bn": (
        "ami", "apni", "apnar", "apnader", "tumi", "tomar", "amar", "amader",
        "korun", "koro", "korchi", "korbe", "ache", "achen", "hobe", "hoyeche",
        "bolchi", "bolun", "bolo", "joruri", "taka", "kichu", "ekhuni",
        "taratari", "bhalo", "kemon", "dada", "didi", "kaku", "kakima", "mashi",
        "obilombe", "pathan", "pathao", "gopon", "bondho", "khub", "ekta",
        "jonno", "theke", "ebong", "noy", "hoy", "chhe", "kora", "kore",
    ),
}

_WORD_RE = re.compile(r"[a-z]+")

# Shahmukhi / Urdu script. Faster-Whisper routinely renders Hindi — and
# sometimes Bengali — in Arabic script, which the repo's own SCAM_PATTERNS
# already acknowledge. Treat it as Indic rather than English, defaulting to
# Hindi unless romanised Bengali markers outweigh it.
_ARABIC_RANGE = ("؀", "ۿ")

# Two markers rather than one: a single "ji" or "na" in an English sentence is
# noise, two or more is a language.
ROMANISED_MIN_MARKERS = 2


def detect_language(text: Optional[str]) -> str:
    """
    Language tag for a transcript: 'bn', 'hi' or 'en'.

    Native script is decisive. Beyond that, this has to cope with what the ASR
    actually emits rather than what the caller actually spoke: Faster-Whisper
    returns Indic speech in Latin script as often as in Devanagari or Bengali,
    and sometimes in Shahmukhi. A script-only check tags all of those 'en' and
    hands an elderly Bengali speaker an English warning screen, which defeats
    the point of localising at all.
    """
    if not text:
        return DEFAULT_LANG

    if any("ঀ" <= c <= "৿" for c in text):
        return "bn"
    if any("ऀ" <= c <= "ॿ" for c in text):
        return "hi"

    words = set(_WORD_RE.findall(text.lower()))
    scores = {
        lang: len(words & set(markers))
        for lang, markers in ROMANISED_MARKERS.items()
    }

    if any(_ARABIC_RANGE[0] <= c <= _ARABIC_RANGE[1] for c in text):
        scores["hi"] += ROMANISED_MIN_MARKERS

    best = max(scores, key=lambda k: scores[k])
    if scores[best] >= ROMANISED_MIN_MARKERS:
        return best
    return DEFAULT_LANG


# Whisper language ids that should map onto one of our three UI languages
# rather than falling back to English. Urdu and Punjabi are routinely returned
# for Hindi speech; Assamese shares most of its script and idiom with Bengali.
ASR_LANG_ALIASES: Dict[str, str] = {
    "ur": "hi",
    "pa": "hi",
    "mr": "hi",
    "ne": "hi",
    "sa": "hi",
    "as": "bn",
}


def lang_from_asr(asr_language: Optional[str]) -> str:
    """Maps an ASR language id onto a UI language, via the aliases above."""
    if not asr_language:
        return DEFAULT_LANG
    base = str(asr_language).strip().lower().replace("_", "-").split("-")[0]
    return ASR_LANG_ALIASES.get(base, normalize_lang(base))


def normalize_lang(lang: Optional[str]) -> str:
    """Maps an arbitrary language tag onto a table we actually have."""
    if not lang:
        return DEFAULT_LANG
    base = str(lang).strip().lower().replace("_", "-").split("-")[0]
    return base if base in SUPPORTED_LANGS else DEFAULT_LANG


def stage_labels(lang: str = DEFAULT_LANG) -> Dict[str, str]:
    return dict(STAGE_LABELS_BY_LANG[normalize_lang(lang)])


def stage_label(stage: str, lang: str = DEFAULT_LANG) -> str:
    return STAGE_LABELS_BY_LANG[normalize_lang(lang)].get(stage, stage)


def stage_explanation(stage: str, lang: str = DEFAULT_LANG) -> str:
    return STAGE_EXPLANATIONS_BY_LANG[normalize_lang(lang)].get(stage, stage)


def stage_clause(stage: str, lang: str = DEFAULT_LANG) -> str:
    return STAGE_CLAUSES_BY_LANG[normalize_lang(lang)].get(stage, stage)


# Back-compatible English views, for callers that predate localisation.
STAGE_LABELS: Dict[str, str] = dict(STAGE_LABELS_BY_LANG["en"])
STAGE_EXPLANATIONS: Dict[str, str] = dict(STAGE_EXPLANATIONS_BY_LANG["en"])

# Per-stage likelihood that a *legitimate* call is excluded by this evidence.
# Used as the noisy-OR factor below, so a larger weight means stronger
# evidence. CREDENTIAL and REMOTE dominate because a genuine bank, police
# officer or relative never needs either.
STAGE_WEIGHTS: Dict[str, float] = {
    AUTHORITY: 0.15,
    URGENCY: 0.20,
    CREDENTIAL: 0.40,
    SECRECY: 0.15,
    REMOTE: 0.35,
}

# The acoustic probability above which the voice is treated as synthetic.
# Matches the 0.7 threshold line drawn on the per-window score chart.
ACOUSTIC_SYNTHETIC_THRESHOLD = 0.70


# ---------------------------------------------------------------------------
# Awareness advice — matched first, then masked out of stage detection
# ---------------------------------------------------------------------------

AWARENESS_PATTERNS: Tuple[str, ...] = (
    r"\b(?:never|not?|do not|don'?t|should not|must not|avoid)\s+(?:ever\s+)?"
    r"(?:share|shares|sharing|give|giving|disclose|tell|telling|send|enter|reveal)\b"
    r"[^.!?]{0,60}?\b(?:otp|pin|cvv|password|passcode|details|credential|card number|account number)\b",
    r"\b(?:bank|rbi|police|we)\s+(?:will\s+)?(?:never|does not|do not|don'?t)\s+"
    r"(?:ask|call to ask|request|demand|need)\b[^.!?]{0,60}?"
    r"\b(?:otp|pin|cvv|password|credentials|card number)\b",
    r"\b(?:beware|be aware|be careful|stay alert)\s+of\s+(?:fraud|scam|fake)\b",
    r"(?:ओटीपी|पिन|सीवीवी|पासवर्ड|डिटेल्स)\s*(?:कभी\s*)?(?:किसी\s*(?:को|के\s*साथ)\s*)?"
    r"(?:मत|ना|नहीं)\s*(?:दीजिए|देना|दें|बताइए|बताना|शेयर|करना|करें)",
    r"बैंक\s*(?:कभी\s*)?(?:आपसे\s*)?(?:ओटीपी|पिन|पासवर्ड|सीवीवी)\s*नहीं\s*(?:मांगता|मांगती|मांगते)",
    r"(?:ওটিপি|পিন|সিভিভি|পাসওয়ার্ড)\s*(?:কখনও\s*)?(?:কাউকে\s*)?"
    r"(?:দেবেন\s*না|দিবেন\s*না|বলবেন\s*না|শেয়ার\s*করবেন\s*না)",
    r"ব্যাংক\s*(?:কখনও\s*)?(?:আপনার\s*)?(?:ওটিপি|পিন|পাসওয়ার্ড|সিভিভি)\s*চায়\s*না",
)


# ---------------------------------------------------------------------------
# Demand verbs — CREDENTIAL requires one of these near the sensitive noun,
# so that merely *mentioning* an OTP is not treated as demanding one.
# ---------------------------------------------------------------------------

DEMAND_VERBS = (
    r"\b(?:share|send|sent|tell|give|provide|transfer|deposit|pay|enter|type|read out|"
    r"forward|confirm|verify|update|reverse|release|credit|install|download|open)\b",
    # Indic verb stems are left deliberately open-ended (no inflection suffix
    # required): Hindi and Bengali attach aspect and politeness endings freely
    # ("शेयर करके", "शेयर करें", "প্রদান করুন"), and \b does not apply to
    # Devanagari or Bengali, so anchoring on the stem is both safer and the
    # only thing that generalises across the dialects in the test fixtures.
    r"(?:भेज|दीजिए|दे\s*दो|दें|बताओ|बताइए|शेयर\s*कर|ट्रांसफर\s*कर|डाल|"
    r"जमा\s*कर|लिख|प्रदान\s*कर|वेरीफाई|वेरिफाई|अपडेट\s*कर|कन्फर्म)",
    r"(?:পাঠা|দাও|দিন|দিবেন|দেবেন|শেয়ার\s*কর|ট্রান্সফার\s*কর|জমা\s*দ|"
    r"প্রদান\s*কর|ভেরিফাই|আপডেট\s*কর|কনফার্ম|লিখ)",
    # Romanised Hindi/Bengali. Faster-Whisper very often returns Indic speech
    # in Latin script rather than the native one — the shipped Hindi fixture
    # transcribes as "Namaskar, HDFC Bank Head Office. Aapke pan kard..." — so
    # native-script patterns alone miss the live ASR path entirely.
    # `\w*` after each stem rather than a bare \b: romanised Hindi and Bengali
    # attach their endings directly to the verb ("kar" -> "karke", "karen",
    # "kariye"), so an anchored stem would match the prefix and then fail the
    # word boundary. The leading \b still prevents matching mid-word.
    r"\b(?:bhej\w*|bata(?:o|iye|na)\w*|share\s*kar\w*|transfer\s*kar\w*|dal(?:o|iye)|"
    r"jama\s*kar\w*|pradan\s*kar\w*|verify\s*kar\w*|re-?verify|update\s*kar\w*|"
    r"confirm\s*kar\w*|de\s*d(?:o|ijiye))\b",
    r"\b(?:patha\w*|share\s*kor\w*|transfer\s*kor\w*|joma\s*d\w*|pradan\s*kor\w*|"
    r"diye\s*din)\b",
)

# How far from a demand verb a sensitive noun still counts as demanded.
DEMAND_PROXIMITY_CHARS = 90


# ---------------------------------------------------------------------------
# Stage patterns
# ---------------------------------------------------------------------------

STAGE_PATTERNS: Dict[str, Tuple[str, ...]] = {
    AUTHORITY: (
        # Institutional identity claim in English / Hinglish.
        r"\b(?:this is|i am|i'?m|calling from|speaking from|we are from|on behalf of)\b"
        r"[^.!?]{0,40}?\b(?:sbi|state bank|hdfc|icici|axis|kotak|punjab national|pnb|bank|rbi|"
        r"reserve bank|police|cbi|customs|income tax|trai|trai department|cyber cell|cyber crime|"
        r"fraud team|fraud department|customer care|head office|branch manager|aadhaar|uidai)\b",
        # Bare institutional self-identification, with no "this is" lead-in.
        # Real vishing openers routinely drop the verb entirely —
        # "नमस्कार, एचडीएफसी बैंक हेड ऑफिस" — so requiring a first-person
        # pronoun misses the most common Hindi and English scripts outright.
        r"\b(?:notice|call|calling|message|alert|warning)\s+from\b[^.!?]{0,40}?"
        r"\b(?:sbi|state bank|hdfc|icici|axis|kotak|pnb|bank|rbi|police|cbi|customs|"
        r"income tax|trai|cyber cell|cyber crime)\b",
        r"\b(?:sbi|state bank(?: of india)?|hdfc|icici|axis|kotak|punjab national|pnb|rbi|"
        r"reserve bank|cbi|customs|income tax|trai|uidai)\b[^.!?]{0,25}?"
        r"\b(?:bank|fraud team|fraud department|head office|customer care|branch|"
        r"department|helpline|security team)\b",
        # Transliterated institution names, which is how the ASR renders them
        # on Hindi and Bengali calls: "एचडीएफसी बैंक हेड ऑफिस".
        r"(?:एसबीआई|एचडीएफसी|आईसीआईसीआई|एक्सिस|स्टेट\s*बैंक|पंजाब\s*नेशनल|आरबीआई|"
        r"सीबीआई|आयकर\s*विभाग|यूआईडीएआई)[^।.!?]{0,25}?"
        r"(?:बैंक|हेड\s*ऑफिस|कस्टमर\s*केयर|शाखा|ब्रांच|विभाग|फ्रॉड\s*टीम|हेल्पलाइन)",
        r"(?:এসবিআই|এইচডিএফসি|আইসিআইসিআই|অ্যাক্সিস|স্টেট\s*ব্যাংক|পাঞ্জাব\s*ন্যাশনাল|"
        r"আরবিআই|সিবিআই)[^।.!?]{0,25}?"
        r"(?:ব্যাংক|ব্যাঙ্ক|হেড\s*অফিস|কাস্টমার\s*কেয়ার|শাখা|ব্রাঞ্চ|বিভাগ|হেল্পলাইন)",
        # Relative impersonation — the "grandson in trouble" opening.
        r"\b(?:i am|i'?m|this is|it'?s)\s+(?:your\s+)?"
        r"(?:son|daughter|grandson|granddaughter|nephew|niece|brother|sister|cousin|uncle|aunt)\b",
        r"\b(?:your)\s+(?:son|daughter|grandson|granddaughter|nephew|niece)\s+(?:here|speaking|calling)\b",
        # Hindi institutional / relative claim.
        r"(?:मैं|हम)\s*[^।.!?]{0,30}?(?:बैंक|आरबीआई|पुलिस|सीबीआई|कस्टम|कस्टम्स|आयकर|"
        r"साइबर\s*सेल|हेड\s*ऑफिस|कस्टमर\s*केयर|शाखा\s*प्रबंधक|ब्रांच\s*मैनेजर)",
        r"(?:मैं\s*(?:आपका|तुम्हारा)\s*(?:बेटा|बेटी|पोता|पोती|भाई|बहन|भतीजा|भांजा)|"
        r"(?:आपका|तुम्हारा)\s*(?:बेटा|पोता|भतीजा)\s*बोल\s*रहा)",
        # Bengali institutional / relative claim. "আমি <name> বলছি" (I am X speaking)
        # is the canonical relative-impersonation opener.
        r"আমি\s+[^\s,।]{2,20}\s+বলছি",
        r"(?:আমি|আমরা)\s*[^।.!?]{0,30}?(?:ব্যাংক|ব্যাঙ্ক|আরবিআই|পুলিশ|সিবিআই|কাস্টমস|"
        r"সাইবার\s*সেল|কাস্টমার\s*কেয়ার|হেড\s*অফিস|ব্রাঞ্চ\s*ম্যানেজার)",
        r"(?:তোমার|আপনার)\s*(?:নাতি|নাতনি|ছেলে|মেয়ে|ভাই|বোন|ভাইপো|ভাগ্নে)",
    ),
    URGENCY: (
        r"\b(?:right now|immediately|at once|urgent|urgently|emergency|as soon as possible|"
        r"within\s+\d+\s*(?:minute|minutes|hour|hours)|before\s+(?:midnight|tonight|the deadline)|"
        r"last warning|final notice|time is running out|expires? (?:today|tonight|in))\b",
        r"\b(?:accident|hospital|icu|operation|surgery|admitted|critical condition|"
        r"arrest(?:ed|ing)?|arrest warrant|warrant|jail|custody|court case|fir|"
        r"blocked|block(?:ing)? your|suspended|deactivat(?:e|ed|ing)|frozen|freeze|"
        r"illegal (?:debit|transaction)|money laundering|drugs (?:were )?found)\b",
        r"(?:तुरंत|अभी\s*ही|फ़ौरन|जल्दी\s*कर|आपातकाल|इमरजेंसी|दुर्घटना|एक्सीडेंट|अस्पताल|"
        r"गिरफ्तार|गिरफ्तारी|वॉरंट|वारंट|जेल|हिरासत|ब्लॉक\s*हो|बंद\s*हो\s*जाएगा|"
        r"स्थगित|अवैध\s*निकासी|सील\s*हो)",
        r"(?:অবিলম্বে|এখনই|তাড়াতাড়ি|জরুরি|দুর্ঘটনা|অ্যাক্সিডেন্ট|হাসপাতাল|আইসিইউ|"
        r"গ্রেফতার|গ্রেপ্তার|ওয়ারেন্ট|জেল|হাজত|বন্ধ\s*হবে|বন্ধ\s*হয়ে\s*যাবে|স্থগিত|"
        r"ব্লক\s*হবে|অবৈধ\s*লেনদেন)",
        # Romanised Hindi/Bengali, for the same ASR reason as the demand verbs.
        r"\b(?:turant|turent|abhi\s*hi|foran|fauran|jaldi(?:\s*karo)?|"
        r"giraftar|giraftari|hirasat|jail\s*ho|durghatna|haspatal|aspatal|"
        r"khata\s*(?:band|block)|band\s*ho\s*jayega|block\s*ho\s*jayega|"
        r"avaidh|galat\s*nikasi|cyber\s*crime|chalan)\b",
        r"\b(?:taratari|ekhuni|obilombe|obiloms|joruri|durghotona|hashpatal|"
        r"bondho\s*hobe|grephtar|greptar|sthogit)\b",
    ),
    # CREDENTIAL entries are *nouns*; a demand verb must appear nearby.
    # Enforced in _match_credential, not by the regex itself.
    CREDENTIAL: (
        r"\b(?:otp|o\.t\.p|one[- ]time password|cvv|c\.v\.v|atm pin|upi pin|mpin|"
        r"net ?banking password|card number|debit card number|credit card number|"
        r"account number|ifsc|aadhaar number|pan number|verification code|security code|"
        r"upi id|qr code|gift card|google play code)\b",
        r"\b(?:pin|password|passcode)\b",
        r"(?:ओटीपी|सीवीवी|एटीएम\s*पिन|यूपीआई\s*पिन|पिन|पासवर्ड|कार्ड\s*नंबर|"
        r"खाता\s*संख्या|अकाउंट\s*नंबर|आधार\s*नंबर|पैन\s*नंबर|वेरिफिकेशन\s*कोड|क्यूआर)",
        r"(?:ওটিপি|সিভিভি|এটিএম\s*পিন|ইউপিআই\s*পিন|পিন|পাসওয়ার্ড|কার্ড\s*নম্বর|"
        r"অ্যাকাউন্ট\s*নম্বর|একাউন্ট\s*নম্বর|আধার\s*নম্বর|প্যান\s*নম্বর|ভেরিফিকেশন\s*কোড|কিউআর)",
        # Romanised identity documents the ASR renders in Latin script.
        r"\b(?:pan\s*(?:card|kard)|aadhaar|aadhar|adhaar|adhar|"
        r"khata\s*(?:number|sankhya)|atm\s*ka\s*pin)\b",
    ),
    SECRECY: (
        r"\b(?:do not|don'?t|never)\s+(?:tell|inform|mention|say anything to|discuss (?:this )?with)\b"
        r"[^.!?]{0,40}?\b(?:anyone|anybody|mother|mom|mum|father|dad|wife|husband|family|son|daughter|police|bank)\b",
        r"\b(?:keep (?:this|it) (?:a )?secret|between (?:us|you and me)|strictly confidential|"
        r"no one should know|nobody should know|do not disclose to anyone)\b",
        r"(?:किसी\s*को\s*(?:कुछ\s*)?(?:मत|ना|नहीं)\s*(?:बताना|बताओ|बताइए|कहना)|"
        r"(?:मम्मी|पापा|माँ|पिताजी|घर\s*में|परिवार\s*को)\s*(?:कुछ\s*)?(?:मत|ना)\s*(?:बताना|बताओ|कहना)|"
        r"गुप्त\s*रख|राज़\s*रख|हमारे\s*बीच\s*(?:ही\s*)?रहे)",
        r"(?:কাউকে\s*(?:কিছু\s*)?বল(?:ো|বেন|বে|বেননা)?\s*না|"
        r"(?:মা|বাবা|মা-?কে|বাবা-?কে|বাড়িতে|পরিবারকে)\s*(?:কিছু\s*)?বল(?:ো|বেন)?\s*না|"
        r"গোপন\s*রাখ|গোপনে\s*রাখ|আমাদের\s*মধ্যে\s*(?:ই\s*)?থাক)",
        # Romanised Hindi/Bengali secrecy instructions.
        r"\b(?:kisi\s*ko\s*(?:kuch\s*)?(?:mat|na|nahi)\s*(?:batana|batao|bataiye|kehna|bolna)|"
        r"(?:mummy|papa|maa|ghar\s*(?:me|mein)|parivar)\s*(?:ko\s*)?(?:kuch\s*)?(?:mat|na)\s*(?:batana|batao|bolna)|"
        r"gupt\s*rakh\w*|raaz\s*rakh\w*|hamare\s*bich\s*(?:hi\s*)?rahe)\b",
        r"\b(?:kauke\s*(?:kichu\s*)?bolo\s*na|maa?-?ke\s*(?:kichu\s*)?bolo\s*na|"
        r"gopon\s*rakh\w*|gopone\s*rakh\w*)\b",
    ),
    REMOTE: (
        r"\b(?:anydesk|any desk|teamviewer|team viewer|quick ?support|rustdesk|ultraviewer|"
        r"screen ?share|share your screen|mirror your screen|remote access|remote control|"
        r"desktop sharing)\b",
        r"\b(?:install|download|side ?load)\b[^.!?]{0,40}?\b(?:app|apk|application|software|tool)\b",
        r"(?:एनीडेस्क|टीमव्यूअर|स्क्रीन\s*शेयर|स्क्रीन\s*साझा|रिमोट\s*एक्सेस|"
        r"ऐप\s*(?:डाउनलोड|इंस्टॉल)|एप्लिकेशन\s*डाउनलोड|एपीके)",
        r"(?:এনিডেস্ক|টিমভিউয়ার|স্ক্রিন\s*শেয়ার|রিমোট\s*অ্যাক্সেস|"
        r"অ্যাপ\s*(?:ডাউনলোড|ইনস্টল)|এপিকে)",
    ),
}

# A concrete money amount. Used both as CREDENTIAL evidence (when demanded) and
# to flag a payment attempt for the circuit breaker.
AMOUNT_PATTERNS: Tuple[str, ...] = (
    # The multiplier word is read from the text *after* the match rather than
    # captured, so the reported span stays tight around the figure itself and
    # the UI highlight does not trail whitespace.
    r"(?:₹|rs\.?|inr|rupees?)\s*([\d][\d,]*(?:\.\d+)?)",
    r"([\d][\d,]*(?:\.\d+)?)\s*(?:lakhs?|lacs?|crores?|thousand)\s*(?:rupees?|₹|टके|টাকা)?",
    r"([\d][\d,]*(?:\.\d+)?)\s*(?:रुपये|रुपए|টাকা|টাকার)",
)

MULTIPLIERS = (
    ("crore", 10_000_000),
    ("crores", 10_000_000),
    ("lakh", 100_000),
    ("lakhs", 100_000),
    ("lac", 100_000),
    ("lacs", 100_000),
    ("thousand", 1_000),
)

_COMPILED_STAGES: Dict[str, Tuple[re.Pattern, ...]] = {
    stage: tuple(re.compile(p, re.IGNORECASE | re.UNICODE) for p in patterns)
    for stage, patterns in STAGE_PATTERNS.items()
}
_COMPILED_AWARENESS: Tuple[re.Pattern, ...] = tuple(
    re.compile(p, re.IGNORECASE | re.UNICODE) for p in AWARENESS_PATTERNS
)
_COMPILED_DEMANDS: Tuple[re.Pattern, ...] = tuple(
    re.compile(p, re.IGNORECASE | re.UNICODE) for p in DEMAND_VERBS
)
_COMPILED_AMOUNTS: Tuple[re.Pattern, ...] = tuple(
    re.compile(p, re.IGNORECASE | re.UNICODE) for p in AMOUNT_PATTERNS
)

# Credentials no legitimate caller ever needs. A demand for one of these is
# qualitatively worse than a request for, say, an account number, which a
# genuine bank might reasonably confirm — so only these arm the
# CREDENTIAL_DEMAND_UNDER_PRESSURE hard rule below.
HIGH_SENSITIVITY_CREDENTIALS = re.compile(
    r"(?:\botp\b|\bo\.t\.p\b|one[- ]time password|\bcvv\b|\bc\.v\.v\b|\bpin\b|\bmpin\b|"
    r"\bpassword\b|\bpasscode\b|card number|verification code|security code|"
    r"ओटीपी|सीवीवी|पिन|पासवर्ड|कार्ड\s*नंबर|वेरिफिकेशन\s*कोड|"
    r"ওটিপি|সিভিভি|পিন|পাসওয়ার্ড|কার্ড\s*নম্বর|ভেরিফিকেশন\s*কোড)",
    re.IGNORECASE | re.UNICODE,
)


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class StageMatch:
    """One phrase in the transcript that fired one stage."""

    stage: str
    start: int
    end: int
    text: str

    def as_dict(self, lang: str = DEFAULT_LANG) -> dict:
        return {
            "stage": self.stage,
            "label": stage_label(self.stage, lang),
            "start": self.start,
            "end": self.end,
            "text": self.text,
        }


@dataclass
class StageAnalysis:
    """Everything the stage pillar concluded about one transcript."""

    stages: List[str] = field(default_factory=list)
    matches: List[StageMatch] = field(default_factory=list)
    awareness_spans: List[Tuple[int, int]] = field(default_factory=list)
    is_awareness_advice: bool = False
    payment_attempt: bool = False
    amount_inr: Optional[float] = None
    high_sensitivity_credential: bool = False

    def as_dict(self, lang: str = DEFAULT_LANG) -> dict:
        lang = normalize_lang(lang)
        return {
            "stages": list(self.stages),
            "stage_count": len(self.stages),
            "stage_total": len(STAGE_ORDER),
            "matches": [m.as_dict(lang) for m in self.matches],
            "labels": stage_labels(lang),
            "explanations": [stage_explanation(s, lang) for s in self.stages],
            "lang": lang,
            "is_awareness_advice": self.is_awareness_advice,
            "payment_attempt": self.payment_attempt,
            "amount_inr": self.amount_inr,
            "high_sensitivity_credential": self.high_sensitivity_credential,
        }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _overlaps(start: int, end: int, spans: Sequence[Tuple[int, int]]) -> bool:
    return any(start < s_end and end > s_start for s_start, s_end in spans)


def _find_awareness_spans(text: str) -> List[Tuple[int, int]]:
    spans: List[Tuple[int, int]] = []
    for pat in _COMPILED_AWARENESS:
        for m in pat.finditer(text):
            spans.append((m.start(), m.end()))
    return _merge_spans(spans)


def _merge_spans(spans: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
    if not spans:
        return []
    spans = sorted(spans)
    merged = [spans[0]]
    for start, end in spans[1:]:
        last_start, last_end = merged[-1]
        if start <= last_end:
            merged[-1] = (last_start, max(last_end, end))
        else:
            merged.append((start, end))
    return merged


def _demand_spans(text: str) -> List[Tuple[int, int]]:
    spans: List[Tuple[int, int]] = []
    for pat in _COMPILED_DEMANDS:
        for m in pat.finditer(text):
            spans.append((m.start(), m.end()))
    return _merge_spans(spans)


def _near_demand(start: int, end: int, demands: Sequence[Tuple[int, int]]) -> bool:
    """True when a demand verb sits within DEMAND_PROXIMITY_CHARS of the noun."""
    for d_start, d_end in demands:
        # Distance between the nearest edges of the two spans; 0 if they overlap.
        gap = max(0, start - d_end, d_start - end)
        if gap <= DEMAND_PROXIMITY_CHARS:
            return True
    return False


def _parse_amount(raw: str, tail: str) -> Optional[float]:
    try:
        value = float(raw.replace(",", ""))
    except (TypeError, ValueError):
        return None
    tail_lower = tail.lower()
    for token, factor in MULTIPLIERS:
        if token in tail_lower:
            return value * factor
    return value


def _find_amounts(text: str, exclude: Sequence[Tuple[int, int]]) -> List[Tuple[int, int, float]]:
    found: List[Tuple[int, int, float]] = []
    for pat in _COMPILED_AMOUNTS:
        for m in pat.finditer(text):
            if _overlaps(m.start(), m.end(), exclude):
                continue
            amount = _parse_amount(m.group(1), text[m.end(): m.end() + 12])
            if amount is not None and amount > 0:
                found.append((m.start(), m.end(), amount))
    return found


# ---------------------------------------------------------------------------
# Stage classification
# ---------------------------------------------------------------------------


def classify_stages(transcript: Optional[str]) -> StageAnalysis:
    """
    Decomposes a transcript into vishing playbook stages.

    Returns an empty analysis for empty input. Awareness advice is detected
    first and its spans are excluded from every stage, so a bank's own
    "never share your OTP" recording scores zero rather than maximum.
    """
    analysis = StageAnalysis()
    if not transcript or not transcript.strip():
        return analysis

    text = transcript
    awareness = _find_awareness_spans(text)
    analysis.awareness_spans = awareness
    analysis.is_awareness_advice = bool(awareness)

    demands = [d for d in _demand_spans(text) if not _overlaps(d[0], d[1], awareness)]

    matches: List[StageMatch] = []

    for stage in STAGE_ORDER:
        for pat in _COMPILED_STAGES[stage]:
            for m in pat.finditer(text):
                start, end = m.start(), m.end()
                if _overlaps(start, end, awareness):
                    continue
                if stage == CREDENTIAL and not _near_demand(start, end, demands):
                    # A sensitive noun with no demand attached is a mention,
                    # not an attack. "I never got the OTP" must not fire.
                    continue
                matches.append(StageMatch(stage, start, end, text[start:end]))

    # A demanded money amount is credential-grade evidence in its own right:
    # "send ₹50,000 to this UPI ID" contains no OTP or PIN token at all.
    amounts = _find_amounts(text, awareness)
    for start, end, amount in amounts:
        if _near_demand(start, end, demands):
            matches.append(StageMatch(CREDENTIAL, start, end, text[start:end]))
            if analysis.amount_inr is None or amount > analysis.amount_inr:
                analysis.amount_inr = amount
            analysis.payment_attempt = True

    analysis.matches = _dedupe_matches(matches)
    present = {m.stage for m in analysis.matches}
    analysis.stages = [s for s in STAGE_ORDER if s in present]
    analysis.high_sensitivity_credential = any(
        m.stage == CREDENTIAL and HIGH_SENSITIVITY_CREDENTIALS.search(m.text)
        for m in analysis.matches
    )
    return analysis


def _dedupe_matches(matches: List[StageMatch]) -> List[StageMatch]:
    """
    Keeps the longest match per (stage, overlapping region) and sorts by
    position, so the UI gets non-overlapping highlight spans per stage.
    """
    by_stage: Dict[str, List[StageMatch]] = {}
    for m in matches:
        by_stage.setdefault(m.stage, []).append(m)

    kept: List[StageMatch] = []
    for stage, items in by_stage.items():
        items.sort(key=lambda m: (m.start, -(m.end - m.start)))
        stage_kept: List[StageMatch] = []
        for m in items:
            if any(m.start < k.end and m.end > k.start for k in stage_kept):
                continue
            stage_kept.append(m)
        kept.extend(stage_kept)

    kept.sort(key=lambda m: (m.start, m.stage))
    return kept


# ---------------------------------------------------------------------------
# Risk fusion
# ---------------------------------------------------------------------------


def fuse_risk(
    stages: Sequence[str],
    acoustic_prob: float = 0.0,
    payment_attempt: bool = False,
    blacklisted: bool = False,
    voip_spoofed: bool = False,
    credential_high_sensitivity: bool = False,
) -> dict:
    """
    Combines the stage evidence with the acoustic pillar into one 0-100 score.

    Noisy-OR rather than a weighted sum: each independent piece of evidence
    multiplicatively erodes the probability that the call is legitimate, so
    three weak signals can convict where no single one would, and no signal
    can single-handedly saturate the score. A weighted sum, by contrast, lets
    one loud pillar drown the others — which is how the existing threshold
    ladder in `vanishield_engine` produces its 99% acoustic false positives.

    `acoustic_prob` is the synthetic-voice probability in 0..1 (not percent).
    """
    acoustic_prob = _clamp(float(acoustic_prob or 0.0), 0.0, 1.0)

    if blacklisted:
        return {
            "risk_score": 99.0,
            "band": BAND_RED,
            "legitimate_prob": 1.0,
            "contributions": {"blacklist": 99.0},
            "hard_rule": "KNOWN_SCAMMER_NUMBER",
            "stages": list(stages),
            "acoustic_prob": round(acoustic_prob, 4),
        }

    # Prior: even a call with no detected evidence is not certified safe, so
    # legitimacy starts at 0.9 rather than 1.0.
    legitimate = 0.9
    contributions: Dict[str, float] = {}

    seen = []
    for stage in stages:
        if stage in STAGE_WEIGHTS and stage not in seen:
            seen.append(stage)
            weight = STAGE_WEIGHTS[stage]
            before = legitimate
            legitimate *= 1.0 - weight
            contributions[stage] = round((before - legitimate) * 100.0, 1)

    if acoustic_prob > 0:
        before = legitimate
        legitimate *= 1.0 - 0.35 * acoustic_prob
        contributions["acoustic"] = round((before - legitimate) * 100.0, 1)

    if payment_attempt:
        before = legitimate
        legitimate *= 0.8 * 0.85
        contributions["payment_attempt"] = round((before - legitimate) * 100.0, 1)

    risk = (1.0 - legitimate) * 100.0
    hard_rule = None

    # Hard rule: a demand for an OTP, PIN, CVV or password under manufactured
    # time pressure is the entire phishing playbook in two moves, and convicts
    # on the transcript alone. Without this floor a human fraudster — no
    # synthetic voice to detect — tops out in amber, which is exactly the
    # attack that most often succeeds against the people this is built for.
    if (
        CREDENTIAL in seen
        and URGENCY in seen
        and credential_high_sensitivity
        and risk < 70.0
    ):
        risk = 70.0
        hard_rule = "CREDENTIAL_DEMAND_UNDER_PRESSURE"

    # A credential demand in a synthetic voice is the signature of a
    # cloned-relative attack; it outranks the rule above and floors at 80.
    if CREDENTIAL in seen and acoustic_prob >= ACOUSTIC_SYNTHETIC_THRESHOLD:
        if risk < 80.0:
            risk = 80.0
        hard_rule = "CREDENTIAL_DEMAND_IN_SYNTHETIC_VOICE"

    if voip_spoofed and risk > AMBER_THRESHOLD:
        risk = min(100.0, risk + 10.0)

    risk = round(_clamp(risk, 0.0, 100.0), 1)

    return {
        "risk_score": risk,
        "band": band_for(risk),
        "legitimate_prob": round(legitimate, 4),
        "contributions": contributions,
        "hard_rule": hard_rule,
        "stages": seen,
        "acoustic_prob": round(acoustic_prob, 4),
    }


def band_for(risk_score: float) -> str:
    if risk_score >= RED_THRESHOLD:
        return BAND_RED
    if risk_score >= AMBER_THRESHOLD:
        return BAND_AMBER
    return BAND_GREEN


def band_label(band: str, lang: str = DEFAULT_LANG) -> str:
    return BAND_LABELS_BY_LANG[normalize_lang(lang)].get(band, band)


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


# ---------------------------------------------------------------------------
# Plain-language explanation
# ---------------------------------------------------------------------------


def explain(analysis: StageAnalysis, fusion: dict, lang: str = DEFAULT_LANG) -> str:
    """
    One sentence a non-technical person can act on, in the language the caller
    is speaking. This is what the call screen shows and what a spoken warning
    would read out, so it names the behaviour observed rather than any model,
    pillar or score.

    Stage clauses are chained with the language's own connective, which is why
    the clause tables hold bare verb phrases: Hindi and Bengali are verb-final,
    so a list of complete sentences cannot simply be comma-joined the way the
    English one can.
    """
    lang = normalize_lang(lang)
    strings = VERDICT_STRINGS[lang]
    synthetic = fusion.get("acoustic_prob", 0.0) >= ACOUSTIC_SYNTHETIC_THRESHOLD

    if fusion.get("hard_rule") == "KNOWN_SCAMMER_NUMBER":
        return strings["blacklisted"]

    if not analysis.stages:
        if synthetic:
            return strings["synthetic_only"]
        if analysis.is_awareness_advice:
            return strings["awareness"]
        return strings["nothing"]

    body = strings["then"].join(stage_clause(s, lang) for s in analysis.stages)
    subject = strings["synthetic_prefix"] if synthetic else strings["subject"]
    sentence = f"{subject} {body}{strings['terminator']}"

    # Only Latin script has a case distinction to fix up.
    if lang == "en" and sentence:
        sentence = sentence[0].upper() + sentence[1:]
    return sentence
