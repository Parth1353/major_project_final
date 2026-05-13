"""Shared preprocessing and lightweight inference helpers.

The functions in this module mirror the Kaggle training notebook so that local
API inference uses the same text cleaning and metadata features.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Any


FORWARDED_RE = re.compile(r"^\s*Forwarded\s+many\s+times\s*", flags=re.IGNORECASE)
URL_RE = re.compile(r"https?://\S+|www\.\S+", flags=re.IGNORECASE)
HANDLE_RE = re.compile(r"(?<!\w)@[A-Za-z0-9_]{2,}")
WHITESPACE_RE = re.compile(r"\s+")
EMOJI_MODIFIER_RE = re.compile(r"[\uFE0E\uFE0F\u200D]")
GURMUKHI_RE = re.compile(r"[\u0A00-\u0A7F]")
DEVANAGARI_RE = re.compile(r"[\u0900-\u097F]")
EMOJI_RE = re.compile(
    "["
    "\U0001F1E6-\U0001F1FF"
    "\U0001F300-\U0001F5FF"
    "\U0001F600-\U0001F64F"
    "\U0001F680-\U0001F6FF"
    "\U0001F700-\U0001F77F"
    "\U0001F780-\U0001F7FF"
    "\U0001F800-\U0001F8FF"
    "\U0001F900-\U0001F9FF"
    "\U0001FA70-\U0001FAFF"
    "\u2600-\u26FF"
    "\u2700-\u27BF"
    "]+",
    flags=re.UNICODE,
)

SHARE_KEYWORDS = [
    "जरूर शेयर",
    "शेयर करें",
    "आगे भेज",
    "फॉरवर्ड",
    "सबको भेज",
    "सभी को भेज",
    "ਜ਼ਰੂਰ ਸ਼ੇਅਰ",
    "ਸ਼ੇਅਰ ਕਰੋ",
    "ਅੱਗੇ ਭੇਜ",
    "ਸਭ ਨੂੰ ਭੇਜ",
    "share",
    "forward",
    "send everyone",
    "must share",
]
SOURCE_KEYWORDS = [
    "source:",
    "ani",
    "pti",
    "reuters",
    "bbc",
    "the hindu",
    "indian express",
    "news18",
    "ndtv",
    "स्रोत",
    "ਸਰੋਤ",
]
URGENCY_KEYWORDS = [
    "तुरंत",
    "जरूरी",
    "अभी",
    "आज ही",
    "चेतावनी",
    "ਫੌਰਨ",
    "ਜ਼ਰੂਰੀ",
    "ਹੁਣੇ",
    "warning",
    "urgent",
    "immediately",
]
POSITIVE_WORDS = {
    "सही",
    "सच",
    "अच्छा",
    "लाभ",
    "मदद",
    "जीत",
    "मुफ्त",
    "फ्री",
    "खुश",
    "बधाई",
    "ਸੱਚ",
    "ਚੰਗਾ",
    "ਲਾਭ",
    "ਮਦਦ",
    "ਜਿੱਤ",
    "ਮੁਫਤ",
    "ਵਧਾਈ",
    "true",
    "good",
    "benefit",
    "free",
    "win",
    "help",
}
NEGATIVE_WORDS = {
    "खतरा",
    "झूठ",
    "फेक",
    "धोखा",
    "बीमारी",
    "मौत",
    "घोटाला",
    "डर",
    "ब्लॉक",
    "चेतावनी",
    "ਖਤਰਾ",
    "ਝੂਠ",
    "ਫੇਕ",
    "ਧੋਖਾ",
    "ਬਿਮਾਰੀ",
    "ਮੌਤ",
    "ਘੋਟਾਲਾ",
    "ਡਰ",
    "ਚੇਤਾਵਨੀ",
    "fake",
    "fraud",
    "danger",
    "scam",
    "death",
    "warning",
    "blocked",
}

METADATA_FEATURES = [
    "emoji_count",
    "exclamation_count",
    "text_length",
    "forwarded_flag",
    "share_words",
    "caps_ratio",
    "sentiment_score",
]


@dataclass(frozen=True)
class PreprocessedText:
    original_text: str
    clean_text: str
    transformer_text: str
    language_detected: str
    model_language: str
    features: dict[str, float]
    red_flags: list[str]


def count_emojis(text: str) -> int:
    return len(EMOJI_RE.findall(str(text)))


def remove_emojis(text: str) -> str:
    without_emoji = EMOJI_RE.sub(" ", str(text))
    return EMOJI_MODIFIER_RE.sub(" ", without_emoji)


def detect_language(text: str) -> str:
    """Return hi, pa, mixed, or unknown from script evidence."""
    text = str(text)
    has_gurmukhi = bool(GURMUKHI_RE.search(text))
    has_devanagari = bool(DEVANAGARI_RE.search(text))
    if has_gurmukhi and has_devanagari:
        return "mixed"
    if has_gurmukhi:
        return "pa"
    if has_devanagari:
        return "hi"
    return "unknown"


def model_language(text: str) -> str:
    """Training-compatible language prefix: any Gurmukhi goes through PA."""
    return "pa" if GURMUKHI_RE.search(str(text)) else "hi"


def transformer_prefix(text: str) -> str:
    return "[PA]" if model_language(text) == "pa" else "[HI]"


def clean_whatsapp_text(text: str) -> str:
    text = str(text)
    text = FORWARDED_RE.sub(" ", text)
    text = URL_RE.sub(" ", text)
    text = HANDLE_RE.sub(" ", text)
    text = remove_emojis(text)
    return WHITESPACE_RE.sub(" ", text).strip()


def caps_ratio(text: str) -> float:
    letters = [char for char in str(text) if char.isalpha()]
    if not letters:
        return 0.0
    return sum(1 for char in letters if char.isupper()) / len(letters)


def count_share_keywords(text: str) -> int:
    lower = str(text).lower()
    return sum(lower.count(keyword.lower()) for keyword in SHARE_KEYWORDS)


def simple_sentiment_score(text: str) -> float:
    tokens = re.findall(r"[\w\u0900-\u097F\u0A00-\u0A7F]+", str(text).lower())
    if not tokens:
        return 0.0
    positive = sum(1 for token in tokens if token in POSITIVE_WORDS)
    negative = sum(1 for token in tokens if token in NEGATIVE_WORDS)
    return (positive - negative) / math.sqrt(len(tokens))


def extract_metadata_features(text: str) -> dict[str, float]:
    original = str(text)
    return {
        "emoji_count": float(count_emojis(original)),
        "exclamation_count": float(original.count("!") + original.count("！")),
        "text_length": float(len(original)),
        "forwarded_flag": float(int("forwarded" in original.lower())),
        "share_words": float(count_share_keywords(original)),
        "caps_ratio": float(caps_ratio(original)),
        "sentiment_score": float(simple_sentiment_score(original)),
    }


def detect_red_flags(text: str) -> list[str]:
    original = str(text)
    lower = original.lower()
    flags: list[str] = []
    if "forwarded" in lower:
        flags.append("forwarded_message")
    if count_share_keywords(original) > 0:
        flags.append("share_urgency_words")
    if any(keyword in lower for keyword in URGENCY_KEYWORDS):
        flags.append("urgent_or_emotional_language")
    if URL_RE.search(original):
        flags.append("external_link")
    if not any(keyword in lower for keyword in SOURCE_KEYWORDS):
        flags.append("no_source")
    if count_emojis(original) >= 2:
        flags.append("multiple_emojis")
    if original.count("!") >= 2:
        flags.append("excessive_punctuation")
    return flags


def preprocess_whatsapp(text: str) -> PreprocessedText:
    original = str(text)
    clean_text = clean_whatsapp_text(original)
    language_detected = detect_language(original)
    training_language = model_language(original)
    transformer_text = f"{transformer_prefix(original)} {clean_text}".strip()
    return PreprocessedText(
        original_text=original,
        clean_text=clean_text,
        transformer_text=transformer_text,
        language_detected=language_detected,
        model_language=training_language,
        features=extract_metadata_features(original),
        red_flags=detect_red_flags(original),
    )


def heuristic_fake_probability(text: str) -> float:
    """Transparent fallback used only when no trained model is available."""
    processed = preprocess_whatsapp(text)
    features = processed.features
    score = 0.28
    score += 0.16 * min(features["forwarded_flag"], 1.0)
    score += 0.08 * min(features["share_words"], 3.0)
    score += 0.05 * min(features["emoji_count"], 4.0)
    score += 0.05 if "no_source" in processed.red_flags else -0.05
    score += 0.08 if "external_link" in processed.red_flags else 0.0
    score += 0.06 if "urgent_or_emotional_language" in processed.red_flags else 0.0
    return max(0.02, min(0.98, score))


def feature_vector(features: dict[str, Any]) -> list[float]:
    return [float(features[name]) for name in METADATA_FEATURES]

