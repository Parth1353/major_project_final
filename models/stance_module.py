"""Optional stance analysis for WhatsApp replies/comments.

The detector can work without comments. When comments are available, this module
estimates whether the reply stream disagrees with the forwarded claim. More
disagreement increases the final fake-news score.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable


STANCE_LABELS = ["AGREE", "DISAGREE", "DISCUSS", "UNRELATED"]

AGREE_WORDS = {
    "true",
    "correct",
    "yes",
    "सच",
    "सही",
    "हाँ",
    "ठीक",
    "ਸੱਚ",
    "ਹਾਂ",
    "ਠੀਕ",
}
DISAGREE_WORDS = {
    "fake",
    "false",
    "wrong",
    "not true",
    "झूठ",
    "फेक",
    "गलत",
    "सच नहीं",
    "ਝੂਠ",
    "ਫੇਕ",
    "ਗਲਤ",
}
DISCUSS_WORDS = {
    "source",
    "proof",
    "link",
    "कहाँ",
    "स्रोत",
    "सबूत",
    "ਪ੍ਰਮਾਣ",
    "ਸਰੋਤ",
    "ਕਿੱਥੇ",
}


@dataclass
class StanceResult:
    stance_score: float
    comment_weight: float
    stance_counts: dict[str, int]
    used_model: str


class StanceAnalyzer:
    """Classify comment stance with NLI if available, otherwise heuristics."""

    def __init__(self, model_name: str = "cross-encoder/nli-MiniLM2-L6-H768", enable_model: bool = True) -> None:
        self.model_name = model_name
        self.pipeline = None
        self.used_model = "heuristic"
        if enable_model:
            self._try_load_pipeline()

    def _try_load_pipeline(self) -> None:
        try:
            from transformers import pipeline

            self.pipeline = pipeline("zero-shot-classification", model=self.model_name)
            self.used_model = self.model_name
        except Exception:
            self.pipeline = None
            self.used_model = "heuristic"

    def analyze(self, claim_text: str, comments: Iterable[str] | None) -> StanceResult:
        comments = [str(comment).strip() for comment in (comments or []) if str(comment).strip()]
        counts = {label: 0 for label in STANCE_LABELS}
        if not comments:
            return StanceResult(stance_score=0.0, comment_weight=0.0, stance_counts=counts, used_model=self.used_model)

        for comment in comments:
            stance = self.classify_comment(claim_text, comment)
            counts[stance] += 1

        total = max(1, len(comments))
        disagreement = counts["DISAGREE"] / total
        discussion = counts["DISCUSS"] / total
        agreement = counts["AGREE"] / total
        raw_score = max(0.0, min(1.0, disagreement + 0.25 * discussion - 0.15 * agreement))
        comment_weight = 0.5 if len(comments) > 100 else 1.0
        return StanceResult(
            stance_score=raw_score * comment_weight,
            comment_weight=comment_weight,
            stance_counts=counts,
            used_model=self.used_model,
        )

    def classify_comment(self, claim_text: str, comment: str) -> str:
        if self.pipeline is not None:
            try:
                hypothesis_labels = [
                    "agrees with the claim",
                    "disagrees with the claim",
                    "discusses the claim",
                    "is unrelated to the claim",
                ]
                result = self.pipeline(
                    f"Claim: {claim_text}\nComment: {comment}",
                    candidate_labels=hypothesis_labels,
                    multi_label=False,
                )
                best = result["labels"][0]
                return {
                    "agrees with the claim": "AGREE",
                    "disagrees with the claim": "DISAGREE",
                    "discusses the claim": "DISCUSS",
                    "is unrelated to the claim": "UNRELATED",
                }.get(best, "DISCUSS")
            except Exception:
                return self._heuristic_stance(comment)
        return self._heuristic_stance(comment)

    def _heuristic_stance(self, comment: str) -> str:
        lower = comment.lower()
        if any(word in lower for word in DISAGREE_WORDS):
            return "DISAGREE"
        if any(word in lower for word in AGREE_WORDS):
            return "AGREE"
        if any(word in lower for word in DISCUSS_WORDS) or "?" in lower:
            return "DISCUSS"
        return "UNRELATED"


def combine_model_and_stance_scores(model_score: float, stance_score: float) -> float:
    return max(0.0, min(1.0, 0.7 * float(model_score) + 0.3 * float(stance_score)))

