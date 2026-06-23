"""
tools/sentiment.py
Financial sentiment analysis using FinBERT (ProsusAI/finbert).
Scores a list of headlines as positive / negative / neutral with confidence.
"""

import logging
from functools import lru_cache
from typing import Optional

logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def _load_pipeline():
    """Load FinBERT once and cache it — model is ~400MB."""
    from transformers import pipeline
    from config import config

    logger.info("Loading FinBERT sentiment model (first run may take a moment)...")
    return pipeline(
        "text-classification",
        model=config.SENTIMENT_MODEL,
        top_k=None,  # return all labels
    )


def score_headlines(headlines: list[str]) -> dict:
    """
    Run FinBERT over a list of headline strings.
    Returns aggregated sentiment scores and an overall label.
    """
    if not headlines:
        return {
            "label": "neutral",
            "positive": 0.0,
            "negative": 0.0,
            "neutral": 1.0,
            "confidence": 0.0,
            "article_count": 0,
        }

    try:
        pipe = _load_pipeline()
        totals = {"positive": 0.0, "negative": 0.0, "neutral": 0.0}

        for headline in headlines:
            # Truncate to 512 tokens (FinBERT limit)
            results = pipe(headline[:512])
            for item in results[0]:
                label = item["label"].lower()
                totals[label] = totals.get(label, 0.0) + item["score"]

        count = len(headlines)
        avg = {k: v / count for k, v in totals.items()}

        # Dominant label
        dominant = max(avg, key=avg.get)  # type: ignore[arg-type]

        return {
            "label": dominant,
            "positive": round(avg["positive"], 4),
            "negative": round(avg["negative"], 4),
            "neutral": round(avg["neutral"], 4),
            "confidence": round(avg[dominant], 4),
            "article_count": count,
        }

    except Exception as e:
        logger.error(f"Sentiment scoring failed: {e}")
        return {
            "label": "neutral",
            "positive": 0.0,
            "negative": 0.0,
            "neutral": 1.0,
            "confidence": 0.0,
            "article_count": 0,
        }


def sentiment_summary(sentiment: dict) -> str:
    """Format sentiment dict as a readable string for LLM prompts."""
    return (
        f"Sentiment: {sentiment['label'].upper()} "
        f"(pos={sentiment['positive']:.0%}, "
        f"neg={sentiment['negative']:.0%}, "
        f"neu={sentiment['neutral']:.0%}) "
        f"across {sentiment['article_count']} articles"
    )
