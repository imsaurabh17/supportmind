import logging
from typing import Literal
from functools import lru_cache


logger = logging.getLogger(__name__)

sentimentType = Literal['frustrated', 'neutral', 'happy']

@lru_cache(maxsize=1)
def _get_classifier():
    """Load sentiment model once, lazily, on first use."""
    from transformers import pipeline
    return pipeline(
        "sentiment-analysis",
        model="distilbert-base-uncased-finetuned-sst-2-english"
    )

def detect_sentiment(text: str) -> sentimentType:
    """Classify user sentiment from message text.

    Uses TextBlob polarity score:
        < -0.1 -> frustrated
        -0.1 to 0.3 -> neutral
        > 0.3 -> happy

    Args:
        text: The user's raw message.

    Returns:
        Sentiment label as string literal.
    """
    classifier = _get_classifier()
    result = classifier(text)[0]

    if result['score'] < 0.75:
        return 'neutral'
    if result['label'] == "NEGATIVE":
        return 'frustrated'
    return 'happy'

SENTIMENT_EMOJI = {
    'frustrated': '😤',
    'neutral':    '😐',
    'happy':      '😊',
}

SENTIMENT_COLOR = {
    'frustrated': '#F78166',
    'neutral':    '#8B949E',
    'happy':      '#3FB950',
}

if __name__=="__main__":
    print(detect_sentiment("I want my refund as soon as possible, service isn't expected"))