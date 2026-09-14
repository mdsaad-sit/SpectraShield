"""Shared classification and voting utilities.

Used by: tests/test_voting.py

Note: The route modules (routes/recorded.py, routes/live.py) contain
their own inline classification and majority voting functions. This
module provides the canonical implementations used by the test suite.
"""

import numpy as np
import logging
from app.inference.model_loader import ModelLoader

logger = logging.getLogger(__name__)

CLASSIFICATION_THRESHOLD = 0.30


def classify_chunk(fake_probability):
    """Classify a chunk based on fake probability.

    Args:
        fake_probability: float from sigmoid output

    Returns:
        ('FAKE' or 'REAL', fake_probability)
    """
    if fake_probability >= CLASSIFICATION_THRESHOLD:
        return "FAKE", fake_probability
    else:
        return "REAL", fake_probability


def majority_vote(chunk_predictions, chunk_probabilities):
    """Determine final prediction using majority voting with tie-breaking.

    Args:
        chunk_predictions: list of ('FAKE' or 'REAL')
        chunk_probabilities: list of fake probabilities

    Returns:
        (final_prediction, confidence, real_count, fake_count, mean_fake_probability)
    """
    real_count = sum(1 for p in chunk_predictions if p == "REAL")
    fake_count = sum(1 for p in chunk_predictions if p == "FAKE")

    if fake_count > real_count:
        final_prediction = "FAKE"
    elif real_count > fake_count:
        final_prediction = "REAL"
    else:
        # Tie-breaking: use mean fake probability
        mean_fake_prob = np.mean(chunk_probabilities) if chunk_probabilities else 0.0
        if mean_fake_prob >= CLASSIFICATION_THRESHOLD:
            final_prediction = "FAKE"
        else:
            final_prediction = "REAL"

    # Calculate confidence
    if final_prediction == "FAKE":
        confidence = fake_count / len(chunk_predictions) if chunk_predictions else 0.0
    else:
        confidence = real_count / len(chunk_predictions) if chunk_predictions else 0.0

    mean_fake_prob = np.mean(chunk_probabilities) if chunk_probabilities else 0.0

    return final_prediction, confidence, real_count, fake_count, mean_fake_prob