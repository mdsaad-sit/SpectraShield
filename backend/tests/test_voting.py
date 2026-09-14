import sys
import numpy as np
sys.path.insert(0, '/c/Users/Bharath PR/Major-Deepfake/backend')

from app.inference.prediction import classify_chunk, majority_vote


def test_classify_threshold_029_real():
    """fake_probability = 0.29 => REAL"""
    pred, _ = classify_chunk(0.29)
    assert pred == "REAL", f"Expected REAL for 0.29, got {pred}"


def test_classify_threshold_030_fake():
    """fake_probability = 0.30 => FAKE"""
    pred, _ = classify_chunk(0.30)
    assert pred == "FAKE", f"Expected FAKE for 0.30, got {pred}"


def test_classify_below_threshold_real():
    """fake_probability = 0.25 => REAL"""
    pred, _ = classify_chunk(0.25)
    assert pred == "REAL", f"Expected REAL for 0.25, got {pred}"


def test_classify_above_threshold_fake():
    """fake_probability = 0.35 => FAKE"""
    pred, _ = classify_chunk(0.35)
    assert pred == "FAKE", f"Expected FAKE for 0.35, got {pred}"


def test_majority_real():
    """ Majority of chunks are REAL => final prediction REAL """
    predictions = ["REAL", "REAL", "REAL", "FAKE", "REAL"]
    probs = [0.1, 0.2, 0.15, 0.85, 0.18]
    final, confidence, real_count, fake_count, mean_fake = majority_vote(predictions, probs)
    assert final == "REAL", f"Expected REAL, got {final}"
    assert real_count == 4, f"Expected 4 real chunks, got {real_count}"
    assert fake_count == 1, f"Expected 1 fake chunk, got {fake_count}"


def test_majority_fake():
    """ Majority of chunks are FAKE => final prediction FAKE """
    predictions = ["FAKE", "FAKE", "FAKE", "REAL", "FAKE"]
    probs = [0.9, 0.85, 0.95, 0.3, 0.8]
    final, confidence, real_count, fake_count, mean_fake = majority_vote(predictions, probs)
    assert final == "FAKE", f"Expected FAKE, got {final}"
    assert fake_count == 4, f"Expected 4 fake chunks, got {fake_count}"
    assert real_count == 1, f"Expected 1 real chunk, got {real_count}"


def test_tie_breaking_fake():
    """Tie-breaking: mean fake prob >= 0.30 => FAKE"""
    predictions = ["REAL", "FAKE", "REAL", "FAKE", "REAL"]  # 3 REAL, 2 FAKE - that's not a tie
    # Let's make a real tie: 3 each
    predictions = ["REAL", "FAKE", "REAL", "FAKE", "REAL", "FAKE"]  # 3 each - but we have 6 chunks
    # Actually the spec says 5 chunks, so tie is 2.5 each which isn't possible
    # Let's test with the tie-breaking rule directly
    predictions = ["REAL", "FAKE"]  # 1 each - tie
    probs = [0.2, 0.8]  # mean fake = 0.5 >= 0.30
    final, confidence, real_count, fake_count, mean_fake = majority_vote(predictions, probs)
    assert final == "FAKE", f"Expected FAKE with mean fake prob 0.5 >= 0.30, got {final}"


def test_tie_breaking_real():
    """Tie-breaking: mean fake prob < 0.30 => REAL"""
    predictions = ["REAL", "FAKE"]  # 1 each - tie
    probs = [0.2, 0.1]  # mean fake = 0.15 < 0.30
    final, confidence, real_count, fake_count, mean_fake = majority_vote(predictions, probs)
    assert final == "REAL", f"Expected REAL with mean fake prob 0.15 < 0.30, got {final}"
    assert mean_fake < 0.30, f"Expected mean_fake < 0.30, got {mean_fake}"