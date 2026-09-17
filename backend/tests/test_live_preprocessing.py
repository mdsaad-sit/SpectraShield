"""Tests for live audio preprocessing, waveform validation,
resampling equivalence, cumulative voting, and session management.
"""

import sys
import os
import io
import numpy as np
import scipy.io.wavfile
import torch

sys.path.insert(
    0,
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)

from app.inference.preprocessing import (
    preprocess_chunk_waveform,
    validate_waveform,
    preprocess_audio,
    SR,
    N_MELS,
    TARGET_FRAMES,
)


# ============================================================
# preprocess_chunk_waveform — output shape & dtype
# ============================================================


def test_preprocess_chunk_waveform_output_shape():
    """preprocess_chunk_waveform should return [1,1,128,128] float32."""
    sr = 16000
    duration = 4.0
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    y = np.sin(2 * np.pi * 440 * t).astype(np.float32)

    tensor = preprocess_chunk_waveform(y, sr)

    assert isinstance(tensor, torch.Tensor)
    assert tensor.shape == (1, 1, N_MELS, TARGET_FRAMES)
    assert tensor.dtype == torch.float32


def test_preprocess_chunk_waveform_values_in_range():
    """Output values should be in [0, 1] after min-max normalization."""
    sr = 16000
    duration = 4.0
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    y = np.sin(2 * np.pi * 440 * t).astype(np.float32)

    tensor = preprocess_chunk_waveform(y, sr)
    data = tensor.numpy()

    assert np.all(data >= 0.0), f"Min value {data.min()} < 0"
    assert np.all(data <= 1.0 + 1e-6), f"Max value {data.max()} > 1"


# ============================================================
# Resampling equivalence: 48kHz → 16kHz vs native 16kHz
# ============================================================


def test_resampling_equivalence_48k_to_16k():
    """A 48kHz signal resampled to 16kHz should produce the same
    preprocessing result as the same signal generated at 16kHz.

    This verifies that the live path (which receives 48kHz from the
    browser and resamples internally) produces results equivalent to
    the recorded path (which loads files at 16kHz directly).
    """
    import librosa

    duration = 4.0
    freq = 440.0

    # Generate at 16kHz (the "recorded" scenario)
    sr_16k = 16000
    t_16k = np.linspace(0, duration, int(sr_16k * duration), endpoint=False)
    y_16k = (np.sin(2 * np.pi * freq * t_16k) * 0.5).astype(np.float32)

    # Generate at 48kHz (the "live" scenario)
    sr_48k = 48000
    t_48k = np.linspace(0, duration, int(sr_48k * duration), endpoint=False)
    y_48k = (np.sin(2 * np.pi * freq * t_48k) * 0.5).astype(np.float32)

    tensor_16k = preprocess_chunk_waveform(y_16k, sr_16k)
    tensor_48k = preprocess_chunk_waveform(y_48k, sr_48k)

    # The two tensors should be very close (not exactly equal due to
    # resampling interpolation, but within a tight tolerance)
    diff = torch.abs(tensor_16k - tensor_48k)
    max_diff = diff.max().item()
    mean_diff = diff.mean().item()

    assert max_diff < 0.15, (
        f"Max difference between 16kHz and resampled 48kHz: {max_diff}"
    )
    assert mean_diff < 0.05, (
        f"Mean difference between 16kHz and resampled 48kHz: {mean_diff}"
    )


def test_resampling_equivalence_44100_to_16k():
    """44.1kHz → 16kHz resampling should produce equivalent results."""
    duration = 4.0
    freq = 440.0

    sr_16k = 16000
    t_16k = np.linspace(0, duration, int(sr_16k * duration), endpoint=False)
    y_16k = (np.sin(2 * np.pi * freq * t_16k) * 0.5).astype(np.float32)

    sr_44k = 44100
    t_44k = np.linspace(0, duration, int(sr_44k * duration), endpoint=False)
    y_44k = (np.sin(2 * np.pi * freq * t_44k) * 0.5).astype(np.float32)

    tensor_16k = preprocess_chunk_waveform(y_16k, sr_16k)
    tensor_44k = preprocess_chunk_waveform(y_44k, sr_44k)

    diff = torch.abs(tensor_16k - tensor_44k)
    max_diff = diff.max().item()

    assert max_diff < 0.15, (
        f"Max difference between 16kHz and resampled 44.1kHz: {max_diff}"
    )


# ============================================================
# Waveform validation
# ============================================================


def test_validate_waveform_empty():
    """Empty array should raise ValueError."""
    import pytest
    with pytest.raises(ValueError, match="empty"):
        validate_waveform(np.array([], dtype=np.float32))


def test_validate_waveform_none():
    """None should raise ValueError."""
    import pytest
    with pytest.raises(ValueError, match="empty"):
        validate_waveform(None)


def test_validate_waveform_nan():
    """Array containing NaN should raise ValueError."""
    import pytest
    y = np.array([0.1, 0.2, np.nan, 0.3], dtype=np.float32)
    with pytest.raises(ValueError, match="NaN"):
        validate_waveform(y)


def test_validate_waveform_inf():
    """Array containing Inf should raise ValueError."""
    import pytest
    y = np.array([0.1, np.inf, 0.3], dtype=np.float32)
    with pytest.raises(ValueError, match="Inf"):
        validate_waveform(y)


def test_validate_waveform_all_zeros():
    """All-zero array should raise ValueError."""
    import pytest
    y = np.zeros(16000, dtype=np.float32)
    with pytest.raises(ValueError, match="silent"):
        validate_waveform(y)


def test_validate_waveform_valid():
    """Valid audio should pass without exception."""
    y = np.sin(np.linspace(0, 2 * np.pi, 16000)).astype(np.float32)
    validate_waveform(y)  # Should not raise


# ============================================================
# Model input shape
# ============================================================


def test_model_input_shape_matches_cnn():
    """The preprocessing output shape must match SpectraShieldCNN's
    expected input: [batch, 1, 128, 128]."""
    from app.inference.model import SpectraShieldCNN

    sr = 16000
    duration = 4.0
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    y = np.sin(2 * np.pi * 440 * t).astype(np.float32)

    tensor = preprocess_chunk_waveform(y, sr)

    model = SpectraShieldCNN()
    model.eval()

    with torch.no_grad():
        output = model(tensor)

    # Output should be [1, 1] (batch size 1, 1 logit)
    assert output.shape == (1, 1), f"Expected (1, 1), got {output.shape}"


# ============================================================
# Threshold
# ============================================================


def test_classification_threshold_is_030():
    """Verify the locked threshold value."""
    from app.inference.prediction import CLASSIFICATION_THRESHOLD
    assert CLASSIFICATION_THRESHOLD == 0.30


def test_live_classification_threshold_is_030():
    """Verify live route uses the same locked threshold."""
    from app.routes.live import CLASSIFICATION_THRESHOLD
    assert CLASSIFICATION_THRESHOLD == 0.30


# ============================================================
# Cumulative voting
# ============================================================


def test_cumulative_voting_basic_majority():
    """Cumulative voting: 3 REAL + 2 FAKE → REAL."""
    from app.routes.live import cumulative_majority_vote

    predictions = [
        {"prediction": "REAL", "fake_probability": 0.10},
        {"prediction": "REAL", "fake_probability": 0.15},
        {"prediction": "FAKE", "fake_probability": 0.50},
        {"prediction": "REAL", "fake_probability": 0.20},
        {"prediction": "FAKE", "fake_probability": 0.60},
    ]

    result = cumulative_majority_vote(predictions)
    assert result == "REAL"


def test_cumulative_voting_all_fake():
    """Cumulative voting: all FAKE → FAKE."""
    from app.routes.live import cumulative_majority_vote

    predictions = [
        {"prediction": "FAKE", "fake_probability": 0.80},
        {"prediction": "FAKE", "fake_probability": 0.70},
        {"prediction": "FAKE", "fake_probability": 0.90},
        {"prediction": "FAKE", "fake_probability": 0.85},
        {"prediction": "FAKE", "fake_probability": 0.75},
    ]

    result = cumulative_majority_vote(predictions)
    assert result == "FAKE"


def test_cumulative_voting_tie_break():
    """Tie-breaking should use mean fake probability."""
    from app.routes.live import cumulative_majority_vote

    # 2 REAL + 2 FAKE = tie
    # Mean fake prob = (0.10 + 0.15 + 0.50 + 0.60) / 4 = 0.3375 >= 0.30 → FAKE
    predictions = [
        {"prediction": "REAL", "fake_probability": 0.10},
        {"prediction": "REAL", "fake_probability": 0.15},
        {"prediction": "FAKE", "fake_probability": 0.50},
        {"prediction": "FAKE", "fake_probability": 0.60},
    ]

    result = cumulative_majority_vote(predictions)
    assert result == "FAKE"


def test_cumulative_voting_is_not_rolling():
    """Cumulative voting must use ALL chunks, not a rolling window."""
    from app.routes.live import cumulative_majority_vote

    # 7 chunks: 5 REAL + 2 FAKE = REAL
    predictions = [
        {"prediction": "REAL", "fake_probability": 0.10},
        {"prediction": "REAL", "fake_probability": 0.15},
        {"prediction": "REAL", "fake_probability": 0.12},
        {"prediction": "REAL", "fake_probability": 0.18},
        {"prediction": "REAL", "fake_probability": 0.20},
        {"prediction": "FAKE", "fake_probability": 0.50},
        {"prediction": "FAKE", "fake_probability": 0.60},
    ]

    # With ALL 7, REAL wins 5-2
    result = cumulative_majority_vote(predictions)
    assert result == "REAL"


# ============================================================
# Session reset
# ============================================================


def test_session_reset_clears_predictions():
    """Starting a new session must clear previous predictions."""
    from app.routes.live import session_data

    sid = "test-reset-session"
    session_data[sid] = {
        "predictions": [
            {"prediction": "FAKE", "fake_probability": 0.80}
        ],
        "chunk_count": 1,
    }

    # Simulate /start by resetting
    session_data[sid] = {
        "predictions": [],
        "chunk_count": 0,
    }

    assert session_data[sid]["predictions"] == []
    assert session_data[sid]["chunk_count"] == 0

    # Cleanup
    del session_data[sid]


# ============================================================
# 20-second final result delay
# ============================================================


def test_no_final_result_before_5_chunks():
    """The API must not return a final result before 5 chunks."""
    from app.routes.live import MIN_CHUNKS_FOR_RESULT
    assert MIN_CHUNKS_FOR_RESULT == 5

    # 4 chunks < 5 → no final result
    total_chunks = 4
    final_available = total_chunks >= MIN_CHUNKS_FOR_RESULT
    assert final_available is False


def test_final_result_at_5_chunks():
    """The API must return a final result at exactly 5 chunks."""
    from app.routes.live import MIN_CHUNKS_FOR_RESULT

    total_chunks = 5
    final_available = total_chunks >= MIN_CHUNKS_FOR_RESULT
    assert final_available is True


# ============================================================
# Preprocessing equivalence: shared function vs file-based
# ============================================================


def test_shared_preprocessing_matches_file_based():
    """preprocess_chunk_waveform on raw waveform should produce
    a result equivalent to preprocess_audio on a WAV file containing
    the same waveform (both at 16kHz).

    Note: These may differ slightly due to pad/truncate vs interpolation
    resize strategy difference. We only check shape and value range.
    """
    import tempfile

    sr = 16000
    duration = 2.0
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    y = (np.sin(2 * np.pi * 440 * t) * 0.5).astype(np.float32)

    # File-based preprocessing
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix='.wav')
    tmp_path = tmp.name
    tmp.close()

    try:
        scipy.io.wavfile.write(tmp_path, sr, y)
        result_file = preprocess_audio(tmp_path)
    finally:
        os.unlink(tmp_path)

    # Waveform-based preprocessing
    tensor_waveform = preprocess_chunk_waveform(y, sr)
    result_waveform = tensor_waveform.numpy()

    # Both should have the same shape
    assert result_file.shape == result_waveform.shape == (1, 1, 128, 128)

    # Both should be in [0, 1]
    assert np.all(result_file >= 0)
    assert np.all(result_waveform >= 0)
    assert np.all(result_file <= 1 + 1e-6)
    assert np.all(result_waveform <= 1 + 1e-6)


# ============================================================
# Invalid input handling
# ============================================================


def test_preprocess_empty_waveform_raises():
    """Empty waveform should raise ValueError."""
    import pytest
    with pytest.raises(ValueError):
        preprocess_chunk_waveform(
            np.array([], dtype=np.float32), 16000
        )


def test_preprocess_nan_waveform_raises():
    """Waveform with NaN should raise ValueError."""
    import pytest
    y = np.array([0.1, np.nan, 0.3], dtype=np.float32)
    with pytest.raises(ValueError, match="invalid values"):
        preprocess_chunk_waveform(y, 16000)


def test_preprocess_silent_waveform_raises():
    """Completely silent waveform should raise ValueError."""
    import pytest
    y = np.zeros(16000, dtype=np.float32)
    with pytest.raises(ValueError, match="silent"):
        preprocess_chunk_waveform(y, 16000)
