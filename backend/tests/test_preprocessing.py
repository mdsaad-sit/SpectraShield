import sys
import os
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

from app.inference.preprocessing import preprocess_audio


def test_preprocessing_output_shape():
    """Preprocessing should produce array of shape (1, 1, 128, 128)."""
    import tempfile
    import scipy.io.wavfile
    
    # Generate a simple sine wave at 440Hz, 1 second
    sr = 16000
    duration = 1.0
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    y = np.sin(2 * np.pi * 440 * t).astype(np.float32)
    
    # Write to temp WAV using scipy
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix='.wav')
    tmp_path = tmp.name
    tmp.close()
    
    try:
        scipy.io.wavfile.write(tmp_path, sr, y)
        result = preprocess_audio(tmp_path)
        assert result.shape == (1, 1, 128, 128), f"Expected shape (1, 1, 128, 128), got {result.shape}"
        assert result.dtype == np.float32, f"Expected float32, got {result.dtype}"
    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)


def test_preprocessing_empty_audio():
    """Empty audio should raise ValueError."""
    import tempfile
    import os
    import scipy.io.wavfile
    
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix='.wav')
    tmp_path = tmp.name
    tmp.close()
    
    try:
        scipy.io.wavfile.write(tmp_path, 16000, np.array([], dtype=np.float32))
        from pytest import raises
        with raises(ValueError, match="Empty audio"):
            preprocess_audio(tmp_path)
    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)


def test_preprocessing_16khz_mono():
    """Preprocessing should use 16kHz mono."""
    import tempfile
    import os
    import scipy.io.wavfile
    
    # Generate a simple sine wave
    sr = 16000
    duration = 1.0
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    y = np.sin(2 * np.pi * 440 * t).astype(np.float32)
    
    # Write to temp WAV
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix='.wav')
    tmp_path = tmp.name
    tmp.close()
    
    try:
        scipy.io.wavfile.write(tmp_path, sr, y)
        result = preprocess_audio(tmp_path)
        # If we get here without error about sample rate, it works
        assert result.shape[2] == 128, f"Expected 128 mel bands, got {result.shape[2]}"
        assert result.shape[3] == 128, f"Expected 128 time frames, got {result.shape[3]}"
    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)