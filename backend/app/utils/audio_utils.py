import logging
import tempfile
import os

logger = logging.getLogger(__name__)


def save_temp_audio(audio_data, sample_rate=16000):
    """Save audio data to a temporary WAV file.

    Args:
        audio_data: numpy array of audio time series
        sample_rate: sampling rate

    Returns:
        Path to temporary file
    """
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
    tmp_path = tmp.name
    tmp.close()

    import scipy.io.wavfile
    scipy.io.wavfile.write(tmp_path, sample_rate, audio_data)

    return tmp_path


def cleanup_temp_file(path):
    """Remove temporary file if it exists."""
    if path and os.path.exists(path):
        try:
            os.unlink(path)
        except OSError as e:
            logger.warning(f"Could not remove temp file {path}: {e}")