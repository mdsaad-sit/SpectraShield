import logging
import numpy as np

logger = logging.getLogger(__name__)


def chunk_audio(audio_array, sr=16000, chunk_duration=5):
    """Split audio into chunks of specified duration (seconds).

    Args:
        audio_array: numpy array of audio time series
        sr: sampling rate
        chunk_duration: chunk duration in seconds

    Returns:
        list of (chunk_index, start_sample, end_sample, chunk_audio) tuples
    """
    samples_per_chunk = sr * chunk_duration
    total_samples = len(audio_array)
    chunks = []

    if total_samples <= 0:
        return chunks

    chunk_index = 0
    start = 0
    while start < total_samples:
        end = min(start + samples_per_chunk, total_samples)
        chunk_audio_data = audio_array[start:end]

        if len(chunk_audio_data) < samples_per_chunk:
            pad_len = samples_per_chunk - len(chunk_audio_data)
            chunk_audio_data = np.pad(chunk_audio_data, (0, pad_len), mode='constant')

        chunks.append((chunk_index, start, end, chunk_audio_data))
        chunk_index += 1
        start += samples_per_chunk

    return chunks


def get_chunk_count(duration_seconds, sr=16000, chunk_duration=5):
    """Calculate number of chunks for a given duration.

    Args:
        duration_seconds: total duration in seconds
        sr: sampling rate
        chunk_duration: chunk duration in seconds

    Returns:
        number of chunks (rounds up, final partial chunk padded)
    """
    samples_per_chunk = sr * chunk_duration
    total_samples = int(duration_seconds * sr)
    if total_samples <= 0:
        return 0
    chunks = total_samples // samples_per_chunk
    if total_samples % samples_per_chunk > 0:
        chunks += 1
    return chunks


def get_chunk_count_live(duration_seconds, sr=16000):
    """Calculate number of live 4-second chunks for a given duration.

    This is the same as get_chunk_count with chunk_duration=4.

    Args:
        duration_seconds: total duration in seconds
        sr: sampling rate

    Returns:
        number of 4-second live chunks
    """
    return get_chunk_count(duration_seconds, sr=sr, chunk_duration=4)