"""Shared preprocessing utility for SpectraShieldCNN inference.

Used by: tests/test_preprocessing.py

Note: The route modules (routes/recorded.py, routes/live.py) contain
their own inline preprocess_audio_chunk() optimised for chunk-level
inference with interpolation resizing. This module uses pad/truncate
resizing and operates on whole audio files.
"""

import librosa
import numpy as np
import logging

logger = logging.getLogger(__name__)


SR = 16000
N_MELS = 128
N_FFT = 1024
HOP_LENGTH = 256
TARGET_FRAMES = 128
MAX_AMPLITUDE = 1e-8


def preprocess_audio(path):
    """Preprocess audio file for SpectraShieldCNN inference.

    Returns:
        numpy array of shape (1, 1, 128, 128) as float32
    """
    y, _ = librosa.load(path, sr=SR, mono=True)

    if len(y) == 0:
        raise ValueError("Empty audio")

    max_amp = np.max(np.abs(y))
    if max_amp > 0:
        y = y / max_amp

    mel = librosa.feature.melspectrogram(
        y=y,
        sr=SR,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        n_mels=N_MELS,
        power=2.0,
    )

    logmel = librosa.power_to_db(mel, ref=np.max)

    logmel = (logmel - logmel.min()) / (logmel.max() - logmel.min() + 1e-8)

    t = logmel.shape[1]
    if t < TARGET_FRAMES:
        pad_width = TARGET_FRAMES - t
        logmel = np.pad(logmel, ((0, 0), (0, pad_width)), mode='constant')
    else:
        logmel = logmel[:, :TARGET_FRAMES]

    logmel = logmel.astype(np.float32)
    logmel = logmel[np.newaxis, np.newaxis, :, :]

    return logmel