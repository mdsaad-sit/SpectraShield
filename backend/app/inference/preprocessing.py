"""Shared preprocessing utility for SpectraShieldCNN inference.

This module provides the CANONICAL preprocessing functions used by
both the recorded and live inference pipelines. Both routes delegate
to `preprocess_chunk_waveform()` to ensure identical preprocessing.

Architecture:
    waveform (numpy array at any sample rate)
        ↓
    preprocess_chunk_waveform()
        ↓
    [1, 1, 128, 128] torch.Tensor (float32)
        ↓
    SpectraShieldCNN

The only difference between recorded and live is how chunks are created:
    Recorded: file → librosa.load(sr=16000) → 5-sec chunks
    Live:     browser PCM at native rate → 4-sec chunks
"""

import librosa
import numpy as np
import torch
import logging

logger = logging.getLogger(__name__)


SR = 16000
N_MELS = 128
N_FFT = 1024
HOP_LENGTH = 256
TARGET_FRAMES = 128
MAX_AMPLITUDE = 1e-8


def validate_waveform(y):
    """Validate that a waveform contains usable audio data.

    Checks for: empty, NaN, Inf, all-zeros, extremely low amplitude.

    Args:
        y: numpy array of audio time series (float32)

    Raises:
        ValueError: if the waveform is invalid or unusable
    """
    if y is None or len(y) == 0:
        raise ValueError("Audio chunk is empty")

    if not np.all(np.isfinite(y)):
        nan_count = np.sum(np.isnan(y))
        inf_count = np.sum(np.isinf(y))
        raise ValueError(
            f"Audio contains invalid values: "
            f"{nan_count} NaN, {inf_count} Inf "
            f"out of {len(y)} samples"
        )

    max_abs = np.max(np.abs(y))
    if max_abs == 0:
        raise ValueError(
            "Audio chunk is completely silent (all zeros)"
        )

    # Warn but don't reject extremely quiet audio
    if max_abs < 1e-6:
        logger.warning(
            "Audio chunk has extremely low amplitude: "
            "max_abs=%.2e", max_abs
        )


def preprocess_chunk_waveform(chunk_y, sr):
    """Convert one audio chunk into the exact Log-Mel representation
    expected by SpectraShieldCNN.

    This is the CANONICAL preprocessing function. Both the recorded
    and live inference routes MUST delegate to this function to ensure
    identical preprocessing.

    Steps:
        1. Validate waveform
        2. Resample to 16 kHz if needed
        3. Normalize by max absolute amplitude
        4. Mel spectrogram (power=2.0)
        5. Power-to-dB (ref=np.max)
        6. Min-max normalize to [0, 1]
        7. Interpolate time dimension to exactly 128 frames
        8. Return as [1, 1, 128, 128] torch.Tensor

    Args:
        chunk_y: numpy array of audio time series (float32)
        sr: sample rate of the input audio

    Returns:
        torch.Tensor of shape [1, 1, 128, 128], dtype float32
    """
    chunk_y = np.asarray(chunk_y, dtype=np.float32)

    # Validate
    validate_waveform(chunk_y)

    # Resample to training sample rate if needed
    if sr != SR:
        chunk_y = librosa.resample(
            chunk_y,
            orig_sr=sr,
            target_sr=SR
        )
        sr = SR

    # Normalize waveform by max absolute amplitude
    max_amp = np.max(np.abs(chunk_y))
    if max_amp > 0:
        chunk_y = chunk_y / max_amp

    # Mel spectrogram
    mel = librosa.feature.melspectrogram(
        y=chunk_y,
        sr=sr,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        n_mels=N_MELS,
        power=2.0
    )

    # Log scale
    logmel = librosa.power_to_db(mel, ref=np.max)

    # Min-max normalization
    logmel_min = logmel.min()
    logmel_max = logmel.max()
    logmel = (logmel - logmel_min) / (logmel_max - logmel_min + 1e-8)

    # Resize to exactly TARGET_FRAMES time frames
    current_frames = logmel.shape[1]

    if current_frames != TARGET_FRAMES:
        old_x = np.linspace(0, 1, current_frames)
        new_x = np.linspace(0, 1, TARGET_FRAMES)

        resized = np.empty(
            (N_MELS, TARGET_FRAMES),
            dtype=np.float32
        )

        for m in range(N_MELS):
            resized[m] = np.interp(new_x, old_x, logmel[m])

        logmel = resized

    # Convert to tensor [1, 1, 128, 128]
    logmel = logmel.astype(np.float32)
    tensor = torch.from_numpy(logmel).unsqueeze(0).unsqueeze(0)

    return tensor


def preprocess_audio(path):
    """Preprocess audio file for SpectraShieldCNN inference.

    Loads audio from a file path and runs the canonical preprocessing.

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