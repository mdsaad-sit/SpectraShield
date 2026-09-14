from flask import Blueprint, request, jsonify
from app.inference.model_loader import ModelLoader
from app.inference.chunking import chunk_audio
from app.inference.gradcam import GradCAM
import logging
import tempfile
import os
import base64
from io import BytesIO

import numpy as np
import torch
import librosa
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)

recorded_bp = Blueprint("recorded", __name__)

# SpectraShield inference configuration
SR = 16000
N_MELS = 128
N_FFT = 1024
HOP_LENGTH = 256
TARGET_FRAMES = 128

# IMPORTANT:
# 0 = REAL
# 1 = FAKE
CLASSIFICATION_THRESHOLD = 0.30


@recorded_bp.route("/predict", methods=["POST"])
def predict_recorded():
    """
    Handle recorded audio prediction.

    Expected:
        multipart/form-data
        field name: audio

    Processing:
        1. Load audio at 16 kHz mono
        2. Split into 5-second chunks
        3. Generate Log-Mel spectrogram for every chunk
        4. Predict every chunk independently
        5. Apply 0.30 threshold
        6. Majority vote across all chunks
    """

    # ---------------------------------------------------------
    # 1. Validate uploaded file
    # ---------------------------------------------------------
    if "audio" not in request.files:
        return jsonify({
            "error": "No audio file provided"
        }), 400

    file = request.files["audio"]

    if file.filename == "":
        return jsonify({
            "error": "No audio file provided"
        }), 400

    # ---------------------------------------------------------
    # 2. Check model
    # ---------------------------------------------------------
    if not ModelLoader.is_loaded():
        return jsonify({
            "error": (
                "Trained model files are not installed. "
                "Place best_model.pth and model_config.json "
                "inside backend/model/"
            )
        }), 500

    tmp_path = None

    try:
        # -----------------------------------------------------
        # 3. Save uploaded audio temporarily
        # -----------------------------------------------------
        suffix = os.path.splitext(file.filename)[1] or ".wav"

        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=suffix
        ) as tmp:
            tmp_path = tmp.name
            file.save(tmp_path)

        # -----------------------------------------------------
        # 4. Load audio
        # -----------------------------------------------------
        y, sr = librosa.load(
            tmp_path,
            sr=SR,
            mono=True
        )

        if y is None or len(y) == 0:
            return jsonify({
                "error": "Empty audio file"
            }), 400

        # -----------------------------------------------------
        # 5. Split into 5-second chunks
        # -----------------------------------------------------
        chunks = chunk_audio(
            y,
            sr=sr,
            chunk_duration=5
        )

        if not chunks:
            return jsonify({
                "error": "Could not chunk audio"
            }), 400

        logger.info(
            "Recorded audio split into %d chunks",
            len(chunks)
        )

        # -----------------------------------------------------
        # 6. Process every chunk
        # -----------------------------------------------------
        chunk_results = []

        for chunk_idx, start_sample, end_sample, chunk_y in chunks:

            try:
                # ---------------------------------------------
                # Preprocess
                # ---------------------------------------------
                tensor = preprocess_audio_chunk(
                    chunk_y,
                    sr
                )

                # ---------------------------------------------
                # Model prediction
                # ---------------------------------------------
                _, fake_prob = ModelLoader.predict(tensor)

                prediction = (
                    "FAKE"
                    if fake_prob >= CLASSIFICATION_THRESHOLD
                    else "REAL"
                )

                # Confidence:
                # For FAKE, fake probability is confidence.
                # For REAL, 1 - fake probability is confidence.
                confidence = (
                    fake_prob
                    if prediction == "FAKE"
                    else 1.0 - fake_prob
                )

                # Actual time values
                start_time = start_sample / sr
                end_time = end_sample / sr

                # ---------------------------------------------
                # Grad-CAM explanation
                # ---------------------------------------------
                gradcam_image = None
                gradcam_error = None
                try:
                    model = ModelLoader.get_model()
                    gradcam = GradCAM(model)

                    try:
                        gradcam_heatmap = gradcam.generate(
                            tensor.to(ModelLoader.get_device())
                        )
                    finally:
                        gradcam.close()

                    gradcam_image = create_gradcam_image(
                        tensor,
                        gradcam_heatmap,
                        chunk_idx,
                        start_time,
                        end_time,
                        prediction,
                        fake_prob
                    )
                    original_logmel_image = create_logmel_image(
                        tensor,
                        chunk_idx,
                        start_time,
                        end_time
                    )

                except Exception as e:
                    gradcam_error = str(e)
                    original_logmel_image = None
                    logger.exception(
                        "Grad-CAM failed for chunk %s",
                        chunk_idx
                    )

                chunk_results.append({
                    "chunk_index": chunk_idx,
                    "start_time": round(start_time, 3),
                    "end_time": round(end_time, 3),
                    "prediction": prediction,
                    "fake_probability": round(
                        float(fake_prob),
                        4
                    ),
                    "confidence": round(
                        float(confidence),
                        4
                    ),
                    "gradcam_image": gradcam_image,
                    "original_logmel_image": original_logmel_image,
                    "gradcam_error": gradcam_error
                })

            except Exception as e:
                logger.exception(
                    "Error processing chunk %s",
                    chunk_idx
                )

                # Do NOT silently classify a failed chunk as REAL.
                # Return an error instead so the result is trustworthy.
                return jsonify({
                    "error": (
                        f"Error processing chunk "
                        f"{chunk_idx}: {str(e)}"
                    )
                }), 500

        # -----------------------------------------------------
        # 7. Majority voting
        # -----------------------------------------------------
        chunk_predictions = [
            r["prediction"]
            for r in chunk_results
        ]

        chunk_probs = [
            r["fake_probability"]
            for r in chunk_results
        ]

        (
            final_prediction,
            confidence,
            real_chunks,
            fake_chunks,
            mean_fake_prob
        ) = majority_vote(
            chunk_predictions,
            chunk_probs
        )

        total_chunks = len(chunk_results)

        # -----------------------------------------------------
        # 8. Final response
        # -----------------------------------------------------
        result = {
            "mode": "recorded",
            "final_prediction": final_prediction,
            "confidence": round(
                float(confidence),
                4
            ),
            "mean_fake_probability": round(
                float(mean_fake_prob),
                4
            ),
            "total_chunks": total_chunks,
            "real_chunks": real_chunks,
            "fake_chunks": fake_chunks,
            "chunks": chunk_results
        }

        return jsonify(result), 200

    except Exception as e:

        logger.exception(
            "Error in recorded prediction"
        )

        return jsonify({
            "error": str(e)
        }), 500

    finally:
        # -----------------------------------------------------
        # 9. Always remove temporary file
        # -----------------------------------------------------
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.unlink(tmp_path)
            except Exception:
                logger.warning(
                    "Could not delete temporary file: %s",
                    tmp_path
                )


def preprocess_audio_chunk(chunk_y, sr):
    """
    Convert one audio chunk into the exact Log-Mel
    representation expected by SpectraShieldCNN.

    Output:
        torch.Tensor with shape:
        [1, 1, 128, 128]
    """

    chunk_y = np.asarray(
        chunk_y,
        dtype=np.float32
    )

    if len(chunk_y) == 0:
        raise ValueError(
            "Audio chunk is empty"
        )

    # ---------------------------------------------------------
    # Normalize waveform
    # ---------------------------------------------------------
    max_amp = np.max(
        np.abs(chunk_y)
    )

    if max_amp > 0:
        chunk_y = chunk_y / max_amp

    # ---------------------------------------------------------
    # Generate Mel spectrogram
    # ---------------------------------------------------------
    mel = librosa.feature.melspectrogram(
        y=chunk_y,
        sr=sr,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        n_mels=N_MELS,
        power=2.0
    )

    # ---------------------------------------------------------
    # Convert to logarithmic scale
    # ---------------------------------------------------------
    logmel = librosa.power_to_db(
        mel,
        ref=np.max
    )

    # ---------------------------------------------------------
    # Min-Max normalization
    # ---------------------------------------------------------
    logmel_min = logmel.min()
    logmel_max = logmel.max()

    logmel = (
        logmel - logmel_min
    ) / (
        logmel_max - logmel_min + 1e-8
    )

    # ---------------------------------------------------------
    # Resize complete spectrogram to 128 frames
    #
    # IMPORTANT:
    # This matches the preprocessing used during training.
    # ---------------------------------------------------------
    current_frames = logmel.shape[1]

    if current_frames != TARGET_FRAMES:

        old_x = np.linspace(
            0,
            1,
            current_frames
        )

        new_x = np.linspace(
            0,
            1,
            TARGET_FRAMES
        )

        resized = np.empty(
            (N_MELS, TARGET_FRAMES),
            dtype=np.float32
        )

        for m in range(N_MELS):
            resized[m] = np.interp(
                new_x,
                old_x,
                logmel[m]
            )

        logmel = resized

    # ---------------------------------------------------------
    # Convert to tensor
    #
    # [128, 128]
    #       ↓
    # [1, 128, 128]
    #       ↓
    # [1, 1, 128, 128]
    # ---------------------------------------------------------
    logmel = logmel.astype(
        np.float32
    )

    tensor = torch.from_numpy(
        logmel
    ).unsqueeze(0).unsqueeze(0)

    return tensor


def _encode_figure(figure):
    """Encode a matplotlib figure as an in-memory PNG data URL."""

    image_buffer = BytesIO()
    figure.savefig(
        image_buffer,
        format="png",
        bbox_inches="tight"
    )
    plt.close(figure)

    encoded_image = base64.b64encode(
        image_buffer.getvalue()
    ).decode("ascii")

    return f"data:image/png;base64,{encoded_image}"


def create_logmel_image(
    input_tensor,
    chunk_idx,
    start_time,
    end_time
):
    """Render the exact normalized Log-Mel tensor used by the model."""

    spectrogram = input_tensor[0, 0].detach().cpu().numpy()
    figure, axis = plt.subplots(figsize=(8, 4.5), dpi=120)

    axis.imshow(
        spectrogram,
        origin="lower",
        aspect="auto",
        cmap="magma",
        vmin=0,
        vmax=1
    )
    axis.set_title(
        f"Chunk {chunk_idx + 1} | "
        f"{start_time:.1f}s - {end_time:.1f}s"
    )
    axis.set_xlabel("Normalized time")
    axis.set_ylabel("Mel frequency bin")
    figure.tight_layout()

    return _encode_figure(figure)


def create_gradcam_image(
    input_tensor,
    heatmap,
    chunk_idx,
    start_time,
    end_time,
    prediction,
    fake_probability
):
    """Render a Log-Mel spectrogram and Grad-CAM overlay in memory."""

    spectrogram = input_tensor[0, 0].detach().cpu().numpy()

    figure, axis = plt.subplots(
        figsize=(8, 4.5),
        dpi=120
    )

    axis.imshow(
        spectrogram,
        origin="lower",
        aspect="auto",
        cmap="magma",
        vmin=0,
        vmax=1
    )
    axis.imshow(
        heatmap,
        origin="lower",
        aspect="auto",
        cmap="jet",
        alpha=0.42,
        vmin=0,
        vmax=1
    )
    axis.set_title(
        f"Chunk {chunk_idx + 1} | "
        f"{start_time:.1f}s - {end_time:.1f}s | "
        f"{prediction} | Fake probability: "
        f"{fake_probability * 100:.1f}%"
    )
    axis.set_xlabel("Normalized time")
    axis.set_ylabel("Mel frequency bin")

    figure.tight_layout()

    return _encode_figure(figure)


def majority_vote(predictions, fake_probabilities):
    """
    Perform majority voting across chunk predictions.

    predictions:
        ["REAL", "FAKE", ...]

    fake_probabilities:
        [0.12, 0.87, ...]

    Tie-breaking:
        If REAL and FAKE counts are equal,
        use mean FAKE probability.

        mean >= 0.30 -> FAKE
        mean < 0.30  -> REAL

    Returns:
        final_prediction,
        confidence,
        real_chunks,
        fake_chunks,
        mean_fake_probability
    """

    if not predictions:
        raise ValueError(
            "No predictions available for voting"
        )

    if len(predictions) != len(fake_probabilities):
        raise ValueError(
            "Predictions and probabilities "
            "must have the same length"
        )

    real_chunks = predictions.count(
        "REAL"
    )

    fake_chunks = predictions.count(
        "FAKE"
    )

    mean_fake_prob = float(
        np.mean(fake_probabilities)
    )

    # ---------------------------------------------------------
    # Majority
    # ---------------------------------------------------------
    if fake_chunks > real_chunks:

        final_prediction = "FAKE"

    elif real_chunks > fake_chunks:

        final_prediction = "REAL"

    else:
        # -----------------------------------------------------
        # Tie-break
        # -----------------------------------------------------
        final_prediction = (
            "FAKE"
            if mean_fake_prob >= CLASSIFICATION_THRESHOLD
            else "REAL"
        )

    # ---------------------------------------------------------
    # Final confidence
    # ---------------------------------------------------------
    if final_prediction == "FAKE":

        confidence = mean_fake_prob

    else:

        confidence = 1.0 - mean_fake_prob

    return (
        final_prediction,
        confidence,
        real_chunks,
        fake_chunks,
        mean_fake_prob
    )