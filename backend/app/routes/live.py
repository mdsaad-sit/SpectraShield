from flask import Blueprint, request, jsonify
from app.inference.model_loader import ModelLoader
from app.inference.preprocessing import (
    preprocess_chunk_waveform,
    validate_waveform,
    SR,
    N_MELS,
    N_FFT,
    HOP_LENGTH,
    TARGET_FRAMES,
)
import logging

import numpy as np

logger = logging.getLogger(__name__)

live_bp = Blueprint("live", __name__)

# ============================================================
# SpectraShield configuration
# ============================================================

# 0 = REAL
# 1 = FAKE
CLASSIFICATION_THRESHOLD = 0.30

# Live monitoring:
# 4 seconds per chunk
# First final result after 5 chunks = 20 seconds
CHUNK_DURATION = 4
MIN_CHUNKS_FOR_RESULT = 5

# Session storage
#
# session_data = {
#     session_id: {
#         "predictions": [],
#         "chunk_count": 0
#     }
# }
#
# Predictions are NEVER discarded during a session.
session_data = {}


# ============================================================
# Live health
# ============================================================

@live_bp.route("/live/health", methods=["GET"])
def live_health():
    """Live monitoring health check."""

    device = ModelLoader.get_device()

    return jsonify({
        "status": "ok",
        "model_loaded": ModelLoader.is_loaded(),
        "device": device.type if device else "unknown",
        "chunk_duration": CHUNK_DURATION,
        "minimum_chunks": MIN_CHUNKS_FOR_RESULT,
        "minimum_duration": CHUNK_DURATION * MIN_CHUNKS_FOR_RESULT
    }), 200


# ============================================================
# Start live session
# ============================================================

@live_bp.route("/start", methods=["POST"])
def live_start():
    """
    Start a NEW live monitoring session.

    A new session always starts with zero accumulated
    predictions.
    """

    data = request.get_json(silent=True) or {}

    session_id = data.get("session_id")

    if not session_id:
        return jsonify({
            "error": "session_id is required"
        }), 400

    if not ModelLoader.is_loaded():
        return jsonify({
            "error": (
                "Trained model files are not installed. "
                "Place best_model.pth and model_config.json "
                "inside backend/model/"
            )
        }), 500

    # IMPORTANT:
    # Always reset the session when /start is called.
    session_data[session_id] = {
        "predictions": [],
        "chunk_count": 0
    }

    logger.info(
        "Started live session: %s",
        session_id
    )

    return jsonify({
        "status": "started",
        "session_id": session_id,
        "model_loaded": True,
        "device": ModelLoader.get_device().type,
        "chunk_duration": CHUNK_DURATION,
        "chunks_per_initial_result": MIN_CHUNKS_FOR_RESULT,
        "initial_result_after_seconds": (
            CHUNK_DURATION * MIN_CHUNKS_FOR_RESULT
        ),
        "voting_mode": "cumulative"
    }), 200


# ============================================================
# Predict one live chunk
# ============================================================

@live_bp.route("/predict_chunk", methods=["POST"])
def predict_live_chunk():
    """
    Process exactly one live audio chunk.

    Expected JSON:

    {
        "session_id": "abc123",
        "audio_data": [...],
        "sample_rate": 16000,
        "chunk_index": 0
    }

    Rules:

    Chunk 1  -> 4 sec  -> NO FINAL RESULT
    Chunk 2  -> 8 sec  -> NO FINAL RESULT
    Chunk 3  -> 12 sec -> NO FINAL RESULT
    Chunk 4  -> 16 sec -> NO FINAL RESULT
    Chunk 5  -> 20 sec -> FINAL RESULT using chunks 1-5
    Chunk 6  -> 24 sec -> UPDATED RESULT using chunks 1-6
    Chunk 7  -> 28 sec -> UPDATED RESULT using chunks 1-7

    And so on.

    This is cumulative voting, NOT rolling-window voting.
    """

    data = request.get_json(silent=True)

    if not data:
        return jsonify({
            "error": "No data provided"
        }), 400

    # --------------------------------------------------------
    # Session ID
    # --------------------------------------------------------

    session_id = data.get("session_id")

    if not session_id:
        return jsonify({
            "error": "session_id is required"
        }), 400

    # --------------------------------------------------------
    # Make sure session exists
    # --------------------------------------------------------

    if session_id not in session_data:
        return jsonify({
            "error": (
                "Live session not found. "
                "Call /api/start first."
            )
        }), 400

    # --------------------------------------------------------
    # Model check
    # --------------------------------------------------------

    if not ModelLoader.is_loaded():
        return jsonify({
            "error": "Trained model files are not installed."
        }), 500

    # --------------------------------------------------------
    # Audio data
    # --------------------------------------------------------

    raw_audio = data.get("audio_data", [])

    if not raw_audio:
        return jsonify({
            "error": "Empty audio data"
        }), 400

    try:
        audio_data = np.asarray(
            raw_audio,
            dtype=np.float32
        )
    except Exception:
        return jsonify({
            "error": "Invalid audio_data"
        }), 400

    if audio_data.size == 0:
        return jsonify({
            "error": "Empty audio data"
        }), 400

    # --------------------------------------------------------
    # Sample rate
    # --------------------------------------------------------

    try:
        sample_rate = int(
            data.get("sample_rate", SR)
        )
    except (TypeError, ValueError):
        return jsonify({
            "error": "Invalid sample_rate"
        }), 400

    if sample_rate <= 0:
        return jsonify({
            "error": "Invalid sample_rate"
        }), 400

    # --------------------------------------------------------
    # Chunk index
    # --------------------------------------------------------

    try:
        chunk_index = int(
            data.get("chunk_index", 0)
        )
    except (TypeError, ValueError):
        return jsonify({
            "error": "Invalid chunk_index"
        }), 400

    if chunk_index < 0:
        return jsonify({
            "error": "chunk_index must be >= 0"
        }), 400

    # --------------------------------------------------------
    # Waveform validation
    # --------------------------------------------------------

    try:
        validate_waveform(audio_data)
    except ValueError as e:
        logger.warning(
            "Live chunk %d (session %s) waveform validation "
            "failed: %s",
            chunk_index, session_id, str(e)
        )
        return jsonify({
            "error": f"Invalid audio: {str(e)}"
        }), 400

    # --------------------------------------------------------
    # Diagnostic logging
    # --------------------------------------------------------

    _log_chunk_diagnostics(
        session_id, chunk_index, audio_data, sample_rate,
        stage="received"
    )

    try:

        # ----------------------------------------------------
        # Preprocess using the shared canonical function
        # ----------------------------------------------------

        tensor = preprocess_chunk_waveform(
            audio_data,
            sample_rate
        )

        # ----------------------------------------------------
        # Model prediction
        # ----------------------------------------------------

        logit, fake_prob = ModelLoader.predict(
            tensor
        )

        fake_prob = float(fake_prob)
        logit_val = float(logit.item()) if hasattr(logit, 'item') else float(logit)

        # ----------------------------------------------------
        # Classification
        # ----------------------------------------------------

        prediction = classify_probability(
            fake_prob
        )

        confidence = calculate_confidence(
            prediction,
            fake_prob
        )

        # Diagnostic logging for prediction
        logger.info(
            "[LIVE DIAG] session=%s chunk=%d | "
            "logit=%.4f sigmoid=%.4f prediction=%s",
            session_id, chunk_index,
            logit_val, fake_prob, prediction
        )

        # ----------------------------------------------------
        # Add prediction to cumulative session history
        # ----------------------------------------------------

        session = session_data[session_id]

        session["predictions"].append({
            "chunk_index": chunk_index,
            "prediction": prediction,
            "fake_probability": fake_prob,
            "elapsed_seconds": (
                (chunk_index + 1) * CHUNK_DURATION
            )
        })

        session["chunk_count"] += 1

        total_chunks = session["chunk_count"]

        elapsed_time = (
            total_chunks * CHUNK_DURATION
        )

        # ----------------------------------------------------
        # Determine whether final result is available
        # ----------------------------------------------------

        final_result_available = (
            total_chunks >= MIN_CHUNKS_FOR_RESULT
        )

        response = {
            "status": "ok",
            "session_id": session_id,

            # Current chunk
            "chunk_index": chunk_index,
            "prediction": prediction,
            "fake_probability": round(
                fake_prob,
                4
            ),
            "confidence": round(
                confidence,
                4
            ),

            # Session progress
            "elapsed_time": elapsed_time,
            "total_chunks": total_chunks,

            # IMPORTANT:
            # No final result before 20 seconds.
            "final_result_available": (
                final_result_available
            )
        }

        # ----------------------------------------------------
        # Before 20 seconds
        # ----------------------------------------------------

        if not final_result_available:

            response.update({
                "final_prediction": None,
                "final_confidence": None,
                "real_chunks": None,
                "fake_chunks": None,
                "mean_fake_probability": None,
                "message": (
                    "Final result will be available "
                    "after 20 seconds."
                )
            })

            return jsonify(response), 200

        # ----------------------------------------------------
        # After 20 seconds:
        # CUMULATIVE majority voting
        # ----------------------------------------------------

        predictions = session["predictions"]

        final_prediction = cumulative_majority_vote(
            predictions
        )

        real_chunks = sum(
            1
            for item in predictions
            if item["prediction"] == "REAL"
        )

        fake_chunks = sum(
            1
            for item in predictions
            if item["prediction"] == "FAKE"
        )

        mean_fake_probability = float(
            np.mean([
                item["fake_probability"]
                for item in predictions
            ])
        )

        if final_prediction == "FAKE":
            final_confidence = mean_fake_probability
        else:
            final_confidence = (
                1.0 - mean_fake_probability
            )

        response.update({
            "final_prediction": final_prediction,
            "final_confidence": round(
                final_confidence,
                4
            ),
            "real_chunks": real_chunks,
            "fake_chunks": fake_chunks,
            "mean_fake_probability": round(
                mean_fake_probability,
                4
            ),
            "voting_scope": (
                f"all {total_chunks} chunks "
                f"from session start"
            )
        })

        return jsonify(response), 200

    except Exception as e:

        logger.exception(
            "Error predicting live chunk"
        )

        return jsonify({
            "error": str(e)
        }), 500


# ============================================================
# Stop live session
# ============================================================

@live_bp.route("/stop", methods=["POST"])
def live_stop():
    """
    Stop a live session and clear all accumulated
    predictions.
    """

    data = request.get_json(silent=True) or {}

    session_id = data.get("session_id")

    if not session_id:
        return jsonify({
            "error": "session_id is required"
        }), 400

    existed = session_id in session_data

    if existed:
        del session_data[session_id]

    logger.info(
        "Stopped live session: %s",
        session_id
    )

    return jsonify({
        "status": "stopped",
        "session_id": session_id,
        "session_cleared": existed
    }), 200


# ============================================================
# Diagnostic logging
# ============================================================

def _log_chunk_diagnostics(
    session_id, chunk_index, audio_data, sample_rate,
    stage="received"
):
    """Log detailed signal statistics for a live audio chunk.

    This aids debugging by showing exact waveform characteristics
    at each stage of the pipeline.
    """

    duration = len(audio_data) / sample_rate if sample_rate > 0 else 0
    finite_count = int(np.sum(np.isfinite(audio_data)))
    min_val = float(np.min(audio_data))
    max_val = float(np.max(audio_data))
    mean_val = float(np.mean(audio_data))
    std_val = float(np.std(audio_data))
    rms = float(np.sqrt(np.mean(audio_data ** 2)))
    peak = float(np.max(np.abs(audio_data)))
    zero_count = int(np.sum(audio_data == 0))

    logger.info(
        "[LIVE DIAG] session=%s chunk=%d stage=%s | "
        "samples=%d sr=%d duration=%.3fs | "
        "finite=%d zeros=%d | "
        "min=%.6f max=%.6f mean=%.6f std=%.6f | "
        "rms=%.6f peak=%.6f",
        session_id, chunk_index, stage,
        len(audio_data), sample_rate, duration,
        finite_count, zero_count,
        min_val, max_val, mean_val, std_val,
        rms, peak
    )


# ============================================================
# Classification
# ============================================================

def classify_probability(fake_probability):
    """
    Convert FAKE probability to class.

    >= 0.30 -> FAKE
    <  0.30 -> REAL
    """

    return (
        "FAKE"
        if fake_probability >= CLASSIFICATION_THRESHOLD
        else "REAL"
    )


def calculate_confidence(
    prediction,
    fake_probability
):
    """
    Confidence of the predicted class.
    """

    if prediction == "FAKE":
        return fake_probability

    return 1.0 - fake_probability


# ============================================================
# Cumulative majority voting
# ============================================================

def cumulative_majority_vote(
    predictions
):
    """
    Perform majority voting across EVERY chunk
    accumulated during the current session.

    This is intentionally NOT a rolling window.

    Example:

        5 chunks  -> vote 1-5
        6 chunks  -> vote 1-6
        7 chunks  -> vote 1-7
        8 chunks  -> vote 1-8
    """

    if not predictions:
        raise ValueError(
            "No predictions available"
        )

    real_count = sum(
        1
        for item in predictions
        if item["prediction"] == "REAL"
    )

    fake_count = sum(
        1
        for item in predictions
        if item["prediction"] == "FAKE"
    )

    # --------------------------------------------------------
    # Normal majority
    # --------------------------------------------------------

    if fake_count > real_count:
        return "FAKE"

    if real_count > fake_count:
        return "REAL"

    # --------------------------------------------------------
    # Tie-break using mean FAKE probability
    # --------------------------------------------------------

    mean_fake_probability = float(
        np.mean([
            item["fake_probability"]
            for item in predictions
        ])
    )

    return (
        "FAKE"
        if mean_fake_probability >= CLASSIFICATION_THRESHOLD
        else "REAL"
    )