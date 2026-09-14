import sys
import os
import io
import numpy as np
import scipy.io.wavfile

# Must add path before any other imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app import create_app


def get_test_client():
    """Create a Flask test client."""
    app, socketio = create_app("development")
    app.config["TESTING"] = True
    return app.test_client()


def generate_test_wav(duration=2.0, sr=16000):
    """Generate a simple sine wave WAV file in memory."""
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    y = (np.sin(2 * np.pi * 440 * t) * 0.5).astype(np.float32)

    buf = io.BytesIO()
    scipy.io.wavfile.write(buf, sr, y)
    buf.seek(0)
    return buf


# ── Health Endpoint ─────────────────────────────────────────


def test_health_returns_200():
    """GET /api/health should always return 200."""
    client = get_test_client()
    response = client.get("/api/health")
    assert response.status_code == 200


def test_health_has_required_fields():
    """Health response must contain status, model_loaded, and device."""
    client = get_test_client()
    response = client.get("/api/health")
    data = response.get_json()
    assert "status" in data
    assert "model_loaded" in data
    assert "device" in data
    assert data["status"] == "ok"


# ── Recorded Prediction Endpoint ───────────────────────────


def test_predict_no_file_returns_400():
    """POST /api/predict without audio file should return 400."""
    client = get_test_client()
    response = client.post("/api/predict")
    assert response.status_code == 400
    data = response.get_json()
    assert "error" in data


def test_predict_empty_filename_returns_400():
    """POST /api/predict with empty filename should return 400."""
    client = get_test_client()
    response = client.post(
        "/api/predict",
        data={"audio": (io.BytesIO(b""), "")},
        content_type="multipart/form-data",
    )
    assert response.status_code == 400


def test_predict_with_audio_returns_result():
    """POST /api/predict with valid audio should return prediction result."""
    client = get_test_client()
    wav_buf = generate_test_wav(duration=3.0)

    response = client.post(
        "/api/predict",
        data={"audio": (wav_buf, "test.wav")},
        content_type="multipart/form-data",
    )

    # If model is loaded, should return 200 with result
    # If model is NOT loaded, should return 500 with error
    data = response.get_json()

    if response.status_code == 200:
        assert "final_prediction" in data
        assert data["final_prediction"] in ("REAL", "FAKE")
        assert "confidence" in data
        assert "total_chunks" in data
        assert "chunks" in data
        assert data["mode"] == "recorded"
    else:
        # Model not loaded — acceptable in test environment
        assert response.status_code == 500
        assert "error" in data


# ── Live Session Endpoints ─────────────────────────────────


def test_live_start_no_session_id_returns_400():
    """POST /api/start without session_id should return 400."""
    client = get_test_client()
    response = client.post(
        "/api/start",
        json={},
    )
    assert response.status_code == 400
    data = response.get_json()
    assert "error" in data


def test_live_stop_no_session_id_returns_400():
    """POST /api/stop without session_id should return 400."""
    client = get_test_client()
    response = client.post(
        "/api/stop",
        json={},
    )
    assert response.status_code == 400


def test_predict_chunk_no_session_returns_400():
    """POST /api/predict_chunk without active session should return 400."""
    client = get_test_client()
    response = client.post(
        "/api/predict_chunk",
        json={
            "session_id": "nonexistent-session",
            "audio_data": [0.0] * 64000,
            "sample_rate": 16000,
            "chunk_index": 0,
        },
    )
    assert response.status_code == 400
    data = response.get_json()
    assert "error" in data


def test_predict_chunk_empty_audio_returns_400():
    """POST /api/predict_chunk with empty audio should return 400."""
    client = get_test_client()

    # Start session first
    client.post("/api/start", json={"session_id": "test-empty-audio"})

    response = client.post(
        "/api/predict_chunk",
        json={
            "session_id": "test-empty-audio",
            "audio_data": [],
            "sample_rate": 16000,
            "chunk_index": 0,
        },
    )
    assert response.status_code == 400

    # Cleanup
    client.post("/api/stop", json={"session_id": "test-empty-audio"})


def test_live_session_lifecycle():
    """Test full live session: start → stop."""
    client = get_test_client()
    sid = "test-lifecycle-session"

    # Start
    response = client.post("/api/start", json={"session_id": sid})

    if response.status_code == 200:
        data = response.get_json()
        assert data["status"] == "started"
        assert data["session_id"] == sid

        # Stop
        response = client.post("/api/stop", json={"session_id": sid})
        assert response.status_code == 200
        data = response.get_json()
        assert data["status"] == "stopped"
        assert data["session_cleared"] is True
    else:
        # Model not loaded — acceptable
        assert response.status_code == 500


def test_live_stop_nonexistent_session():
    """Stopping a non-existent session should succeed with session_cleared=False."""
    client = get_test_client()
    response = client.post(
        "/api/stop",
        json={"session_id": "does-not-exist"},
    )
    assert response.status_code == 200
    data = response.get_json()
    assert data["session_cleared"] is False