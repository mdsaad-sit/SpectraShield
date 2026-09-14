from flask import Blueprint, jsonify
from app.inference.model_loader import ModelLoader

health_bp = Blueprint("health", __name__)


@health_bp.route("/health", methods=["GET"])
def health():
    """Health check endpoint.

    Returns:
        JSON with status, model_loaded, and device information.
    """
    model_loaded = ModelLoader.is_loaded()
    device = ModelLoader.get_device()

    return jsonify({
        "status": "ok",
        "model_loaded": model_loaded,
        "device": device.type if hasattr(device, "type") else str(device),
    })