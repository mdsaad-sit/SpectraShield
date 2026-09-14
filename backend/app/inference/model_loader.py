import os
import logging
import torch

from app.config import Config
from .model import SpectraShieldCNN

logger = logging.getLogger(__name__)


class ModelLoader:
    _model = None
    _model_loaded = False
    _device = None
    _error = None

    @classmethod
    def initialize(cls):
        if cls._model_loaded:
            return cls._model, cls._device

        cls._device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        model_path = os.path.abspath(Config.MODEL_PATH)

        print(f"[SpectraShield] Model path: {model_path}")
        print(f"[SpectraShield] Model exists: {os.path.exists(model_path)}")
        print(f"[SpectraShield] Device: {cls._device}")

        if not os.path.exists(model_path):
            cls._error = f"Model file not found: {model_path}"
            print(f"[SpectraShield] ERROR: {cls._error}")
            return None, cls._device

        try:
            print("[SpectraShield] Creating model...")
            model = SpectraShieldCNN()

            print("[SpectraShield] Loading checkpoint...")
            checkpoint = torch.load(
                model_path,
                map_location=cls._device,
                weights_only=False
            )

            print(f"[SpectraShield] Checkpoint type: {type(checkpoint)}")

            # Handle common checkpoint formats
            if isinstance(checkpoint, dict):
                if "state_dict" in checkpoint:
                    state_dict = checkpoint["state_dict"]
                elif "model_state_dict" in checkpoint:
                    state_dict = checkpoint["model_state_dict"]
                else:
                    state_dict = checkpoint
            else:
                state_dict = checkpoint

            print("[SpectraShield] Loading state dictionary...")
            model.load_state_dict(state_dict, strict=True)

            model.to(cls._device)
            model.eval()

            cls._model = model
            cls._model_loaded = True
            cls._error = None

            print(
                f"[SpectraShield] MODEL LOADED SUCCESSFULLY "
                f"ON {cls._device}"
            )

        except Exception as e:
            cls._model = None
            cls._model_loaded = False
            cls._error = f"{type(e).__name__}: {str(e)}"

            print(
                f"[SpectraShield] MODEL LOAD FAILED: "
                f"{cls._error}"
            )

            logger.exception(
                "Failed to load SpectraShield model"
            )

        return cls._model, cls._device

    @classmethod
    def is_loaded(cls):
        return cls._model_loaded

    @classmethod
    def get_device(cls):
        return cls._device

    @classmethod
    def get_error(cls):
        return cls._error

    @classmethod
    def predict(cls, input_tensor):
        if not cls._model_loaded or cls._model is None:
            raise RuntimeError(
                f"Model is not loaded. {cls._error or ''}"
            )

        with torch.no_grad():
            input_tensor = input_tensor.to(cls._device)
            logit = cls._model(input_tensor)
            probability = torch.sigmoid(logit).item()

        return logit, probability

    @classmethod
    def get_model(cls):
        """
        Return the already-loaded SpectraShield model.

        Used by XAI methods such as Grad-CAM.
        Unlike predict(), this method does not use
        torch.no_grad(), because Grad-CAM requires gradients.
        """
        if not cls._model_loaded or cls._model is None:
            raise RuntimeError(
                f"Model is not loaded. {cls._error or ''}"
            )

        return cls._model