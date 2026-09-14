import os
from dotenv import load_dotenv

load_dotenv()

# backend/ directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
BACKEND_DIR = os.path.dirname(BASE_DIR)


class Config:
    SECRET_KEY = os.getenv("SECRET_KEY", "dev-secret-key")
    FRONTEND_URL = os.getenv("FRONTEND_URL", "http://localhost:5173")

    HOST = os.getenv("HOST", "127.0.0.1")
    PORT = int(os.getenv("PORT", 5000))

    # Model files are located at:
    # backend/model/best_model.pth
    # backend/model/model_config.json
    MODEL_PATH = os.getenv(
        "MODEL_PATH",
        os.path.join(BACKEND_DIR, "model", "best_model.pth")
    )

    MODEL_CONFIG_PATH = os.getenv(
        "MODEL_CONFIG_PATH",
        os.path.join(BACKEND_DIR, "model", "model_config.json")
    )

    MAX_UPLOAD_SIZE_MB = int(
        os.getenv("MAX_UPLOAD_SIZE_MB", 100)
    )

    # Locked SpectraShield threshold
    CLASSIFICATION_THRESHOLD = float(
        os.getenv("CLASSIFICATION_THRESHOLD", 0.30)
    )


class DevelopmentConfig(Config):
    FLASK_ENV = "development"
    FLASK_DEBUG = True


class ProductionConfig(Config):
    FLASK_ENV = "production"
    FLASK_DEBUG = False


config_by_name = {
    "development": DevelopmentConfig,
    "production": ProductionConfig,
}