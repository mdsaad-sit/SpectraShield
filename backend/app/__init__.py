from .config import config_by_name


def create_app(config_name="development"):
    from flask import Flask
    from flask_cors import CORS
    from flask_socketio import SocketIO

    app = Flask(__name__)

    # Load application configuration
    config_class = config_by_name.get(
        config_name,
        config_by_name["development"]
    )
    app.config.from_object(config_class)

    # CORS configuration
    CORS(
        app,
        origins=[app.config["FRONTEND_URL"]]
    )

    # Socket.IO configuration
    socketio = SocketIO(
        app,
        cors_allowed_origins=[app.config["FRONTEND_URL"]]
    )

    # Register blueprints
    from .routes.health import health_bp
    from .routes.recorded import recorded_bp
    from .routes.live import live_bp

    app.register_blueprint(
        health_bp,
        url_prefix="/api"
    )

    app.register_blueprint(
        recorded_bp,
        url_prefix="/api"
    )

    app.register_blueprint(
        live_bp,
        url_prefix="/api"
    )

    # Initialize SpectraShield trained model
    from .inference.model_loader import ModelLoader

    ModelLoader.initialize()

    # Disable browser/proxy caching for API responses
    @app.after_request
    def add_cors(resp):
        resp.headers["Cache-Control"] = (
            "no-cache, no-store, must-revalidate"
        )
        resp.headers["Pragma"] = "no-cache"
        resp.headers["Expires"] = "0"
        return resp

    return app, socketio