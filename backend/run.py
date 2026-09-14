from app import create_app

app, socketio = create_app()

if __name__ == "__main__":
    host = "127.0.0.1"
    port = 5000
    debug = True

    socketio.run(
        app,
        host=host,
        port=port,
        debug=debug,
        allow_unsafe_werkzeug=True
    )