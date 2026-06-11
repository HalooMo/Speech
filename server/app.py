"""Flask-приложение SpeechLab."""
from __future__ import annotations

from flask import Flask, jsonify, request

from server.config import ServerConfig
from server.jobs import JobStore
from server.routes import bp


def create_app(config: ServerConfig | None = None) -> Flask:
    cfg = config or ServerConfig()
    cfg.validate()
    cfg.ensure_dirs()

    app = Flask(__name__)
    app.config["MAX_CONTENT_LENGTH"] = cfg.max_content_length
    app.extensions["server_config"] = cfg
    app.extensions["job_store"] = JobStore(cfg.jobs_dir)

    @app.before_request
    def _check_api_key():
        if request.endpoint in (None, "api.health"):
            return None
        if not cfg.require_api_key and not cfg.api_key:
            return None
        if not cfg.api_key:
            return jsonify({"error": "API key не настроен на сервере"}), 503
        key = request.headers.get("X-API-Key") or ""
        if key != cfg.api_key:
            bearer = request.headers.get("Authorization", "")
            if bearer.startswith("Bearer "):
                key = bearer[7:].strip()
        if key != cfg.api_key:
            return jsonify({"error": "Неверный или отсутствующий API key (X-API-Key)"}), 401
        return None

    @app.errorhandler(413)
    def _too_large(_exc):
        return jsonify({"error": "Файл слишком большой (SPEECHLAB_MAX_UPLOAD_MB)"}), 413

    app.register_blueprint(bp)
    return app
