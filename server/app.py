"""Flask-приложение SpeechLab.

Тонкий HTTP-слой: принимает запросы, ставит задачи в очередь, отдаёт статусы.
Тяжёлый GPU-пайплайн выполняется в отдельном subprocess (server/run_job.py).
"""
from __future__ import annotations

from flask import Flask, jsonify, request

from server.config import ServerConfig
from server.jobs import JobStore
from server.routes import bp


def create_app(config: ServerConfig | None = None) -> Flask:
    # --- Инициализация: конфиг, каталоги, хранилище задач ---
    cfg = config or ServerConfig()
    cfg.validate()
    cfg.ensure_dirs()

    app = Flask(__name__)
    app.config["MAX_CONTENT_LENGTH"] = cfg.max_content_length
    # extensions — общий доступ routes → config и JobStore без глобальных переменных
    app.extensions["server_config"] = cfg
    app.extensions["job_store"] = JobStore(cfg.jobs_dir)

    # --- Авторизация: все эндпоинты кроме /health требуют X-API-Key или Bearer ---
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

    # --- Ошибка превышения лимита загрузки (nginx + Flask MAX_CONTENT_LENGTH) ---
    @app.errorhandler(413)
    def _too_large(_exc):
        return jsonify({"error": "Файл слишком большой (SPEECHLAB_MAX_UPLOAD_MB)"}), 413

    # REST-маршруты: /health, /api/v1/dub, /api/v1/jobs/*
    app.register_blueprint(bp)
    return app
