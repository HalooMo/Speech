"""Gunicorn entrypoint: gunicorn -c deploy/gunicorn.conf.py wsgi:app

Загружает config/.env до импорта Flask-приложения.
В production Gunicorn слушает 127.0.0.1:8080, снаружи — nginx + TLS.
"""
from pathlib import Path

from dotenv import load_dotenv

_root = Path(__file__).resolve().parent
load_dotenv(_root / ".env", override=False)
# config/.env — основной файл секретов; перекрывает корневой .env
load_dotenv(_root / "config" / ".env", override=True)

from server.app import create_app

app = create_app()
