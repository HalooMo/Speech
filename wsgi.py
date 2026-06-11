"""Gunicorn: gunicorn -w 1 -b 0.0.0.0:8443 --certfile=cert.pem --keyfile=key.pem wsgi:app"""
from pathlib import Path

from dotenv import load_dotenv

_root = Path(__file__).resolve().parent
load_dotenv(_root / ".env", override=False)
load_dotenv(_root / "config" / ".env", override=False)

from server.app import create_app

app = create_app()
