"""Gunicorn: gunicorn -c deploy/gunicorn.conf.py wsgi:app

workers=1 — один GPU-пайплайн одновременно.
timeout=0 — HTTP только ставит job в очередь; долгая работа в subprocess.
"""
import os
from pathlib import Path

_root = Path(__file__).resolve().parent.parent
bind = os.environ.get("SPEECHLAB_GUNICORN_BIND", "127.0.0.1:8080")
workers = 1  # один GPU-пайплайн; больше — только если несколько GPU
threads = 4
timeout = 0  # долгие задачи в subprocess, HTTP только ставит в очередь
graceful_timeout = 30
keepalive = 5
accesslog = "-"
errorlog = "-"
capture_output = True
chdir = str(_root)
raw_env = [f"SPEECHLAB_ENV={os.environ.get('SPEECHLAB_ENV', 'production')}"]
