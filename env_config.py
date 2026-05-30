"""Загрузка секретов из .env (корень репозитория)."""
from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

_ROOT = Path(__file__).resolve().parent
load_dotenv(_ROOT / ".env")


def get_hf_token() -> str:
    token = os.environ.get("HF_TOKEN", "").strip()
    if not token:
        raise ValueError(
            "HF_TOKEN не задан. Создайте .env из .env.example и укажите токен Hugging Face."
        )
    return token


def get_openai_api_key() -> str:
    key = os.environ.get("OPENAI_API_KEY", "").strip()
    if not key:
        raise ValueError(
            "OPENAI_API_KEY не задан. Создайте .env из .env.example."
        )
    return key


def get_openai_base_url() -> str:
    return os.environ.get("OPENAI_BASE_URL", "https://api.agentplatform.ru/v1").strip()


def get_openai_model() -> str:
    return os.environ.get("OPENAI_MODEL", "openai/gpt-5.5").strip()
