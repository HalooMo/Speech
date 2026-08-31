"""Секреты и параметры LLM/HF из config/.env.

Загружается при импорте; main.py и server/run_job.py дополнительно
вызывают load_dotenv для override из config/.env.
"""
import os
from pathlib import Path

from dotenv import load_dotenv

_root = Path(__file__).resolve().parent.parent
load_dotenv(_root / ".env", override=False)
load_dotenv(Path(__file__).resolve().parent / ".env", override=False)


def get_hf_token():
    """Hugging Face — pyannote, wav2vec2 и др."""
    t = os.environ.get("HF_TOKEN", "").strip()
    if not t:
        raise ValueError("HF_TOKEN не задан в config/.env")
    return t


def get_openai_api_key():
    """OpenAI-compatible API — сегментация и перевод через LLM."""
    k = os.environ.get("OPENAI_API_KEY", "").strip()
    if not k:
        raise ValueError("OPENAI_API_KEY не задан в config/.env")
    return k


def get_openai_base_url():
    return os.environ.get("OPENAI_BASE_URL", "https://api.agentplatform.ru/v1").strip()


def get_openai_model():
    return os.environ.get("OPENAI_MODEL", "openai/gpt-5.5").strip()


def get_fish_tts_api_key():
    """Fish Audio TTS — клонирование и озвучка (FISH_TTS_API_KEY)."""
    k = (
        os.environ.get("FISH_TTS_API_KEY", "").strip()
        or os.environ.get("FISH_API_KEY", "").strip()
    )
    if not k:
        raise ValueError("FISH_TTS_API_KEY не задан в config/.env")
    return k
