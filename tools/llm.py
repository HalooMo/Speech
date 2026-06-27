"""OpenAI-compatible LLM (промпты в prompt.py).

Используется для: сегментации реплик, перевода (одиночного и batch).
Ключи и base_url — из config/.env (OPENAI_*).
"""
import time

import prompt
from config.env_config import get_openai_api_key, get_openai_base_url, get_openai_model

_client = None


def _openai():
    """Ленивый клиент OpenAI — один экземпляр на процесс."""
    global _client
    if _client is None:
        from openai import OpenAI
        _client = OpenAI(base_url=get_openai_base_url(), api_key=get_openai_api_key())
    return _client


def llm_response(user_prompt, json_only=False, batch_translate=False):
    """Один запрос к LLM; temperature ниже для JSON (сегментация)."""
    system = prompt.get_system(json_only=json_only, batch_translate=batch_translate)
    temp = 0.2 if json_only else 0.4
    r = _openai().chat.completions.create(
        model=get_openai_model(),
        messages=[{"role": "system", "content": system}, {"role": "user", "content": user_prompt}],
        temperature=temp,
    )
    return (r.choices[0].message.content or "").strip()


def looks_like_refusal(text, json_only=False):
    """Детект отказа LLM или пустого ответа — триггер для retry."""
    s = (text or "").strip().lower()
    if not s:
        return True
    if json_only and s.lstrip().startswith("["):
        return False
    for m in ("can't assist", "cannot assist", "i'm sorry", "unable to"):
        if m in s:
            return True
    return False


def llm_response_retry(user_prompt, json_only=False, batch_translate=False, retries=3, retry_suffix=""):
    """Повтор при отказе или пустом ответе; к prompt добавляется retry_suffix."""
    suffix = retry_suffix or prompt.get_retry_suffix(json_only=json_only, batch_translate=batch_translate)
    last = ""
    for i in range(retries):
        body = f"{user_prompt}\n\n{suffix}" if i and suffix else user_prompt
        last = llm_response(body, json_only=json_only, batch_translate=batch_translate)
        if last and not looks_like_refusal(last, json_only=json_only):
            return last
        if i + 1 < retries:
            time.sleep(1.5)
    return last
