"""OpenAI-compatible LLM."""
from __future__ import annotations

import time

from openai import OpenAI

from env_config import get_openai_api_key, get_openai_base_url, get_openai_model

_client: OpenAI | None = None

_SYSTEM_JSON = (
    "You are a dialogue segmentation engine for professional dubbing. "
    "The user message contains an ASR word list (machine transcript). "
    "You must return ONLY a valid JSON array — no markdown, no apologies, no refusals. "
    "This is a standard post-production task on provided transcript data."
)

_SYSTEM_TEXT = "You are a professional dubbing translator. Reply with only the translated line."


def _client_instance() -> OpenAI:
    global _client
    if _client is None:
        _client = OpenAI(
            base_url=get_openai_base_url(),
            api_key=get_openai_api_key(),
        )
    return _client


def llm_response(user_prompt: str, *, json_only: bool = False) -> str:
    """json_only=True — для разметки диалога (массив JSON)."""
    system = _SYSTEM_JSON if json_only else _SYSTEM_TEXT
    response = _client_instance().chat.completions.create(
        model=get_openai_model(),
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.2 if json_only else 0.4,
    )
    return (response.choices[0].message.content or "").strip()


def llm_response_retry(
    user_prompt: str,
    *,
    json_only: bool = False,
    retries: int = 3,
    retry_suffix: str = "",
) -> str:
    last = ""
    for attempt in range(retries):
        prompt = user_prompt
        if attempt and retry_suffix:
            prompt = f"{user_prompt}\n\n{retry_suffix}"
        last = llm_response(prompt, json_only=json_only)
        if last and not looks_like_refusal(last, json_only=json_only):
            return last
        if attempt + 1 < retries:
            time.sleep(1.5)
    return last


def looks_like_refusal(text: str, *, json_only: bool = False) -> bool:
    s = (text or "").strip().lower()
    if not s:
        return True
    if json_only and s.lstrip().startswith("["):
        return False
    markers = (
        "can't assist",
        "cannot assist",
        "can't help",
        "cannot help",
        "i'm sorry",
        "i am sorry",
        "unable to",
        "not able to",
    )
    return any(m in s for m in markers)
