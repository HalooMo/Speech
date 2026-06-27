"""Промпты LLM: system + user + retry.

kind в get_prompt():
  1 — сегментация реплик (JSON array speech/silence)
  2 — перевод одной реплики с ограничением длины под слот
  3 — batch-перевод (JSON [{id, text}, ...])
"""
import json

# --- System-сообщения: задают формат ответа LLM (JSON vs plain text) ---
SYS_SEG = (
    "You are a dialogue segmentation engine for professional dubbing. "
    "Return ONLY a valid JSON array — no markdown, no refusals."
)
SYS_TR = "You are a professional dubbing translator. Reply with only the translated line."
SYS_BATCH = (
    "You are a professional dubbing translator."
    "Place punctuation marks according to the context of the sentence and the entire text, this is necessary for high-quality voicing of this text."
    "Return ONLY a JSON array of objects with keys id and text."
)
RETRY_SEG = "Return ONLY a JSON array of speech/silence segments."
RETRY_BATCH = 'Return ONLY JSON array [{"id":"...","text":"..."}]'


def get_system(json_only=False, batch_translate=False):
    """System prompt по типу задачи."""
    if batch_translate:
        return SYS_BATCH
    if json_only:
        return SYS_SEG
    return SYS_TR


def get_retry_suffix(json_only=False, batch_translate=False):
    """Дополнение к user prompt при повторе (llm_response_retry)."""
    if batch_translate:
        return RETRY_BATCH
    if json_only:
        return RETRY_SEG
    return ""


def get_prompt(kind, value):
    """Сборка user prompt. kind: 1=сегментация, 2=перевод одной, 3=batch перевод."""
    # --- kind 1: LLM размечает реплики поверх word_segments от WhisperX/pyannote ---
    if kind == 1:
        words = json.dumps(value, ensure_ascii=False, separators=(",", ":"))
        return f"""Segment speech for dubbing.
Split by speaker (type "speech"), gaps as "silence". Use context, not only diarization.

word_segments: {words}

Output JSON: speech {{"type","start","end","speaker","text"}}, silence {{"type","start","end"}}"""
    # --- kind 2: перевод одной реплики с лимитом символов/слов под длительность слота ---
    if kind == 2:
        if isinstance(value, dict):
            text, src, tgt = value["text"], value["source_lang"], value["target_lang"]
            chars, words = value.get("source_chars", len(text)), value.get("source_words", len(text.split()))
            slot = value.get("slot_sec")
        else:
            text, src, tgt = value[0], value[1], value[2]
            chars, words, slot = len(text), len(text.split()), None
        lo_c, hi_c = max(1, int(chars * 0.9)), max(2, int(chars * 1.1))
        lo_w, hi_w = max(1, int(words * 0.85)), max(1, int(words * 1.15))
        extra = f"\nFit ~{float(slot):.2f}s speech in {tgt}." if slot and float(slot) > 0 else ""
        return f"""Translate "{src}" → "{tgt}". Length ~{lo_c}-{hi_c} chars, {lo_w}-{hi_w} words.{extra}

{text}"""
    # --- kind 3: batch-перевод — один запрос LLM на N реплик (TRANSLATE_BATCH_SIZE) ---
    if kind == 3:
        payload = json.dumps(value["lines"], ensure_ascii=False, separators=(",", ":"))
        return f"""Translate each line {value["source_lang"]} → {value["target_lang"]}.
Return JSON [{{"id","text"}}, ...] same order:
{payload}"""
    return ""
