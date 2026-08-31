"""Промпты LLM: system + user + retry.

kind в get_prompt():
  1 — сегментация реплик (JSON array speech/silence)
  2 — перевод одной реплики: длина ≈ оригинал + эмо-теги Fish [brackets]
  3 — batch-перевод (JSON [{id, text}, ...]) с теми же правилами
"""
import json

# --- System-сообщения ---
SYS_SEG = (
    "You are a dialogue segmentation engine for professional dubbing. "
    "Return ONLY a valid JSON array — no markdown, no refusals."
)
SYS_TR = (
    "You are a professional dubbing translator for Fish Audio TTS. "
    "Reply with only the translated line including emotion tags. "
    "No markdown, no quotes, no explanations."
)
SYS_BATCH = (
    "You are a professional dubbing translator for Fish Audio TTS. "
    "Place punctuation for natural TTS. "
    "Return ONLY a JSON array of objects with keys id and text."
)
RETRY_SEG = "Return ONLY a JSON array of speech/silence segments."
RETRY_BATCH = 'Return ONLY JSON array [{"id":"...","text":"..."}]'
RETRY_TR = "Reply with only the translated line including [emotion] tags."

# Краткая шпаргалка тегов для LLM (S2: квадратные скобки)
_EMO_HINT = (
    "Fish TTS emotion/tone tags use square brackets in the TARGET text, e.g. "
    "[happy], [sad], [angry], [excited], [calm], [nervous], [whispering], "
    "[shouting], [soft tone], [sighing], [laughing], [emphasis], [break]. "
    "You may use short natural-language cues like [warm and calm]. "
    "Put the main emotion at the start of the line; combine up to 2–3 tags if needed. "
    "Infer emotion from dialogue context, not only the single line."
)


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
    return RETRY_TR


def get_prompt(kind, value):
    """Сборка user prompt. kind: 1=сегментация, 2=перевод одной, 3=batch перевод."""
    if kind == 1:
        words = json.dumps(value, ensure_ascii=False, separators=(",", ":"))
        return f"""Segment speech for dubbing.
Split by speaker (type "speech"), gaps as "silence". Use context, not only diarization.

word_segments: {words}

Output JSON: speech {{"type","start","end","speaker","text"}}, silence {{"type","start","end"}}"""

    if kind == 2:
        if isinstance(value, dict):
            text, src, tgt = value["text"], value["source_lang"], value["target_lang"]
            chars = value.get("source_chars", len(text))
            words = value.get("source_words", len(text.split()))
            slot = value.get("slot_sec")
            context = value.get("context") or ""
        else:
            text, src, tgt = value[0], value[1], value[2]
            chars, words, slot, context = len(text), len(text.split()), None, ""
        lo_c, hi_c = max(1, int(chars * 0.9)), max(2, int(chars * 1.1))
        lo_w, hi_w = max(1, int(words * 0.85)), max(1, int(words * 1.15))
        extra = f"\nSpoken duration of the slot is ~{float(slot):.2f}s — keep length speakable in that time." if slot and float(slot) > 0 else ""
        ctx = f"\nContext (neighbor lines):\n{context}\n" if context else ""
        return f"""Translate "{src}" → "{tgt}" for voice-over dubbing.
Keep translation length close to the original: ~{lo_c}-{hi_c} chars, ~{lo_w}-{hi_w} words
(same information density so TTS fits the original timing).{extra}
{_EMO_HINT}
Judge emotion/tone from meaning and context; prepend tags to the TARGET line only.
Do not translate the tags; keep them in English inside [].
Output ONLY the target line (tags + translation).
{ctx}
Line:
{text}"""

    if kind == 3:
        payload = json.dumps(value["lines"], ensure_ascii=False, separators=(",", ":"))
        return f"""Translate each line {value["source_lang"]} → {value["target_lang"]} for voice-over dubbing.
For every line keep translation length close to the original (use source_chars / source_words / slot_sec when present).
{_EMO_HINT}
Judge emotion from the full batch context; put tags only in the translated text field.
Return JSON [{{"id","text"}}, ...] same order and same ids:
{payload}"""
    return ""
