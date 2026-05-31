# type==1: value = word_segments (list[dict])
# type==2: value = dict с text, source_lang, target_lang, source_chars, source_words, slot_sec

import json


def get_prompt(type, value):
    if type == 2:
        if isinstance(value, dict):
            text = value["text"]
            source_lang = value["source_lang"]
            target_lang = value["target_lang"]
            src_chars = value.get("source_chars", len(text))
            src_words = value.get("source_words", len(text.split()))
            slot_sec = value.get("slot_sec")
        else:
            text, source_lang, target_lang = value[0], value[1], value[2]
            src_chars = len(text)
            src_words = len(text.split())
            slot_sec = None
        lo_c = max(1, int(src_chars * 0.9))
        hi_c = max(lo_c + 1, int(src_chars * 1.1))
        lo_w = max(1, int(src_words * 0.85))
        hi_w = max(lo_w, int(src_words * 1.15))
        slot_line = ""
        if slot_sec is not None and float(slot_sec) > 0:
            slot_line = (
                f"\nThe dubbed line must fit ~{float(slot_sec):.2f}s of speech "
                f"(±5% timing handled later; match length in {target_lang})."
            )
        return f"""
You are a voice-over dubbing studio translator.
Translate the line from "{source_lang}" to "{target_lang}".
Length: about {lo_c}–{hi_c} characters and {lo_w}–{hi_w} words (same as original).
If longer — shorten; if shorter — add a little detail, no filler.
Preserve meaning, tone, humor; natural conversational dubbing.{slot_line}
Forbidden: explanations, director's notes, speaker names, quotes around the answer, markdown.
Original:
{text}

Your answer must be only the translated text of the single line.
""".strip()

    if type == 1:
        word_segments_json = json.dumps(value, ensure_ascii=False, separators=(",", ":"))
        return f"""
You segment speech for dubbing.
Main task: split utterances by speaker (**type: "speech"**) and fill silent gaps with "silence" (**type: "silence"**). Use context, not only diarization labels.
Input: word_segments — list with `word`, `start`, `end`, `speaker`.
{word_segments_json}
1. Sort words by `start`.
2. Identify speakers by context ("A", "B", "C", …). `speaker` is a hint only.
3. Each "speech" segment = one speaker, 1–2 sentences.
4. Gap between speech → "silence": {{"type": "silence", "start": <end prev>, "end": <start next>}}
5. Do not add or alter words; each word in one segment only.
6. Timing: min(start)/max(end) of included words.
Output: JSON array:
- speech: {{"type": "speech", "start": ..., "end": ..., "speaker": "...", "text": "..."}}
- silence: {{"type": "silence", "start": ..., "end": ...}}
""".strip()
    return None
