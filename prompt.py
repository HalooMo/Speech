# type==1: value = word_segments (list[dict])
# type==2: value = [text, source_lang, target_lang] или dict с теми же ключами

import json


def get_prompt(type, value):
    if type == 2:
        if isinstance(value, dict):
            text = value["text"]
            source_lang = value["source_lang"]
            target_lang = value["target_lang"]
            src_chars = value.get("source_chars", len(text))
            src_words = value.get("source_words", len(text.split()))
        else:
            text, source_lang, target_lang = value[0], value[1], value[2]
        lo_c = max(1, int(src_chars * 0.9))
        hi_c = max(lo_c + 1, int(src_chars * 1.1))
        lo_w = max(1, int(src_words * 0.85))
        hi_w = max(lo_w, int(src_words * 1.15))
        return f'''
You are a voice-over dubbing studio translator.
Translate the line from "{source_lang}" to "{target_lang}".
Main requirement — the **length should be approximately the same as the original**:
- If your translation is longer — shorten it; if shorter — elaborate a bit, but avoid unnecessary filler.
- Preserve the meaning and tone; style should be natural conversational dubbing, and try to keep the original tone and intonation, as well as any humor present.
Forbidden: explanations, director's notes, speaker names, quotation marks around the answer, markdown.
Original:
{text}

Your answer must be only the translated text of the single line.
'''.strip()

    if type == 1:
        word_segments_json = json.dumps(value, ensure_ascii=False, indent=2)
        return f'''
You segment speech for dubbing. 
Main task: correctly split utterances by speaker (**type: "speech"**) and fill silent gaps between them with "silence" segments (**type: "silence"**). Base decisions on meaning and context, not blindly on speaker labels.
Input: word_segments — list of words with `word`, `start`, `end`, `speaker`.
{word_segments_json}
1. Sort words by `start` time.
2. Identify speakers by context ("A", "B", "C", etc). Use `speaker` label as a hint only.
3. Each "speech" segment = one speaker, 1–2 sentences.
4. If there's a gap (silence) between speech segments, insert a "silence" segment:
   {"type": "silence", "start": <end of previous>, "end": <start of next>}
5. Preserve order, do not add or alter words, use each word only in one segment.
6. Segment timing: min(start)/max(end) of included words.
Output: JSON array of segments:
- speech: {"type": "speech", "start": ..., "end": ..., "speaker": "...", "text": "..."}
- silence: {"type": "silence", "start": ..., "end": ...}
'''.strip()
    return None
