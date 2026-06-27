"""ASR-модуль пайплайна (PRD шаг 3.1–3.2).

Имя test.py историческое: модуль используется из main.py на каждом первичном
сегменте; отдельно запускается как `python test.py` для отладки одного WAV.

Цепочка на один segment.wav (~40–90 с):
  pyannote (кто говорит) → WhisperX (слова + таймкоды) →
  LLM kind=1 (финальные реплики) → speech_*.wav + speech_*.txt
"""
import ast
import gc
import json
import os
import re
import shutil
from pathlib import Path

import pandas as pd
import prompt
import soundfile as sf
import torch
import torchaudio
import whisperx
from huggingface_hub import login
from pyannote.audio import Pipeline

from config.env_config import get_hf_token
from tools import llm

ROOT = Path(__file__).resolve().parent

# --- Настройки для локального `python test.py` (не используются из main) ---
WAV_PATH = ROOT / "vocals.wav"
SOURCE_LANG = "en"
OUTPUT_AUDIO_DIR = ROOT / "output_audio_segments"
OUTPUT_TEXT_DIR = ROOT / "output_text_segments"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
COMPUTE_TYPE = os.environ.get("SPEECHLAB_COMPUTE_TYPE", "float32")
WHISPER_MODEL = os.environ.get("SPEECHLAB_WHISPER_MODEL", "large-v3")
MAX_WORDS = int(os.environ.get("SPEECHLAB_MAX_WORDS_LLM", "2800"))

# Синглтон ASR: main вызывает init один раз, затем run_segment_pipeline × N сегментов
_asr = {"lang": None, "diarization": None, "whisper": None, "align_model": None, "align_meta": None}


def clear_dir(path):
    """Очистить папку перед повторной нарезкой (не при resume)."""
    path.mkdir(parents=True, exist_ok=True)
    for item in path.iterdir():
        if item.is_file() or item.is_symlink():
            item.unlink()
        elif item.is_dir():
            shutil.rmtree(item)


def parse_llm_segments(raw):
    """Разбор JSON-массива реплик из ответа LLM (снятие ```json, отказы)."""
    s = (raw or "").strip()
    if not s:
        raise ValueError("LLM: пустой ответ")
    if llm.looks_like_refusal(s, json_only=True):
        raise ValueError(f"LLM отказ: {s[:300]}")
    if s.startswith("```"):
        s = re.sub(r"^```(?:json)?\s*", "", s, flags=re.I)
        s = re.sub(r"\s*```\s*$", "", s).strip()
    i, j = s.find("["), s.rfind("]")
    if i != -1 and j > i:
        s = s[i : j + 1]
    try:
        data = json.loads(s)
    except json.JSONDecodeError:
        data = ast.literal_eval(s)
    if not isinstance(data, list):
        raise ValueError(f"Ожидался list, got {type(data).__name__}")
    return data


def cut_audio(audio, sr, start, end):
    """Вырезать фрагмент numpy-массива по секундам [start, end)."""
    n = len(audio) if audio.ndim == 1 else audio.shape[0]
    if end <= start:
        return audio[:0]
    a = max(0, min(n, int(round(start * sr))))
    b = max(a, min(n, int(round(end * sr))))
    return audio[a:b]


# =============================================================================
# Загрузка / выгрузка ASR-моделей (один раз на все первичные сегменты)
# =============================================================================
def init_asr_models(source_language, hf_token=None):
    """pyannote + Whisper + align-модель. Пропуск, если язык уже загружен."""
    token = hf_token or get_hf_token()
    lang = source_language.strip().lower()
    if _asr["lang"] == lang and _asr["whisper"] is not None:
        return
    unload_asr_models()
    login(token=token)
    print(f"  ASR: загрузка ({lang}, whisper={WHISPER_MODEL})…")
    _asr["lang"] = lang
    _asr["diarization"] = Pipeline.from_pretrained(
        "pyannote/speaker-diarization-community-1", token=token,
    ).to(torch.device(DEVICE))
    _asr["whisper"] = whisperx.load_model(
        WHISPER_MODEL, DEVICE, compute_type=COMPUTE_TYPE,
        language=source_language, vad_method="silero",
    )
    _asr["align_model"], _asr["align_meta"] = whisperx.load_align_model(
        language_code=source_language, device=DEVICE,
    )


def unload_asr_models():
    """Освободить VRAM перед casting/TTS (main вызывает после цикла сегментов)."""
    _asr.update(lang=None, diarization=None, whisper=None, align_model=None, align_meta=None)
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# =============================================================================
# Один первичный segment.wav → output_audio/text_segments
# =============================================================================
def run_segment_pipeline(audio_path, output_audio_dir, output_text_dir, source_language,
                         hf_token=None, clear_outputs=True, reuse_asr=True):
    """Полный ASR+LLM цикл для одного куска vocals (first_seg/.../segment.wav)."""
    token = hf_token or get_hf_token()
    audio_path = Path(audio_path)
    out_a, out_t = Path(output_audio_dir), Path(output_text_dir)
    if clear_outputs:
        clear_dir(out_a)
        clear_dir(out_t)
    else:
        out_a.mkdir(parents=True, exist_ok=True)
        out_t.mkdir(parents=True, exist_ok=True)

    if reuse_asr:
        init_asr_models(source_language, token)
    else:
        unload_asr_models()
        init_asr_models(source_language, token)

    # mono 16 kHz — формат пайплайна
    wave, sr = torchaudio.load(str(audio_path))
    if wave.shape[0] > 1:
        wave = wave.mean(dim=0, keepdim=True)
    if sr != 16000:
        wave = torchaudio.functional.resample(wave, sr, 16000)
        sr = 16000
    audio = wave.squeeze(0).numpy()

    # 1) Diarization: интервалы спикеров SPEAKER_00, SPEAKER_01, …
    diar = _asr["diarization"]({"waveform": wave, "sample_rate": sr})
    rows = []
    for seg, spk in diar.speaker_diarization.itertracks():
        rows.append({"start": max(0, seg.start), "end": min(wave.shape[1] / sr, seg.end), "speaker": spk})
    rows.sort(key=lambda x: x["start"])
    diarize_df = pd.DataFrame(rows)

    # 2) WhisperX: текст + word-level timestamps
    tr = _asr["whisper"].transcribe(audio, batch_size=4)
    tr = whisperx.align(tr["segments"], _asr["align_model"], _asr["align_meta"], audio, DEVICE, return_char_alignments=False)
    # 3) Привязка каждого слова к спикеру из pyannote
    words = whisperx.assign_word_speakers(diarize_df, tr)["word_segments"]

    if len(words) > MAX_WORDS:
        raise ValueError(f"Слишком много слов ({len(words)} > {MAX_WORDS})")

    # 4) LLM улучшает границы реплик и текст (prompt kind=1)
    raw = llm.llm_response_retry(prompt.get_prompt(1, words), json_only=True, retries=3)
    segments = [
        s for s in parse_llm_segments(raw)
        if s.get("type") == "speech" and (s.get("text") or "").strip()
        and float(s["end"]) > float(s["start"])
    ]
    segments.sort(key=lambda s: float(s["start"]))

    # 5) Нарезка WAV и запись speech_NNN.txt (время, спикер, текст)
    n = 0
    for seg in segments:
        chunk = cut_audio(audio, 16000, float(seg["start"]), float(seg["end"]))
        if len(chunk) == 0:
            continue
        n += 1
        sf.write(out_a / f"speech_{n:03d}_{seg['speaker']}_{seg['start']:.2f}-{seg['end']:.2f}s.wav", chunk, 16000)
        (out_t / f"speech_{n:03d}.txt").write_text(
            f"{seg['start']:.2f} - {seg['end']:.2f}\n{seg['speaker']}\n{seg['text']}", encoding="utf-8",
        )
    return segments


def run_test():
    """Локальная отладка: python test.py на WAV_PATH без полного main.py."""
    if not WAV_PATH.is_file():
        raise FileNotFoundError(WAV_PATH)
    try:
        segs = run_segment_pipeline(WAV_PATH, OUTPUT_AUDIO_DIR, OUTPUT_TEXT_DIR, SOURCE_LANG, reuse_asr=False)
    finally:
        unload_asr_models()
    print(f"Готово: {len(segs)} реплик")
    return segs


if __name__ == "__main__":
    run_test()
