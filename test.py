"""PRD п. 3.1–3.2: pyannote + WhisperX → LLM → output_audio/text_segments."""
from __future__ import annotations

import ast
import json
import os
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent

# --- настройки локального теста (python test.py) ---
WAV_PATH = ROOT / "vocals.wav"
SOURCE_LANG = "en"
OUTPUT_AUDIO_DIR = ROOT / "output_audio_segments"
OUTPUT_TEXT_DIR = ROOT / "output_text_segments"

import llm
import pandas as pd
import prompt
import soundfile as sf
import torch
import torchaudio
import whisperx
from huggingface_hub import login
from pyannote.audio import Pipeline

from env_config import get_hf_token

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
COMPUTE_TYPE = os.environ.get("SPEECHLAB_COMPUTE_TYPE", "float32")
WHISPER_MODEL = os.environ.get("SPEECHLAB_WHISPER_MODEL", "large-v3")
MAX_WORDS_LLM = int(os.environ.get("SPEECHLAB_MAX_WORDS_LLM", "2800"))


def clear_directory(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    for item in path.iterdir():
        if item.is_file() or item.is_symlink():
            item.unlink()
        elif item.is_dir():
            import shutil
            shutil.rmtree(item)


def parse_llm_segments(raw: str) -> list:
    """Извлечь JSON-массив сегментов из ответа LLM."""
    s = (raw or "").strip()
    if not s:
        raise ValueError("LLM вернул пустой ответ")

    if llm.looks_like_refusal(s, json_only=True):
        raise ValueError(f"LLM отказалась обработать сегмент: {s[:300]}")

    if s.startswith("```"):
        s = re.sub(r"^```(?:json)?\s*", "", s, flags=re.IGNORECASE)
        s = re.sub(r"\s*```\s*$", "", s).strip()

    start = s.find("[")
    end = s.rfind("]")
    if start != -1 and end > start:
        s = s[start : end + 1]

    try:
        data = json.loads(s)
    except json.JSONDecodeError as exc:
        try:
            data = ast.literal_eval(s)
        except (SyntaxError, ValueError) as exc2:
            raise ValueError(
                f"Не удалось разобрать JSON сегментов: {exc}; {exc2}\n{s[:500]}"
            ) from exc

    if not isinstance(data, list):
        raise ValueError(f"Ожидался JSON-массив, получено: {type(data).__name__}")
    return data


def cut_audio_samples(audio, sr: int, start: float, end: float):
    if end <= start:
        n = len(audio) if audio.ndim == 1 else audio.shape[0]
        return audio[:0]
    n = len(audio) if audio.ndim == 1 else audio.shape[0]
    s = max(0, min(n, int(round(start * sr))))
    e = max(s, min(n, int(round(end * sr))))
    return audio[s:e]


def run_segment_pipeline(
    audio_path: str | Path,
    output_audio_dir: str | Path,
    output_text_dir: str | Path,
    source_language: str,
    *,
    hf_token: str | None = None,
    clear_outputs: bool = True,
) -> list[dict]:
    """Один WAV (16 kHz mono): diarization → WhisperX → LLM → output_*."""
    token = hf_token or get_hf_token()

    audio_path = Path(audio_path)
    output_audio_dir = Path(output_audio_dir)
    output_text_dir = Path(output_text_dir)

    if clear_outputs:
        clear_directory(output_audio_dir)
        clear_directory(output_text_dir)
    else:
        output_audio_dir.mkdir(parents=True, exist_ok=True)
        output_text_dir.mkdir(parents=True, exist_ok=True)

    login(token=token)

    diarization = Pipeline.from_pretrained(
        "pyannote/speaker-diarization-community-1",
        token=token,
    ).to(torch.device(DEVICE))

    whisper = whisperx.load_model(
        WHISPER_MODEL,
        DEVICE,
        compute_type=COMPUTE_TYPE,
        language=source_language,
        vad_method="silero",
    )
    audio = whisperx.load_audio(str(audio_path))

    waveform, sample_rate = torchaudio.load(str(audio_path))
    diar_result = diarization({"waveform": waveform, "sample_rate": sample_rate})

    raw_segments = []
    for seg, spk in diar_result.speaker_diarization.itertracks():
        raw_segments.append({
            "start": max(0, seg.start),
            "end": min(waveform.shape[1] / sample_rate, seg.end),
            "speaker": spk,
        })
    raw_segments.sort(key=lambda x: x["start"])
    diarize_df = pd.DataFrame(raw_segments)

    transcript = whisper.transcribe(audio, batch_size=4)
    align_model, align_meta = whisperx.load_align_model(
        language_code=source_language, device=DEVICE
    )
    transcript = whisperx.align(
        transcript["segments"],
        align_model,
        align_meta,
        audio,
        DEVICE,
        return_char_alignments=False,
    )

    result_with_spk = whisperx.assign_word_speakers(diarize_df, transcript)
    word_segments = result_with_spk["word_segments"]

    n_words = len(word_segments)
    if n_words > MAX_WORDS_LLM:
        raise ValueError(
            f"Слишком много слов для LLM ({n_words} > {MAX_WORDS_LLM}). "
            "Укоротите первичный сегмент (SPEECHLAB_MAX_PRIMARY_SEC) или увеличьте лимит."
        )

    prom = prompt.get_prompt(1, word_segments)
    raw = llm.llm_response_retry(
        prom,
        json_only=True,
        retries=3,
        retry_suffix="Верни ТОЛЬКО JSON-массив speech-сегментов. Без отказов и пояснений.",
    )
    final_segments = parse_llm_segments(raw)

    final_segments = [
        s
        for s in final_segments
        if s.get("type") == "speech"
        and (s.get("text") or "").strip()
        and float(s["end"]) > float(s["start"])
    ]
    final_segments.sort(key=lambda s: float(s["start"]))

    sr = 16000
    speech_cnt = 0
    for seg in final_segments:
        chunk = cut_audio_samples(audio, sr, float(seg["start"]), float(seg["end"]))
        if len(chunk) == 0:
            continue
        speech_cnt += 1
        wav_path = output_audio_dir / (
            f"speech_{speech_cnt:03d}_{seg['speaker']}_{seg['start']:.2f}-{seg['end']:.2f}s.wav"
        )
        sf.write(wav_path, chunk, sr)
        txt_path = output_text_dir / f"speech_{speech_cnt:03d}.txt"
        txt_path.write_text(
            f"{seg['start']:.2f} - {seg['end']:.2f}\n{seg['speaker']}\n{seg['text']}",
            encoding="utf-8",
        )

    return final_segments


def run_test() -> list[dict]:
    """Локальный прогон: пути и язык — константы в начале файла."""
    audio = Path(WAV_PATH).resolve()
    if not audio.is_file():
        raise FileNotFoundError(f"WAV не найден: {audio}")

    print(f"Вход: {audio}")
    print(f"Язык: {SOURCE_LANG}")
    print(f"WAV → {Path(OUTPUT_AUDIO_DIR).resolve()}")
    print(f"TXT → {Path(OUTPUT_TEXT_DIR).resolve()}")

    segments = run_segment_pipeline(
        audio,
        OUTPUT_AUDIO_DIR,
        OUTPUT_TEXT_DIR,
        SOURCE_LANG,
        hf_token=get_hf_token(),
    )

    print(f"\nГотово: {len(segments)} реплик")
    for i, seg in enumerate(segments, 1):
        print(
            f"  {i:3d}  {seg['speaker']:3s}  "
            f"{float(seg['start']):7.2f}-{float(seg['end']):7.2f}s  "
            f"{(seg.get('text') or '')[:60]}"
        )
    return segments


if __name__ == "__main__":
    run_test()
