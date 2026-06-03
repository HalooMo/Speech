"""
SpeechLab — закадровый дубляж по PRD.md.

Точка входа: run(project_name, video_path, source_language, target_language)
→ Path к {project}_dubbed.mp4 и dub_output_path.txt
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from pathlib import Path

import llm
import numpy as np
import prompt
import soundfile as sf

from env_config import get_hf_token
from test import init_asr_models, run_segment_pipeline, unload_asr_models

ROOT = Path(__file__).resolve().parent
SR = 16000

SPEECH_TXT = re.compile(r"^speech_(\d+)\.txt$", re.I)
SPEECH_WAV = re.compile(
    r"^speech_(\d{3})_([^_]+)_([\d.]+)-([\d.]+)s\.wav$",
    re.I,
)
TIME_LINE = re.compile(r"^\s*([\d.]+)\s*-\s*([\d.]+)\s*$")

TTS_LANGUAGE = {
    "ru": "Russian",
    "russian": "Russian",
    "en": "English",
    "english": "English",
    "de": "German",
    "german": "German",
    "es": "Spanish",
    "spanish": "Spanish",
    "fr": "French",
    "french": "French",
    "auto": "Auto",
}


def _tts_language(code: str) -> str:
    return TTS_LANGUAGE.get(code.strip().lower(), code)


def _run(cmd: list[str], *, check: bool = True) -> subprocess.CompletedProcess:
    print("$", " ".join(cmd))
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        if r.stderr:
            print(r.stderr[-4000:])
        if check:
            raise subprocess.CalledProcessError(
                r.returncode, cmd, output=r.stdout, stderr=r.stderr
            )
    return r


_AUDIO_ONLY_EXTS = {".wav", ".mp3", ".flac", ".ogg", ".opus", ".m4a", ".aac"}


def _has_video_stream(path: Path) -> bool:
    if path.suffix.lower() in _AUDIO_ONLY_EXTS:
        return False
    r = subprocess.run(
        [
            "ffprobe", "-v", "error", "-select_streams", "v:0",
            "-show_entries", "stream=codec_type", "-of", "csv=p=0",
            str(path),
        ],
        capture_output=True,
        text=True,
    )
    return r.returncode == 0 and "video" in (r.stdout or "").lower()


def _audio_duration(path: Path) -> float:
    info = sf.info(path)
    return info.frames / info.samplerate


def _pad_or_trim(wav_path: Path, target_sec: float) -> None:
    data, sr = sf.read(wav_path, dtype="float32")
    if data.ndim > 1:
        data = data.mean(axis=1)
    target_len = int(round(target_sec * sr))
    cur_len = len(data)
    if cur_len < target_len:
        data = np.pad(data, (0, target_len - cur_len))
    elif cur_len > target_len:
        data = data[:target_len]
    sf.write(wav_path, data, sr)


def extract_audio_16k(video_path: Path, out_wav: Path) -> None:
    out_wav.parent.mkdir(parents=True, exist_ok=True)
    _run([
        "ffmpeg", "-y", "-i", str(video_path),
        "-vn", "-acodec", "pcm_s16le", "-ar", str(SR), "-ac", "1",
        str(out_wav),
    ])


def separate_stems(audio_wav: Path, stems_dir: Path) -> tuple[Path, Path]:
    """demucs --two-stems=vocals → (vocals.wav, no_vocals.wav)."""
    stems_dir.mkdir(parents=True, exist_ok=True)
    _run(["demucs", "--two-stems=vocals", "-o", str(stems_dir), str(audio_wav)])
    candidates = list(stems_dir.rglob("vocals.wav"))
    if not candidates:
        raise FileNotFoundError(f"demucs не создал vocals.wav в {stems_dir}")
    vocals = candidates[0]
    no_vocals = vocals.parent / "no_vocals.wav"
    if not no_vocals.is_file():
        raise FileNotFoundError(f"Нет no_vocals.wav рядом с {vocals}")
    return vocals, no_vocals


MIN_PRIMARY_SEC = float(os.environ.get("SPEECHLAB_MIN_PRIMARY_SEC", "40"))
MAX_PRIMARY_SEC = float(os.environ.get("SPEECHLAB_MAX_PRIMARY_SEC", "90"))
ORIGINAL_AUDIO_RATIO = float(os.environ.get("SPEECHLAB_ORIGINAL_AUDIO_RATIO", "0.3"))


def _split_long_segment(
    start: float, end: float, max_len: float,
) -> list[tuple[float, float]]:
    segs: list[tuple[float, float]] = []
    s = start
    while end - s > max_len:
        segs.append((s, s + max_len))
        s += max_len
    if end > s + 0.05:
        segs.append((s, end))
    return segs


def _adjust_primary_bounds(
    bounds: list[float],
    min_len: float,
    max_len: float,
) -> list[float]:
    """PRD: первичные сегменты ~40–90 с (нарезка по паузам + слияние/деление)."""
    if len(bounds) < 2:
        return bounds
    segs: list[tuple[float, float]] = []
    for i in range(len(bounds) - 1):
        segs.extend(_split_long_segment(bounds[i], bounds[i + 1], max_len))
    changed = True
    while changed:
        changed = False
        merged: list[tuple[float, float]] = []
        i = 0
        while i < len(segs):
            s, e = segs[i]
            while (e - s) < min_len and i + 1 < len(segs):
                _, ne = segs[i + 1]
                if ne - s <= max_len:
                    e = ne
                    i += 1
                    changed = True
                else:
                    break
            merged.append((s, e))
            i += 1
        segs = merged
    out = [segs[0][0]]
    for _, e in segs:
        if e > out[-1] + 0.05:
            out.append(e)
    return out


def detect_silence_boundaries(
    wav_path: Path,
    noise_db: float = -30,
    min_silence: float = 0.5,
) -> list[float]:
    """Границы первичных сегментов: 0, t1, t2, …, duration."""
    proc = subprocess.run(
        [
            "ffmpeg", "-i", str(wav_path),
            "-af", f"silencedetect=n={noise_db}dB:d={min_silence}",
            "-f", "null", "-",
        ],
        capture_output=True,
        text=True,
    )
    log = proc.stderr
    duration = _audio_duration(wav_path)
    points = [0.0]
    for line in log.splitlines():
        if "silence_end:" in line:
            part = line.split("silence_end:")[1].strip().split()[0]
            try:
                t = float(part)
                if 0 < t < duration:
                    points.append(t)
            except ValueError:
                continue
    if points[-1] < duration - 0.01:
        points.append(duration)
    # уникальные, по возрастанию
    out = sorted(set(points))
    return out


def split_primary_segments(
    vocals_wav: Path,
    first_seg_dir: Path,
) -> list[dict]:
    """
    Нарезка по тишине → first_seg/001_0.00-12.34/segment.wav
    + manifest.json
    """
    first_seg_dir.mkdir(parents=True, exist_ok=True)
    raw_bounds = detect_silence_boundaries(vocals_wav)
    bounds = _adjust_primary_bounds(raw_bounds, MIN_PRIMARY_SEC, MAX_PRIMARY_SEC)
    manifest = []

    for i in range(len(bounds) - 1):
        start, end = bounds[i], bounds[i + 1]
        if end - start < 0.3:
            continue
        folder = first_seg_dir / f"{i + 1:03d}_{start:.2f}-{end:.2f}"
        folder.mkdir(parents=True, exist_ok=True)
        seg_wav = folder / "segment.wav"
        _run([
            "ffmpeg", "-y", "-i", str(vocals_wav),
            "-ss", str(start), "-to", str(end),
            "-acodec", "pcm_s16le", "-ar", str(SR), "-ac", "1",
            str(seg_wav),
        ])
        manifest.append({
            "index": i + 1,
            "start": start,
            "end": end,
            "folder": folder.name,
            "segment_wav": str(seg_wav.relative_to(first_seg_dir.parent)),
        })

    manifest_path = first_seg_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"Первичных сегментов: {len(manifest)} → {manifest_path}")
    return manifest


def _parse_speech_txt(path: Path) -> dict | None:
    lines = path.read_text(encoding="utf-8").splitlines()
    if len(lines) < 3:
        return None
    m = TIME_LINE.match(lines[0])
    if not m:
        return None
    return {
        "start": float(m.group(1)),
        "end": float(m.group(2)),
        "speaker": lines[1].strip(),
        "text": "\n".join(lines[2:]).strip(),
        "file": path.name,
    }


def _list_speech_txt(text_dir: Path) -> list[Path]:
    files = [p for p in text_dir.glob("speech_*.txt") if SPEECH_TXT.match(p.name)]
    return sorted(files, key=lambda p: int(SPEECH_TXT.match(p.name).group(1)))


def _list_speech_wav(audio_dir: Path) -> list[Path]:
    files = [p for p in audio_dir.glob("speech_*.wav") if SPEECH_WAV.match(p.name)]
    return sorted(files, key=lambda p: int(SPEECH_WAV.match(p.name).group(1)))


def build_casting(second_seg_dir: Path) -> dict:
    """Признаки голоса по репликам → casting.json (dict, PRD)."""
    from get_param import profile_from_wav, unload_model

    audio_dir = second_seg_dir / "output_audio_segments"
    text_dir = second_seg_dir / "output_text_segments"
    casting: dict = {"segments": {}}

    for txt in _list_speech_txt(text_dir):
        meta = _parse_speech_txt(txt)
        if not meta:
            continue
        idx = SPEECH_TXT.match(txt.name).group(1)
        wavs = list(audio_dir.glob(f"speech_{int(idx):03d}_*.wav"))
        if not wavs:
            continue
        wav = wavs[0]
        casting["segments"][txt.name] = {
            "file": txt.name,
            "speaker": meta["speaker"],
            "wav": wav.name,
            "start": meta["start"],
            "end": meta["end"],
            "profile": profile_from_wav(wav),
        }

    out_path = second_seg_dir / "casting.json"
    out_path.write_text(
        json.dumps(casting, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    unload_model()
    return casting


TRANSLATE_BATCH_SIZE = int(os.environ.get("SPEECHLAB_TRANSLATE_BATCH_SIZE", "12"))


def _parse_batch_translations(raw: str) -> list[dict]:
    s = (raw or "").strip()
    if s.startswith("```"):
        s = re.sub(r"^```(?:json)?\s*", "", s, flags=re.IGNORECASE)
        s = re.sub(r"\s*```\s*$", "", s).strip()
    start, end = s.find("["), s.rfind("]")
    if start != -1 and end > start:
        s = s[start : end + 1]
    data = json.loads(s)
    if not isinstance(data, list):
        raise ValueError("Ожидался JSON-массив переводов")
    return data


def _translate_one(
    meta: dict,
    txt_name: str,
    source_lang: str,
    target_lang: str,
) -> str:
    orig = meta["text"]
    slot_sec = float(meta["end"]) - float(meta["start"])
    prom = prompt.get_prompt(
        2,
        {
            "text": orig,
            "source_lang": source_lang,
            "target_lang": target_lang,
            "source_chars": len(orig),
            "source_words": len(orig.split()),
            "slot_sec": slot_sec,
        },
    )
    return (llm.llm_response(prom, json_only=False) or "").strip()


def translate_segments(
    second_seg_dir: Path,
    source_lang: str,
    target_lang: str,
) -> Path:
    text_dir = second_seg_dir / "output_text_segments"
    target_dir = second_seg_dir / "target_text"
    target_dir.mkdir(parents=True, exist_ok=True)

    entries: list[tuple[Path, dict]] = []
    for txt in _list_speech_txt(text_dir):
        meta = _parse_speech_txt(txt)
        if meta and meta["text"]:
            entries.append((txt, meta))

    for i in range(0, len(entries), TRANSLATE_BATCH_SIZE):
        batch = entries[i : i + TRANSLATE_BATCH_SIZE]
        lines = []
        for txt, meta in batch:
            orig = meta["text"]
            slot_sec = float(meta["end"]) - float(meta["start"])
            lines.append({
                "id": txt.name,
                "text": orig,
                "source_chars": len(orig),
                "source_words": len(orig.split()),
                "slot_sec": round(slot_sec, 2),
            })

        by_id: dict[str, str] = {}
        try:
            prom = prompt.get_prompt(
                3,
                {
                    "source_lang": source_lang,
                    "target_lang": target_lang,
                    "lines": lines,
                },
            )
            raw = llm.llm_response_retry(
                prom,
                json_only=False,
                batch_translate=True,
                retries=2,
                retry_suffix='Return ONLY JSON array [{"id":"...","text":"..."}]',
            )
            for item in _parse_batch_translations(raw):
                rid = str(item.get("id", "")).strip()
                text = str(item.get("text", "")).strip()
                if rid and text:
                    by_id[rid] = text
        except Exception as exc:
            print(f"  batch перевод не удался ({exc}), по одной реплике…")

        for txt, meta in batch:
            translated = by_id.get(txt.name)
            if not translated:
                translated = _translate_one(meta, txt.name, source_lang, target_lang)
            out = target_dir / txt.name
            out.write_text(
                f"{meta['start']:.2f} - {meta['end']:.2f}\n{meta['speaker']}\n{translated}",
                encoding="utf-8",
            )
        print(f"  перевод batch {i // TRANSLATE_BATCH_SIZE + 1} ({len(batch)} реплик)")

    return target_dir


def _load_casting(second_seg_dir: Path) -> dict:
    path = second_seg_dir / "casting.json"
    if not path.is_file():
        raise FileNotFoundError(
            f"Нет {path}. Запустите build_casting после разметки реплик."
        )
    return json.loads(path.read_text(encoding="utf-8"))


def dub_segments(
    second_seg_dir: Path,
    target_lang: str,
) -> Path:
    from dubbing import dub_from_profile, ensure_voice_bank, unload_model

    tts_lang = _tts_language(target_lang)
    print("  TTS: банк 8 голосов (Design → Base clone)…")
    ensure_voice_bank(tts_lang)

    casting = _load_casting(second_seg_dir)
    seg_map = casting.get("segments", {})
    target_dir = second_seg_dir / "target_text"
    final_dir = second_seg_dir / "final_audio"
    final_dir.mkdir(parents=True, exist_ok=True)

    for txt in _list_speech_txt(target_dir):
        meta = _parse_speech_txt(txt)
        if not meta:
            continue
        seg = seg_map.get(txt.name, {})
        profile = seg.get("profile", {})

        src_wav = second_seg_dir / "output_audio_segments"
        wav_match = list(src_wav.glob(f"speech_{int(SPEECH_TXT.match(txt.name).group(1)):03d}_*.wav"))
        ref_name = wav_match[0].stem if wav_match else txt.stem

        out_wav = final_dir / f"{ref_name}_dub.wav"
        dub_from_profile(
            text=meta["text"],
            language=tts_lang,
            profile=profile,
            out_path=out_wav,
        )
        slot = float(meta["end"]) - float(meta["start"])
        print(f"  озвучка {out_wav.name} (слот {slot:.2f}s, fit при сборке)")

    unload_model()
    return final_dir


def _build_timeline(
    placements: list[tuple[float, Path]],
    min_total_dur: float,
) -> np.ndarray:
    """Суммирование реплик на таймлайне (overlay без обрезки WAV)."""
    max_end = float(min_total_dur)
    for play_start, path in placements:
        max_end = max(max_end, play_start + _audio_duration(path))

    out = np.zeros(int(round(max_end * SR)), dtype=np.float32)
    for play_start, path in placements:
        data, sr = sf.read(path, dtype="float32")
        if data.ndim > 1:
            data = data.mean(axis=1)
        if sr != SR:
            import torch
            import torchaudio

            t = torch.from_numpy(data).unsqueeze(0)
            t = torchaudio.functional.resample(t, sr, SR)
            data = t.squeeze(0).numpy()
        pos = int(round(play_start * SR))
        end_pos = pos + len(data)
        if end_pos > len(out):
            out = np.pad(out, (0, end_pos - len(out)))
        out[pos:end_pos] += data[: end_pos - pos]

    peak = float(np.max(np.abs(out))) or 1.0
    if peak > 1.0:
        out = out / peak * 0.98
    return out


def restore_primary_segment(primary_dir: Path) -> Path:
    """Склейка final_audio: fit ±10%, overlay только у разных спикеров (PRD)."""
    from fit_audio import schedule_placements

    second_root = primary_dir / "second_seg"
    seg_wav = primary_dir / "segment.wav"
    total_dur = _audio_duration(seg_wav)
    slots: list[tuple[float, float, Path, str]] = []

    if second_root.is_dir():
        final_dir = second_root / "final_audio"
        text_dir = second_root / "output_text_segments"
        if final_dir.is_dir() and text_dir.is_dir():
            for txt in _list_speech_txt(text_dir):
                meta = _parse_speech_txt(txt)
                if not meta:
                    continue
                idx = int(SPEECH_TXT.match(txt.name).group(1))
                dub = list(final_dir.glob(f"speech_{idx:03d}_*_dub.wav"))
                if not dub:
                    dub = list(final_dir.glob(f"*{txt.stem}*_dub.wav"))
                if dub:
                    slots.append((
                        meta["start"],
                        meta["end"],
                        dub[0],
                        meta["speaker"],
                    ))

    if not slots:
        data, _ = sf.read(seg_wav, dtype="float32")
        if data.ndim > 1:
            data = data.mean(axis=1)
        out_path = primary_dir / "restored.wav"
        sf.write(out_path, data, SR)
        return out_path

    placements = schedule_placements(slots)
    mixed = _build_timeline(placements, total_dur)
    out_path = primary_dir / "restored.wav"
    sf.write(out_path, mixed, SR)
    print(f"  таймлайн: {len(placements)} реплик, {len(mixed)/SR:.2f}s")
    return out_path


def restore_full_vocals(
    project_dir: Path,
    manifest: list[dict],
    music_stem: Path,
) -> Path:
    """Склейка первичных restored.wav + музыка → full_dub.wav."""
    pieces: list[tuple[float, np.ndarray]] = []
    max_end = 0.0
    for item in manifest:
        folder = project_dir / "first_seg" / item["folder"]
        restored = folder / "restored.wav"
        if not restored.is_file():
            restored = folder / "segment.wav"
        data, sr = sf.read(restored, dtype="float32")
        if data.ndim > 1:
            data = data.mean(axis=1)
        if sr != SR:
            import torch
            import torchaudio
            t = torch.from_numpy(data).unsqueeze(0)
            t = torchaudio.functional.resample(t, sr, SR)
            data = t.squeeze(0).numpy()
        start = float(item["start"])
        pieces.append((start, data))
        max_end = max(max_end, float(item["end"]))

    full_len = int(round(max_end * SR))
    vocals = np.zeros(full_len, dtype=np.float32)
    for start, data in pieces:
        pos = int(round(start * SR))
        end_pos = min(full_len, pos + len(data))
        vocals[pos:end_pos] = data[: end_pos - pos]

    music, sr_m = sf.read(music_stem, dtype="float32")
    if music.ndim > 1:
        music = music.mean(axis=1)
    if sr_m != SR:
        import torch
        import torchaudio
        t = torch.from_numpy(music).unsqueeze(0)
        t = torchaudio.functional.resample(t, sr_m, SR)
        music = t.squeeze(0).numpy()
    if len(music) < full_len:
        music = np.pad(music, (0, full_len - len(music)))
    else:
        music = music[:full_len]

    mixed = vocals + music * 0.85  # duck music slightly under voice
    peak = np.max(np.abs(mixed)) or 1.0
    if peak > 1.0:
        mixed = mixed / peak * 0.98

    out = project_dir / "full_dub.wav"
    sf.write(out, mixed, SR)
    return out


def _resample_mono(data: np.ndarray, sr: int, target_sr: int = SR) -> np.ndarray:
    if data.ndim > 1:
        data = data.mean(axis=1)
    if sr == target_sr:
        return data.astype(np.float32)
    import torch
    import torchaudio

    t = torch.from_numpy(data.astype(np.float32)).unsqueeze(0)
    t = torchaudio.functional.resample(t, sr, target_sr)
    return t.squeeze(0).numpy()


def extract_original_audio_16k(video_path: Path, out_wav: Path) -> None:
    """Оригинальная звуковая дорожка видео → mono 16 kHz."""
    out_wav.parent.mkdir(parents=True, exist_ok=True)
    _run([
        "ffmpeg", "-y", "-i", str(video_path),
        "-vn", "-acodec", "pcm_s16le", "-ar", str(SR), "-ac", "1",
        str(out_wav),
    ])


def mix_dub_with_original(
    video_path: Path,
    dub_wav: Path,
    out_wav: Path,
    *,
    original_ratio: float | None = None,
) -> Path:
    """PRD: финальная дорожка = дубляж + ~30% оригинала из видео."""
    ratio = ORIGINAL_AUDIO_RATIO if original_ratio is None else original_ratio
    if not _has_video_stream(video_path):
        import shutil
        shutil.copy2(dub_wav, out_wav)
        return out_wav

    orig_tmp = out_wav.parent / "_original_16k.wav"
    extract_original_audio_16k(video_path, orig_tmp)

    dub_data, dub_sr = sf.read(dub_wav, dtype="float32")
    orig_data, orig_sr = sf.read(orig_tmp, dtype="float32")
    dub = _resample_mono(dub_data, dub_sr)
    orig = _resample_mono(orig_data, orig_sr)

    n = max(len(dub), len(orig))
    if len(dub) < n:
        dub = np.pad(dub, (0, n - len(dub)))
    if len(orig) < n:
        orig = np.pad(orig, (0, n - len(orig)))
    else:
        orig = orig[:n]
        dub = dub[:n]

    mixed = dub + orig * ratio
    peak = float(np.max(np.abs(mixed))) or 1.0
    if peak > 1.0:
        mixed = mixed / peak * 0.98

    out_wav.parent.mkdir(parents=True, exist_ok=True)
    sf.write(out_wav, mixed.astype(np.float32), SR)
    print(f"  микс: дубляж + оригинал × {ratio:.2f}")
    return out_wav


def mux_video(video_path: Path, audio_path: Path, out_path: Path) -> Path:
    """Склейка видео+аудио. Если вход только аудио (.wav) — MP4 с чёрным кадром."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if not _has_video_stream(video_path):
        dur = _audio_duration(audio_path)
        print(f"  вход без видео ({video_path.name}) → MP4 с dub-аудио")
        _run([
            "ffmpeg", "-y",
            "-f", "lavfi",
            "-i", f"color=c=black:s=1280x720:r=24:d={dur:.3f}",
            "-i", str(audio_path),
            "-c:v", "libx264", "-pix_fmt", "yuv420p",
            "-c:a", "aac", "-b:a", "192k",
            "-shortest",
            str(out_path),
        ])
        return out_path
    _run([
        "ffmpeg", "-y",
        "-i", str(video_path),
        "-i", str(audio_path),
        "-c:v", "copy",
        "-c:a", "aac", "-b:a", "192k",
        "-map", "0:v:0", "-map", "1:a:0",
        "-shortest",
        str(out_path),
    ])
    return out_path


def run(
    project_name: str,
    video_path: str | Path,
    source_language: str,
    target_language: str,
    *,
    hf_token: str | None = None,
) -> Path:
    """
    Полный пайплайн SpeechLab.

    project_name — имя папки проекта (создаётся в корне репозитория).
    video_path — исходное видео (.mp4 и др.).
    source_language — язык оригинала (код Whisper, напр. en).
    target_language — язык дубляжа (код или Russian/English для TTS).
    """
    video_path = Path(video_path).resolve()
    if not video_path.is_file():
        raise FileNotFoundError(video_path)

    project_dir = ROOT / project_name
    project_dir.mkdir(parents=True, exist_ok=True)

    token = hf_token or get_hf_token()

    print("=== 1. Аудио 16 kHz + demucs ===")
    audio_16k = project_dir / "audio_16k.wav"
    extract_audio_16k(video_path, audio_16k)
    stems_dir = project_dir / "demucs_stems"
    vocals_wav, music_stem = separate_stems(audio_16k, stems_dir)
    project_vocals = project_dir / "vocals.wav"
    if project_vocals.exists():
        project_vocals.unlink()
    project_vocals.write_bytes(vocals_wav.read_bytes())
    print(f"vocals: {project_vocals}, music: {music_stem}")

    print("=== 2. Первичная нарезка по тишине ===")
    first_seg_dir = project_dir / "first_seg"
    manifest = split_primary_segments(project_vocals, first_seg_dir)

    print("=== 3. Вторичные сегменты (PRD: pyannote + WhisperX + LLM) ===")
    init_asr_models(source_language, token)
    try:
        for item in manifest:
            primary_dir = first_seg_dir / item["folder"]
            seg_wav = primary_dir / "segment.wav"
            second_seg = primary_dir / "second_seg"
            second_seg.mkdir(parents=True, exist_ok=True)

            print(f"--- Первичный {item['folder']} ---")
            run_segment_pipeline(
                seg_wav,
                second_seg / "output_audio_segments",
                second_seg / "output_text_segments",
                source_language,
                hf_token=token,
                reuse_asr=True,
            )

            print("=== 4. Перевод (LLM batch, длина ≈ слот) ===")
            translate_segments(second_seg, source_language, target_language)

            print("=== 5. casting.json (dict: пол, возраст) ===")
            build_casting(second_seg)

            print("=== 6. Qwen3-TTS → final_audio ===")
            dub_segments(second_seg, target_language)

            print("=== 7. Склейка реплик (fit ±10%, overlay разных спикеров) ===")
            restore_primary_segment(primary_dir)
    finally:
        unload_asr_models()

    print("=== 8. Первичные сегменты + музыка → full_dub.wav ===")
    full_audio = restore_full_vocals(project_dir, manifest, music_stem)

    print("=== 9. Оригинал видео ~30% + mux MP4 ===")
    mux_audio = project_dir / "final_mux_audio.wav"
    mix_dub_with_original(video_path, full_audio, mux_audio)
    out_video = project_dir / f"{project_name}_dubbed.mp4"
    mux_video(video_path, mux_audio, out_video)

    path_file = project_dir / "dub_output_path.txt"
    path_file.write_text(str(out_video.resolve()), encoding="utf-8")
    print(f"Готово: {out_video}")
    print(f"Путь (PRD): {path_file}")
    return out_video


def main() -> None:
    if len(sys.argv) < 5:
        print(
            "Использование: python main.py <project_name> <video_path> "
            "<source_lang> <target_lang>"
        )
        print("Пример: python main.py myfilm input.mp4 en ru")
        sys.exit(1)
    run(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4])


if __name__ == "__main__":
    main()
