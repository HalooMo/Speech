"""
SpeechLab — полный пайплайн закадрового дубляжа.

Точка входа: run(project_name, video_path, source_language, target_language)
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
from test import run_segment_pipeline

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


MAX_PRIMARY_SEC = float(os.environ.get("SPEECHLAB_MAX_PRIMARY_SEC", "45"))


def _refine_bounds(bounds: list[float], max_len: float) -> list[float]:
    """Дробит слишком длинные интервалы (иначе LLM отказывается от огромного промпта)."""
    if len(bounds) < 2:
        return bounds
    out = [bounds[0]]
    for end in bounds[1:]:
        start = out[-1]
        while end - start > max_len:
            start = start + max_len
            out.append(start)
        if end > out[-1] + 0.05:
            out.append(end)
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
    bounds = _refine_bounds(detect_silence_boundaries(vocals_wav), MAX_PRIMARY_SEC)
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


def collect_voice_params(second_seg_dir: Path) -> Path:
    """Пол и эмоция на каждую реплику → voice_param.txt."""
    from get_param import get_emotion, get_sex

    audio_dir = second_seg_dir / "output_audio_segments"
    text_dir = second_seg_dir / "output_text_segments"
    out_file = second_seg_dir / "voice_param.txt"

    lines = ["# voice_param.txt", "", "[segments]"]
    for txt in _list_speech_txt(text_dir):
        meta = _parse_speech_txt(txt)
        if not meta:
            continue
        idx = SPEECH_TXT.match(txt.name).group(1)
        wavs = list(audio_dir.glob(f"speech_{int(idx):03d}_*.wav"))
        if not wavs:
            continue
        wav = wavs[0]
        lines.append(f"file=speech_{idx}.txt")
        lines.append(f"speaker={meta['speaker']}")
        lines.append(f"wav={wav.name}")
        lines.append(f"start={meta['start']}")
        lines.append(f"end={meta['end']}")
        lines.append(f"sex={get_sex(str(wav))}")
        lines.append(f"emotion={get_emotion(str(wav))}")
        lines.append("")

    out_file.write_text("\n".join(lines), encoding="utf-8")

    from get_param import unload_model

    unload_model()
    return out_file


def _load_voice_params(vp_path: Path) -> dict[str, dict]:
    """Сегменты по имени speech_NNN.txt → поля sex, emotion, …"""
    text = vp_path.read_text(encoding="utf-8")
    segments: dict[str, dict] = {}
    section = None
    cur_seg: dict = {}

    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if line == "[segments]":
            section = "segments"
            continue
        if line == "[speakers]":
            section = "speakers"
            continue
        if section != "segments" or "=" not in line:
            continue
        key, val = line.split("=", 1)
        if key == "file":
            if cur_seg.get("file"):
                segments[cur_seg["file"]] = cur_seg
            cur_seg = {"file": val}
        else:
            cur_seg[key] = val
    if cur_seg.get("file"):
        segments[cur_seg["file"]] = cur_seg
    return segments


def translate_segments(
    second_seg_dir: Path,
    source_lang: str,
    target_lang: str,
) -> Path:
    text_dir = second_seg_dir / "output_text_segments"
    target_dir = second_seg_dir / "target_text"
    target_dir.mkdir(parents=True, exist_ok=True)

    for txt in _list_speech_txt(text_dir):
        meta = _parse_speech_txt(txt)
        if not meta or not meta["text"]:
            continue
        orig = meta["text"]
        prom = prompt.get_prompt(
            2,
            {
                "text": orig,
                "source_lang": source_lang,
                "target_lang": target_lang,
                "source_chars": len(orig),
                "source_words": len(orig.split()),
            },
        )
        translated = (llm.llm_response(prom, json_only=False) or "").strip()
        out = target_dir / txt.name
        out.write_text(
            f"{meta['start']:.2f} - {meta['end']:.2f}\n{meta['speaker']}\n{translated}",
            encoding="utf-8",
        )
        print(f"  перевод {txt.name}")
    return target_dir


def dub_segments(
    second_seg_dir: Path,
    target_lang: str,
) -> Path:
    from dubbing import dub_from_voice_param, unload_model

    vp_path = second_seg_dir / "voice_param.txt"
    if not vp_path.is_file():
        raise FileNotFoundError(f"Нет {vp_path}")
    seg_params = _load_voice_params(vp_path)
    target_dir = second_seg_dir / "target_text"
    final_dir = second_seg_dir / "final_audio"
    final_dir.mkdir(parents=True, exist_ok=True)

    tts_lang = _tts_language(target_lang)

    for txt in _list_speech_txt(target_dir):
        meta = _parse_speech_txt(txt)
        if not meta:
            continue
        params = seg_params.get(txt.name, {})
        sex_raw = params.get("sex", "{}")
        emo_raw = params.get("emotion", "{}")

        src_wav = second_seg_dir / "output_audio_segments"
        wav_match = list(src_wav.glob(f"speech_{int(SPEECH_TXT.match(txt.name).group(1)):03d}_*.wav"))
        ref_name = wav_match[0].stem if wav_match else txt.stem

        out_wav = final_dir / f"{ref_name}_dub.wav"
        dub_from_voice_param(
            text=meta["text"],
            language=tts_lang,
            sex=sex_raw,
            emotion=emo_raw,
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
    """placements: (play_start, path) — полная длина WAV, без обрезки."""
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
        out[pos:end_pos] = data
    return out


def restore_primary_segment(primary_dir: Path) -> Path:
    """Склейка final_audio: ±5% fit, сдвиг и наложение при переполнении слота."""
    from fit_audio import schedule_placements

    second_root = primary_dir / "second_seg"
    seg_wav = primary_dir / "segment.wav"
    total_dur = _audio_duration(seg_wav)
    slots: list[tuple[float, float, Path]] = []

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
                    slots.append((meta["start"], meta["end"], dub[0]))

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

    print("=== 3. Вторичные сегменты (test.py) ===")
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
        )

        print("=== 4. voice_param.txt ===")
        collect_voice_params(second_seg)

        print("=== 5. Перевод → target_text ===")
        translate_segments(second_seg, source_language, target_language)

        print("=== 6. Озвучка → final_audio (fit ±5% при сборке) ===")
        dub_segments(second_seg, target_language)

        print(f"=== 7. Сборка первичного {item['folder']} ===")
        restore_primary_segment(primary_dir)

    print("=== 7b. Полная дорожка + музыка ===")
    full_audio = restore_full_vocals(project_dir, manifest, music_stem)

    print("=== 8. Видео ===")
    out_video = project_dir / f"{project_name}_dubbed.mp4"
    mux_video(video_path, full_audio, out_video)
    print(f"Готово: {out_video}")
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
