"""Подгонка длительности WAV без смены высоты тона (Rubber Band → librosa)."""
from __future__ import annotations

import os
import re
import shutil
import subprocess
from pathlib import Path

import numpy as np
import soundfile as sf


def fit_audio_duration(
    in_wav: str | Path,
    target_sec: float,
    source_sec: float | None = None,
    out_wav: str | Path | None = None,
    *,
    min_tempo: float = 0.5,
    max_tempo: float = 2.0,
) -> Path:
    """
    Ускорить/замедлить речь под target_sec без изменения pitch.

    Parameters
    ----------
    in_wav : путь к входному WAV
    target_sec : нужная длительность (сек)
    source_sec : длина исходного аудио (сек); если None — из файла
    out_wav : выход; по умолчанию <stem>_fit.wav
    min_tempo, max_tempo : лимиты для ffmpeg atempo (если нет Rubber Band)

    Returns
    -------
    Path к записанному файлу
    """
    in_path = Path(in_wav)
    if not in_path.is_file():
        raise FileNotFoundError(in_path)

    out_path = Path(out_wav) if out_wav else in_path.with_name(f"{in_path.stem}_fit.wav")
    if target_sec <= 0:
        raise ValueError("target_sec must be > 0")

    y, sr = sf.read(in_path, dtype="float32")
    if y.ndim > 1:
        y = y.mean(axis=1)

    actual_sec = float(source_sec) if source_sec is not None else len(y) / sr
    if actual_sec <= 0:
        raise ValueError("source_sec must be > 0")

    tempo = actual_sec / target_sec
    tempo_clamped = float(np.clip(tempo, min_tempo, max_tempo))

    if _run_rubberband(in_path, out_path, tempo):
        pass
    elif _run_ffmpeg_atempo(in_path, out_path, tempo_clamped):
        pass
    else:
        _run_librosa(y, sr, out_path, tempo)

    return out_path.resolve()


def _run_rubberband(in_path: Path, out_path: Path, tempo: float) -> bool:
    subprocess.run(["apt-get", "-qq", "update"], check=False)
    r = subprocess.run(
        ["apt-get", "-qq", "install", "-y", "rubberband-cli"],
        capture_output=True,
        text=True,
    )
    rb = shutil.which("rubberband") or "/usr/bin/rubberband"
    if r.returncode != 0 or not os.path.isfile(rb):
        return False
    subprocess.run([rb, "-t", f"{tempo:.4f}", str(in_path), str(out_path)], check=True)
    return True


def _run_ffmpeg_atempo(in_path: Path, out_path: Path, tempo: float) -> bool:
    if not shutil.which("ffmpeg"):
        return False
    subprocess.run(
        [
            "ffmpeg", "-y", "-i", str(in_path),
            "-af", f"atempo={tempo:.4f}",
            str(out_path),
        ],
        check=True,
        capture_output=True,
    )
    return True


def _run_librosa(y: np.ndarray, sr: int, out_path: Path, tempo: float) -> None:
    import librosa

    y2 = librosa.effects.time_stretch(y, rate=tempo)
    sf.write(out_path, y2, sr)


def target_sec_from_filename(path: str | Path) -> float | None:
    """13.71-17.47s.wav → 3.76"""
    m = re.search(r"([\d.]+)-([\d.]+)s\.wav$", str(path), re.I)
    if not m:
        return None
    return float(m.group(2)) - float(m.group(1))
