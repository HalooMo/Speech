"""Подгонка длины: макс. ±5% stretch; сдвиг/наложение по PRD (разные спикеры)."""
from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Union

import numpy as np
import soundfile as sf

DEFAULT_SR = 16000
MAX_STRETCH = float(os.environ.get("SPEECHLAB_MAX_STRETCH", "0.05"))
_rubberband_ready: bool | None = None

# (slot_start, slot_end, wav_path) или (+ speaker)
SegmentSlot = Union[
    tuple[float, float, Path],
    tuple[float, float, Path, str],
]


def _find_rubberband() -> str | None:
    exe = shutil.which("rubberband")
    if exe:
        return exe
    for p in ("/usr/bin/rubberband", "/usr/local/bin/rubberband"):
        if os.path.isfile(p):
            return p
    return None


def _ensure_rubberband() -> str | None:
    global _rubberband_ready
    if _rubberband_ready is True:
        return _find_rubberband()
    if _rubberband_ready is False:
        return None

    exe = _find_rubberband()
    if exe:
        _rubberband_ready = True
        return exe

    if os.name == "posix":
        try:
            subprocess.run(["apt-get", "-qq", "update"], check=False, capture_output=True)
            r = subprocess.run(
                ["apt-get", "-qq", "install", "-y", "rubberband-cli"],
                capture_output=True,
                text=True,
            )
            if r.returncode == 0:
                exe = _find_rubberband()
        except Exception:
            exe = None

    _rubberband_ready = bool(exe)
    return exe


def read_duration(wav_path: str | Path, sr: int = DEFAULT_SR) -> float:
    info = sf.info(str(wav_path))
    return info.frames / info.samplerate


def _stretch_file(in_path: Path, tempo: float, out_path: Path, y: np.ndarray, sr: int) -> None:
    if abs(tempo - 1.0) < 0.005:
        if out_path.resolve() != in_path.resolve():
            shutil.copy2(in_path, out_path)
        return

    rb = _ensure_rubberband()
    if rb:
        subprocess.run(
            [rb, "-t", f"{tempo:.4f}", str(in_path), str(out_path)],
            check=True,
            capture_output=True,
            text=True,
        )
    else:
        import librosa

        sf.write(out_path, librosa.effects.time_stretch(y, rate=tempo), sr)

    y2, sr2 = sf.read(out_path, dtype="float32")
    peak = np.max(np.abs(y2)) or 1.0
    if peak > 1.0:
        sf.write(out_path, (y2 / peak * 0.98).astype(np.float32), sr2)


def apply_limited_fit(
    wav_path: str | Path,
    slot_sec: float,
    *,
    sr: int = DEFAULT_SR,
    max_stretch: float | None = None,
) -> float:
    """Подогнать озвучку к слоту не более чем на ±max_stretch (по умолчанию 5%)."""
    limit = MAX_STRETCH if max_stretch is None else max_stretch
    t_min, t_max = 1.0 - limit, 1.0 + limit

    in_path = Path(wav_path)
    y, file_sr = sf.read(in_path, dtype="float32")
    if y.ndim > 1:
        y = y.mean(axis=1)

    actual = len(y) / file_sr
    if slot_sec <= 0 or actual < 1e-6:
        return actual

    tempo = actual / slot_sec
    clamped = max(t_min, min(t_max, tempo))

    if abs(clamped - 1.0) < 0.005:
        return actual

    fd, tmp_name = tempfile.mkstemp(suffix=".wav", prefix="speechlab_fit_")
    os.close(fd)
    tmp_out = Path(tmp_name)
    _stretch_file(in_path, clamped, tmp_out, y, int(file_sr))
    shutil.move(tmp_out, in_path)
    tmp_out.unlink(missing_ok=True)

    return read_duration(in_path, sr)


def _normalize_slot(seg: SegmentSlot) -> tuple[float, float, Path, str]:
    if len(seg) == 3:
        return float(seg[0]), float(seg[1]), Path(seg[2]), ""
    return float(seg[0]), float(seg[1]), Path(seg[2]), str(seg[3]).strip()


def schedule_placements(
    segments: list[SegmentSlot],
    *,
    max_stretch: float | None = None,
) -> list[tuple[float, Path]]:
    """
    PRD: не обрезать WAV; fit ±5%; наложение только если спикер другой и реплика
    не влезает в слот после fit. Один спикер — сдвиг (без overlay).
    """
    if not segments:
        return []

    prepared: list[dict] = []
    for raw in segments:
        slot_start, slot_end, path, speaker = _normalize_slot(raw)
        slot_dur = slot_end - slot_start
        dur = apply_limited_fit(path, slot_dur, max_stretch=max_stretch)
        prepared.append({
            "slot_start": slot_start,
            "slot_end": slot_end,
            "slot_dur": slot_dur,
            "path": path,
            "dur": dur,
            "speaker": speaker,
        })
    prepared.sort(key=lambda x: x["slot_start"])

    placements: list[tuple[float, Path]] = []
    cursor_end = 0.0
    overlap_at: float | None = None
    prev_speaker: str | None = None

    for item in prepared:
        slot_start = item["slot_start"]
        path = item["path"]
        dur = item["dur"]
        slot_dur = item["slot_dur"]
        speaker = item["speaker"]

        overflow = dur > slot_dur + 0.02
        allow_overlap = (
            overflow
            and prev_speaker is not None
            and speaker
            and prev_speaker != speaker
        )

        if overlap_at is not None:
            play_start = overlap_at
            overlap_at = None
        elif allow_overlap:
            play_start = slot_start
        else:
            play_start = max(slot_start, cursor_end)

        placements.append((play_start, path))
        cursor_end = play_start + dur
        prev_speaker = speaker or prev_speaker

        if allow_overlap:
            overlap_at = slot_start

    return placements


def fit_to_duration(
    wav_path: str | Path,
    target_sec: float,
    *,
    sr: int = DEFAULT_SR,
) -> float:
    return apply_limited_fit(wav_path, target_sec, sr=sr)
