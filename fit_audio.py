"""Подгонка длины: макс. ±5% stretch; иначе сдвиг/наложение на таймлайне."""
from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import soundfile as sf

DEFAULT_SR = 16000
# Допустимое отклонение tempo от 1.0 (5% → tempo в [0.95, 1.05])
MAX_STRETCH = float(os.environ.get("SPEECHLAB_MAX_STRETCH", "0.05"))
_rubberband_ready: bool | None = None


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
    """
    Подогнать озвучку к слоту не более чем на ±max_stretch (по умолчанию 5%).
    Возвращает фактическую длительность WAV (может быть > slot_sec).
    """
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

    out_dur = actual / clamped
    tmp_in = in_path
    cleanup: list[Path] = []
    fd, tmp_name = tempfile.mkstemp(suffix=".wav", prefix="speechlab_fit_")
    os.close(fd)
    tmp_out = Path(tmp_name)
    cleanup.append(tmp_out)

    _stretch_file(tmp_in, clamped, tmp_out, y, int(file_sr))
    shutil.move(tmp_out, in_path)
    for p in cleanup:
        p.unlink(missing_ok=True)

    return read_duration(in_path, sr)


def schedule_placements(
    segments: list[tuple[float, float, Path]],
    *,
    max_stretch: float | None = None,
) -> list[tuple[float, Path]]:
    """
    segments: (slot_start, slot_end, dub_wav) по порядку реплик.

    1) На каждый WAV — apply_limited_fit (±5%).
    2) play_start = max(slot_start, конец предыдущего) — отодвигание.
    3) Если после fit длина > слота — следующий сегмент с slot_start предыдущей (наложение).
    """
    if not segments:
        return []

    prepared: list[dict] = []
    for slot_start, slot_end, path in sorted(segments, key=lambda x: x[0]):
        slot_dur = float(slot_end) - float(slot_start)
        dur = apply_limited_fit(path, slot_dur, max_stretch=max_stretch)
        prepared.append({
            "slot_start": float(slot_start),
            "slot_end": float(slot_end),
            "slot_dur": slot_dur,
            "path": Path(path),
            "dur": dur,
        })

    placements: list[tuple[float, Path]] = []
    cursor_end = 0.0
    overlap_at: float | None = None

    for item in prepared:
        slot_start = item["slot_start"]
        path = item["path"]
        dur = item["dur"]
        slot_dur = item["slot_dur"]

        if overlap_at is not None:
            play_start = overlap_at
            overlap_at = None
        else:
            play_start = max(slot_start, cursor_end)

        placements.append((play_start, path))
        cursor_end = play_start + dur

        if dur > slot_dur + 0.02:
            overlap_at = slot_start

    return placements


# Совместимость со старым вызовом (только ±5% fit, без сдвига таймлайна)
def fit_to_duration(
    wav_path: str | Path,
    target_sec: float,
    *,
    sr: int = DEFAULT_SR,
) -> float:
    return apply_limited_fit(wav_path, target_sec, sr=sr)
