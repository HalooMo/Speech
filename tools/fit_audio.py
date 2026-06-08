"""Подгонка длины реплики ±10% и расстановка на таймлайне."""
import os
import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import soundfile as sf

SR = 16000
MAX_STRETCH = float(os.environ.get("SPEECHLAB_MAX_STRETCH", "0.10"))
_rubberband = None


def _rubberband_exe():
    """Путь к rubberband или None."""
    global _rubberband
    if _rubberband is False:
        return None
    if _rubberband:
        return _rubberband
    exe = shutil.which("rubberband")
    if not exe and os.name == "posix":
        try:
            subprocess.run(["apt-get", "-qq", "install", "-y", "rubberband-cli"], capture_output=True)
            exe = shutil.which("rubberband")
        except Exception:
            exe = None
    _rubberband = exe or False
    return exe if exe else None


def read_duration(wav_path):
    info = sf.info(str(wav_path))
    return info.frames / info.samplerate


def apply_limited_fit(wav_path, slot_sec, max_stretch=None):
    """Ускорить/замедлить WAV не более чем на ±max_stretch к слоту."""
    lim = MAX_STRETCH if max_stretch is None else max_stretch
    t_min, t_max = 1.0 - lim, 1.0 + lim
    path = Path(wav_path)
    y, sr = sf.read(path, dtype="float32")
    if y.ndim > 1:
        y = y.mean(axis=1)
    actual = len(y) / sr
    if slot_sec <= 0 or actual < 1e-6:
        return actual
    tempo = max(t_min, min(t_max, actual / slot_sec))
    if abs(tempo - 1.0) < 0.005:
        return actual

    fd, tmp = tempfile.mkstemp(suffix=".wav")
    os.close(fd)
    out = Path(tmp)
    rb = _rubberband_exe()
    if rb:
        subprocess.run([rb, "-t", f"{tempo:.4f}", str(path), str(out)], check=True, capture_output=True)
    else:
        import librosa
        sf.write(out, librosa.effects.time_stretch(y, rate=tempo), sr)
    y2, sr2 = sf.read(out, dtype="float32")
    peak = np.max(np.abs(y2)) or 1.0
    if peak > 1.0:
        y2 = y2 / peak * 0.98
    sf.write(path, y2.astype(np.float32), sr2)
    out.unlink(missing_ok=True)
    return read_duration(path)


def schedule_placements(segments, max_stretch=None):
    """(start, end, wav, speaker) → [(play_start, wav)] с fit и overlay разных спикеров."""
    if not segments:
        return []
    items = []
    for seg in segments:
        if len(seg) == 3:
            s, e, p, spk = float(seg[0]), float(seg[1]), Path(seg[2]), ""
        else:
            s, e, p, spk = float(seg[0]), float(seg[1]), Path(seg[2]), str(seg[3]).strip()
        slot = e - s
        dur = apply_limited_fit(p, slot, max_stretch=max_stretch)
        items.append({"s": s, "e": e, "slot": slot, "p": p, "dur": dur, "spk": spk})
    items.sort(key=lambda x: x["s"])

    out, cursor, overlap_at, prev_spk = [], 0.0, None, None
    for it in items:
        overflow = it["dur"] > it["slot"] + 0.02
        diff_spk = prev_spk and it["spk"] and prev_spk != it["spk"]
        if overlap_at is not None:
            start = overlap_at
            overlap_at = None
        elif overflow and diff_spk:
            start = it["s"]
        else:
            start = max(it["s"], cursor)
        out.append((start, it["p"]))
        cursor = start + it["dur"]
        prev_spk = it["spk"] or prev_spk
        if overflow and diff_spk:
            overlap_at = it["s"]
    return out
