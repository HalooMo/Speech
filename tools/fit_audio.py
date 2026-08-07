"""Подгонка длины реплики ±10% и расстановка на таймлайне (PRD шаг 7).

Используется в main.restore_primary_segment():
  1) apply_limited_fit — ускорить/замедлить *_dub.wav под слот реплики (не более ±10%)
  2) schedule_placements — когда реплика длиннее слота и спикер другой — overlay, не обрезка

Лимит: SPEECHLAB_MAX_STRETCH (по умолчанию 0.10).
"""
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


# =============================================================================
# rubberband-cli (предпочтительно) или fallback librosa.time_stretch
# =============================================================================
def _rubberband_exe():
    """Путь к rubberband или None; кэш в _rubberband (False = уже искали, нет)."""
    global _rubberband
    if _rubberband is False:
        return None
    if _rubberband:
        return _rubberband
    exe = shutil.which("rubberband")
    # На Linux при отсутствии — попытка apt (деплой-сервер)
    if not exe and os.name == "posix":
        try:
            subprocess.run(["apt-get", "-qq", "install", "-y", "rubberband-cli"], capture_output=True)
            exe = shutil.which("rubberband")
        except Exception:
            exe = None
    _rubberband = exe or False
    return exe if exe else None


def read_duration(wav_path):
    """Длительность WAV в секундах (без загрузки всего массива в память)."""
    info = sf.info(str(wav_path))
    return info.frames / info.samplerate


def _rms(y: np.ndarray) -> float:
    """RMS громкости mono float32 (тишина → 0)."""
    if y.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(np.square(y, dtype=np.float64))))


def match_loudness(dub_wav, original_wav, *, max_gain=8.0, min_rms=1e-4):
    """Подогнать громкость дубляжа под оригинальную реплику (RMS).

    1) gain = rms_orig / rms_dub (с потолком max_gain)
    2) gain ограничивается так, чтобы peak ≤ 0.98 — без глобального
       «сжать весь файл после усиления» (это ломало RMS-матч)
    3) редкие выбросы выше 0.98 — soft knee (tanh), не линейный scale всего сигнала

    Перезаписывает dub_wav. Возвращает фактически применённый gain.
    """
    dub_path = Path(dub_wav)
    orig_path = Path(original_wav)
    if not dub_path.is_file() or not orig_path.is_file():
        return 1.0

    dub, sr_d = sf.read(dub_path, dtype="float32")
    orig, _ = sf.read(orig_path, dtype="float32")
    if dub.ndim > 1:
        dub = dub.mean(axis=1)
    if orig.ndim > 1:
        orig = orig.mean(axis=1)

    rms_d, rms_o = _rms(dub), _rms(orig)
    if rms_d < min_rms or rms_o < min_rms:
        return 1.0

    gain = float(np.clip(rms_o / rms_d, 1.0 / max_gain, max_gain))
    # Не усиливаем сильнее, чем позволяет peak (иначе пришлось бы давить весь файл)
    peak_d = float(np.max(np.abs(dub))) or 1e-9
    gain = min(gain, 0.98 / peak_d)
    if abs(gain - 1.0) < 0.02:
        return 1.0

    out = (dub * gain).astype(np.float32)
    # Soft limiter только на сэмплах выше порога — RMS почти не страдает
    lim = 0.98
    abs_o = np.abs(out)
    mask = abs_o > lim
    if np.any(mask):
        excess = abs_o[mask] - lim
        scale = float(np.max(excess)) or 1e-9
        soft = lim + (0.99 - lim) * np.tanh(excess / scale)
        out = out.copy()
        out[mask] = np.sign(out[mask]) * soft.astype(np.float32)

    sf.write(dub_path, out, sr_d)
    return gain


# =============================================================================
# Подгонка одной реплики под слот [start, end] с ограничением ±max_stretch
# =============================================================================
def apply_limited_fit(wav_path, slot_sec, max_stretch=None):
    """Ускорить/замедлить WAV не более чем на ±max_stretch к слоту.

    tempo = actual_duration / slot_sec, зажатый в [1-lim, 1+lim].
    Перезаписывает исходный wav_path на месте.
    """
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


# =============================================================================
# Расстановка реплик на таймлайне первичного сегмента
# =============================================================================
def schedule_placements(segments, max_stretch=None):
    """(start, end, wav, speaker) → [(play_start, wav)] с fit и overlay разных спикеров.

    Логика:
      - сначала fit каждой реплики под слот;
      - если после fit реплика всё ещё длиннее слота И спикер сменился —
        начинаем с исходного start (overlay), иначе — после предыдущей (cursor).
    """
    if not segments:
        return []
    # --- Фаза 1: fit + сбор метаданных по каждой реплике ---
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

    # --- Фаза 2: play_start с учётом overflow и смены спикера ---
    out, cursor, prev_spk = [], 0.0, None
    for it in items:
        overflow = it["dur"] > it["slot"] + 0.02
        diff_spk = prev_spk and it["spk"] and prev_spk != it["spk"]
        if overflow and diff_spk:
            # Другой спикер + не влезли в слот → накладываем с собственного start
            start = it["s"]
        else:
            # Один спикер или влезли — ставим после предыдущей или с start
            start = max(it["s"], cursor)
        out.append((start, it["p"]))
        cursor = start + it["dur"]
        prev_spk = it["spk"] or prev_spk
    return out
