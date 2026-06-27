"""Silero TTS v5 (ru): априорные спикеры для русского дубляжа.

Только target_language ru/russian. Спикеры: aidar, eugene (male); baya, kseniya, xenia (female).
Выход resample → 16 kHz mono (как остальной пайплайн).
"""
from __future__ import annotations

import gc
import os
from pathlib import Path

import numpy as np
import soundfile as sf

# Пол спикера — для сопоставления с casting.json
SILERO_SPEAKERS: dict[str, dict] = {
    "aidar": {"gender": "male"},
    "eugene": {"gender": "male"},
    "baya": {"gender": "female"},
    "kseniya": {"gender": "female"},
    "xenia": {"gender": "female"},
}

SILERO_MODEL_ID = os.environ.get("SPEECHLAB_SILERO_MODEL", "v5_ru")
SILERO_HUB_LANG = "ru"
SILERO_SAMPLE_RATE = int(os.environ.get("SPEECHLAB_SILERO_SAMPLE_RATE", "48000"))
PIPELINE_SR = 16000

_model = None
_device = None


def is_russian_target(language: str) -> bool:
    """Silero v5_ru только для русского целевого языка."""
    code = (language or "").strip().lower()
    return code in ("ru", "russian", "rus")

def normalize_speaker(name: str) -> str:
    """Проверка имени спикера Silero."""
    key = (name or "").strip().lower()
    if key not in SILERO_SPEAKERS:
        raise ValueError(
            f"silero_speaker: ожидается один из {sorted(SILERO_SPEAKERS)}, получено {name!r}"
        )
    return key


def speaker_gender(speaker: str) -> str:
    return SILERO_SPEAKERS[normalize_speaker(speaker)]["gender"]


def _get_device():
    import torch
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _load_model():
    """Ленивая загрузка через torch.hub (snakers4/silero-models)."""
    global _model, _device
    if _model is not None:
        return _model
    import torch
    _device = _get_device()
    print(f"  Silero TTS: загрузка ({SILERO_MODEL_ID}, {_device})…")
    model, _ = torch.hub.load(
        repo_or_dir="snakers4/silero-models",
        model="silero_tts",
        language=SILERO_HUB_LANG,
        speaker=SILERO_MODEL_ID,
        trust_repo=True,
    )
    model.to(_device)
    _model = model
    return _model


def _to_mono_float32(audio) -> np.ndarray:
    import torch
    if isinstance(audio, torch.Tensor):
        audio = audio.detach().cpu().numpy()
    arr = np.asarray(audio, dtype=np.float32).squeeze()
    if arr.ndim > 1:
        arr = arr.mean(axis=0)
    peak = float(np.max(np.abs(arr))) or 1.0
    if peak > 1.0:
        arr = arr / peak * 0.98
    return arr


def _resample(audio: np.ndarray, sr: int, target_sr: int = PIPELINE_SR) -> np.ndarray:
    if sr == target_sr:
        return audio.astype(np.float32)
    import torch
    import torchaudio
    t = torch.from_numpy(audio.astype(np.float32)).unsqueeze(0)
    t = torchaudio.functional.resample(t, sr, target_sr)
    return t.squeeze(0).numpy()


def synthesize(text: str, speaker: str, out_path: Path | str) -> str:
    """Синтез одной реплики → WAV 16 kHz mono."""
    if not (text or "").strip():
        raise ValueError("Silero: пустой text")
    spk = normalize_speaker(speaker)
    model = _load_model()
    audio = model.apply_tts(
        text=text.strip(),
        speaker=spk,
        sample_rate=SILERO_SAMPLE_RATE,
        put_accent=True,
        put_yo=True,
        put_stress_homo=True,
        put_yo_homo=True,
    )
    wav = _resample(_to_mono_float32(audio), SILERO_SAMPLE_RATE, PIPELINE_SR)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(out_path, wav, PIPELINE_SR)
    return str(out_path.resolve())


def unload_model():
    """Освободить Silero после этапа TTS."""
    global _model, _device
    _model = None
    _device = None
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass
