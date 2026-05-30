"""Озвучка текста: биометрия + эмоция → путь к WAV."""
from __future__ import annotations

import json
import uuid
from pathlib import Path
from typing import Any

import numpy as np

MODEL = "Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign"
OUT_DIR = Path("output/tts")

_model: Any = None


def _device() -> str:
    import torch

    return "cuda:0" if torch.cuda.is_available() else "cpu"


def _as_dict(value: dict | str) -> dict:
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError as exc:
            raise ValueError(f"String argument is not valid JSON: {value}") from exc
        if not isinstance(parsed, dict):
            raise TypeError("JSON value must be an object (dict)")
        return parsed
    raise TypeError("voice_biometrics and voice_emotion must be dict or JSON string")


def _to_wav_array(wav: Any) -> np.ndarray:
    if hasattr(wav, "detach"):
        wav = wav.detach().cpu().numpy()
    arr = np.asarray(wav, dtype=np.float32).squeeze()
    if arr.ndim != 1:
        raise ValueError(f"Expected 1-D waveform, got shape {arr.shape}")
    return arr


def unload_model() -> None:
    """Освободить VRAM после озвучки."""
    global _model
    import gc

    import torch

    _model = None
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _get_model():
    global _model
    if _model is None:
        import torch
        from qwen_tts import Qwen3TTSModel
        from qwen_tts.core.models.configuration_qwen3_tts import Qwen3TTSTalkerConfig

        _orig = Qwen3TTSTalkerConfig.__init__

        def __init__(self, *args, **kwargs):
            _orig(self, *args, **kwargs)
            if getattr(self, "pad_token_id", None) is None:
                self.pad_token_id = getattr(self, "codec_pad_id", 4196)

        Qwen3TTSTalkerConfig.__init__ = __init__

        dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
        _model = Qwen3TTSModel.from_pretrained(MODEL, device_map=_device(), dtype=dtype)
    return _model


def synth_voice(
    text: str,
    voice_biometrics: dict | str,
    voice_emotion: dict | str,
    language: str = "Russian",
    out_path: str | Path | None = None,
) -> str:
    """
    text — реплика для озвучки.
    voice_biometrics — dict или JSON с биологическими характеристиками голоса (можно str).
    voice_emotion — dict или JSON с эмоционально-просодическими характеристиками (можно str).
    language — целевой язык (Russian, English, Auto, …).
    Возвращает путь к сохранённому WAV.
    """
    if not (text or "").strip():
        raise ValueError("text must be a non-empty string")

    bio = _as_dict(voice_biometrics)
    emo = _as_dict(voice_emotion)

    instruct = json.dumps(
        {
            "voice_biometrics": bio,
            "voice_emotion": emo,
            "instruction": (
                "Keep voice_biometrics fixed. Apply voice_emotion for delivery. "
                "Natural speech. Speech is as very natural and lively as possible"
            ),
        },
        ensure_ascii=False,
        indent=2,
    )

    import soundfile as sf

    wavs, sr = _get_model().generate_voice_design(
        text=text,
        language=language,
        instruct=instruct,
    )

    if not wavs:
        raise RuntimeError("generate_voice_design returned no audio")

    if out_path is None:
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        out_path = OUT_DIR / f"{uuid.uuid4().hex[:8]}.wav"
    else:
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)

    sf.write(out_path, _to_wav_array(wavs[0]), sr)
    return str(out_path.resolve())


if __name__ == "__main__":
    path = synth_voice(
        text="Вариант передачи через строку JSON.",
        voice_biometrics='{"gender": "female", "age_group": "young", "voice_quality": "clear"}',
        voice_emotion='{"primary_emotion": "happy", "speech_rate": "normal"}',
        language="Russian",
    )
    print(path)
