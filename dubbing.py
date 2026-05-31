"""Озвучка реплики: Qwen3-TTS VoiceDesign по полу, возрасту и эмоции."""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import numpy as np

MODEL_ID = os.environ.get("SPEECHLAB_TTS_MODEL", "Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign")

_model: Any = None

PROMPT = (
    "Professional dubbing voice actor. Native {lang} speaker, studio quality, natural pace. "
    "{gender}. {age}. Delivery is **mostly neutral**; only a **subtle** hint of {emotion} — "
    "no exaggerated, theatrical, or cartoonish emotion. Clear diction, conversational."
)

MAP = {
    "age": {
        "child": "Child 7-12",
        "teenager": "Teen 14-18",
        "mature": "Adult 28-45",
        "elderly": "Senior 65-80",
        "ребенок": "Child 7-12",
        "ребёнок": "Child 7-12",
        "подросток": "Teen 14-18",
        "зрелый": "Adult 28-45",
        "пожилой": "Senior 65-80",
    },
    "gender": {
        "male": "Maximally masculine deep male voice",
        "female": "Maximally feminine bright female voice",
        "child": "Young child voice",
        "мужской": "Maximally masculine deep male voice",
        "женский": "Maximally feminine bright female voice",
    },
    "emotion": {
        "calm": "neutral-calm, understated",
        "neutral": "neutral, even delivery",
        "sad": "slightly subdued, subtle melancholy",
        "disgust": "mild displeasure, restrained",
        "discust": "mild displeasure, restrained",
        "happy": "lightly positive, not bubbly",
        "angry": "controlled irritation, not shouting",
        "fearful": "slight tension, restrained",
        "fear": "slight tension, restrained",
        "surprised": "mild surprise, understated",
        "surprise": "mild surprise, understated",
    },
    "lang": {
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
    },
}


def _map(key: str, value: str, default: str | None = None) -> str:
    v = (value or "").strip().lower()
    return MAP[key].get(v, default if default is not None else value)


def _get_model():
    global _model
    if _model is not None:
        return _model

    import torch
    from qwen_tts import Qwen3TTSModel
    from qwen_tts.core.models.configuration_qwen3_tts import Qwen3TTSTalkerConfig

    _orig = Qwen3TTSTalkerConfig.__init__

    def _init(self, *args, **kwargs):
        _orig(self, *args, **kwargs)
        if getattr(self, "pad_token_id", None) is None:
            self.pad_token_id = getattr(self, "codec_pad_id", 4196)

    Qwen3TTSTalkerConfig.__init__ = _init

    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    _model = Qwen3TTSModel.from_pretrained(
        MODEL_ID,
        device_map=device,
        dtype=dtype,
        attn_implementation="sdpa",
    )
    return _model


def unload_model() -> None:
    global _model
    import gc

    import torch

    _model = None
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def dub_tts(
    text: str,
    language: str,
    *,
    age: str,
    gender: str,
    emotion: str,
    out_path: str | Path,
) -> str:
    """Синтез одной реплики. Возвращает путь к WAV."""
    if not (text or "").strip():
        raise ValueError("text must be non-empty")

    lang = _map("lang", language, language)
    instruct = PROMPT.format(
        lang=lang,
        gender=_map("gender", gender),
        age=_map("age", age),
        emotion=_map("emotion", emotion),
    )

    import soundfile as sf

    wavs, sr = _get_model().generate_voice_design(
        text=text.strip(),
        language=lang,
        instruct=instruct,
        temperature=0.55,
        non_streaming_mode=True,
    )
    if not wavs:
        raise RuntimeError("generate_voice_design returned no audio")

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    arr = np.asarray(wavs[0], dtype=np.float32).squeeze()
    sf.write(out_path, arr, sr)
    return str(out_path.resolve())


def dub_from_profile(
    text: str,
    language: str,
    profile: dict,
    out_path: str | Path,
) -> str:
    """Озвучка по профилю из casting.json."""
    return dub_tts(
        text=text,
        language=language,
        age=profile.get("age_group", "mature"),
        gender=profile.get("gender", "male"),
        emotion=profile.get("primary_emotion", "calm"),
        out_path=out_path,
    )


def dub_from_voice_param(
    text: str,
    language: str,
    sex: str | dict,
    emotion: str | dict,
    out_path: str | Path,
) -> str:
    """Совместимость: sex/emotion как JSON или dict → profile."""
    s = json.loads(sex) if isinstance(sex, str) else sex
    e = json.loads(emotion) if isinstance(emotion, str) else emotion
    profile = {
        "age_group": s.get("age_group", "mature"),
        "gender": s.get("gender", "male"),
        "primary_emotion": e.get("primary_emotion", "calm"),
    }
    return dub_from_profile(text, language, profile, out_path)
