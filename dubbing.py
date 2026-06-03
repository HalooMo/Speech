"""
Озвучка: 8 базовых типов (2 пола × 4 возраста), без эмоций.

1) VoiceDesign — эталон по промптам из VoiceBankConfig.
2) Base — clone по эталону.

Промпты: блок DEFAULT_* ниже или config= / kwargs в ensure_voice_bank().
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

import numpy as np

MODEL_DESIGN = os.environ.get(
    "SPEECHLAB_TTS_DESIGN_MODEL", "Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign"
)
MODEL_BASE = os.environ.get(
    "SPEECHLAB_TTS_BASE_MODEL", "Qwen/Qwen3-TTS-12Hz-1.7B-Base"
)
CACHE_ROOT = Path(
    os.environ.get("SPEECHLAB_VOICE_CACHE", "")
).expanduser() if os.environ.get("SPEECHLAB_VOICE_CACHE") else (
    Path(__file__).resolve().parent / ".speechlab_voice_bank"
)

GENDERS = ("male", "female")
AGES = ("child", "teenager", "mature", "elderly")
VOICE_KEYS = tuple(f"{g}_{a}" for g in GENDERS for a in AGES)

MAP_LANG = {
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

GENDER_HINT = {
    "male": "masculine male voice, warm timbre",
    "female": "feminine female voice, warm timbre",
}

AGE_HINT = {
    "child": "child 8-12, bright and spontaneous",
    "teenager": "teen 15-18, energetic, slightly uneven",
    "mature": "adult 28-45, relaxed and natural",
    "elderly": "senior 65-75, warm, unhurried",
}

# =============================================================================
# Промпты по умолчанию — редактируйте здесь
# =============================================================================

DEFAULT_REF_TEXT: dict[str, str] = {
    "Russian": (
        "Знаешь, я сейчас скажу как есть — без занудства, просто по-человечески, "
        "как в жизни разговаривают."
    ),
    "English": (
        "Look, I'll just say it the way people actually talk — not stiff, "
        "not like a presenter, just natural."
    ),
    "German": "Ich sag's einfach so, wie man im echten Gespräch redet — locker und natürlich.",
    "Spanish": "Te lo digo como en la vida real, sin tono de locutor, natural y cercano.",
    "French": "Je le dis comme dans la vraie vie, pas comme à la radio — naturel et vivant.",
}

# Шаблон VoiceDesign: {lang}, {gender_hint}, {age_hint}
DEFAULT_DESIGN_INSTRUCT_TEMPLATE = (
    "Film dubbing voice, native {lang}. {gender_hint}. {age_hint}. "
    "Sounds alive and human: uneven pace, small breaths, casual intonation — "
    "like a person in a scene, not a studio announcer. Warm, conversational, slightly playful."
)

# Полная замена для ключа: "male_mature", "female_teenager", …
DEFAULT_DESIGN_INSTRUCT_BY_KEY: dict[str, str] = {}
DEFAULT_REF_TEXT_BY_KEY: dict[str, str] = {}

DEFAULT_DESIGN_TEMPERATURE = 0.72


@dataclass
class VoiceBankConfig:
    """
    Промпты для 8 эталонов и клонирования.

    ref_text — фраза для VoiceDesign и ref_text у Base clone.
    design_instruct_template — шаблон с {lang}, {gender_hint}, {age_hint}.
    design_instruct_by_key / ref_text_by_key — переопределение на тип голоса.
    """

    ref_text: dict[str, str] = field(default_factory=lambda: dict(DEFAULT_REF_TEXT))
    design_instruct_template: str = DEFAULT_DESIGN_INSTRUCT_TEMPLATE
    design_instruct_by_key: dict[str, str] = field(
        default_factory=lambda: dict(DEFAULT_DESIGN_INSTRUCT_BY_KEY)
    )
    ref_text_by_key: dict[str, str] = field(
        default_factory=lambda: dict(DEFAULT_REF_TEXT_BY_KEY)
    )
    design_temperature: float = DEFAULT_DESIGN_TEMPERATURE

    def ref_line(self, lang: str, voice_key: str) -> str:
        if voice_key in self.ref_text_by_key:
            return self.ref_text_by_key[voice_key]
        return self.ref_text.get(lang, self.ref_text.get("English", ""))

    def design_instruct(self, lang: str, gender: str, age: str, voice_key: str) -> str:
        if voice_key in self.design_instruct_by_key:
            return self.design_instruct_by_key[voice_key]
        return self.design_instruct_template.format(
            lang=lang,
            gender_hint=GENDER_HINT.get(gender, gender),
            age_hint=AGE_HINT.get(age, age),
        )


DEFAULT_VOICE_BANK = VoiceBankConfig()

_design_model: Any = None
_base_model: Any = None
_clone_prompts: dict[str, Any] = {}
_active_config: VoiceBankConfig = DEFAULT_VOICE_BANK


def _map_lang(language: str) -> str:
    return MAP_LANG.get((language or "").strip().lower(), language)


def set_voice_bank_config(config: VoiceBankConfig) -> None:
    """Задать промпты глобально до ensure_voice_bank / dub_tts."""
    global _active_config
    _active_config = config


def _patch_talker_config() -> None:
    from qwen_tts.core.models.configuration_qwen3_tts import Qwen3TTSTalkerConfig

    if getattr(Qwen3TTSTalkerConfig, "_speechlab_patched", False):
        return
    _orig = Qwen3TTSTalkerConfig.__init__

    def _init(self, *args, **kwargs):
        _orig(self, *args, **kwargs)
        if getattr(self, "pad_token_id", None) is None:
            self.pad_token_id = getattr(self, "codec_pad_id", 4196)

    Qwen3TTSTalkerConfig.__init__ = _init
    Qwen3TTSTalkerConfig._speechlab_patched = True


def _load_model(model_id: str) -> Any:
    import torch
    from qwen_tts import Qwen3TTSModel

    _patch_talker_config()
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    return Qwen3TTSModel.from_pretrained(
        model_id,
        device_map=device,
        dtype=dtype,
        attn_implementation="sdpa",
    )


def _get_design_model() -> Any:
    global _design_model
    if _design_model is None:
        print("  TTS: загрузка VoiceDesign…")
        _design_model = _load_model(MODEL_DESIGN)
    return _design_model


def _get_base_model() -> Any:
    global _base_model
    if _base_model is None:
        print("  TTS: загрузка Base (clone)…")
        _base_model = _load_model(MODEL_BASE)
    return _base_model


def _cache_dir(lang: str) -> Path:
    return CACHE_ROOT / lang.lower().replace(" ", "_")


def normalize_voice_key(profile: dict) -> str:
    gender = (profile.get("gender") or "male").strip().lower()
    age = (profile.get("age_group") or "mature").strip().lower()
    if gender == "child":
        probs = profile.get("gender_probs") or {}
        gender = "female" if probs.get("female", 0) > probs.get("male", 0) else "male"
    if gender not in GENDERS:
        gender = "male"
    if age not in AGES:
        age = "mature"
    return f"{gender}_{age}"


def _prompt_cache_key(lang: str, voice_key: str) -> str:
    return f"{lang}/{voice_key}"


def _write_voice_meta(cache: Path, vk: str, config: VoiceBankConfig, lang: str) -> None:
    gender, age = vk.split("_", 1)
    meta = {
        "voice_key": vk,
        "ref_text": config.ref_line(lang, vk),
        "design_instruct": config.design_instruct(lang, gender, age, vk),
        "design_temperature": config.design_temperature,
    }
    (cache / f"{vk}.meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def ensure_voice_bank(
    language: str,
    *,
    force: bool = False,
    config: VoiceBankConfig | None = None,
    ref_text: dict[str, str] | str | None = None,
    design_instruct_template: str | None = None,
    design_instruct_by_key: dict[str, str] | None = None,
    ref_text_by_key: dict[str, str] | None = None,
    design_temperature: float | None = None,
) -> None:
    """
    8 эталонов + clone-prompts.

    config — полный набор; или отдельные kwargs (мержатся с текущим).
    После смены промптов: force=True или удалите .speechlab_voice_bank/.
    """
    global _active_config, _clone_prompts

    cfg = config or _active_config
    if config is None and any(
        x is not None
        for x in (
            ref_text,
            design_instruct_template,
            design_instruct_by_key,
            ref_text_by_key,
            design_temperature,
        )
    ):
        cfg = replace(cfg)
        if isinstance(ref_text, str):
            cfg.ref_text = {**cfg.ref_text, _map_lang(language): ref_text}
        elif isinstance(ref_text, dict):
            cfg.ref_text = {**cfg.ref_text, **ref_text}
        if design_instruct_template is not None:
            cfg.design_instruct_template = design_instruct_template
        if design_instruct_by_key is not None:
            cfg.design_instruct_by_key = {**cfg.design_instruct_by_key, **design_instruct_by_key}
        if ref_text_by_key is not None:
            cfg.ref_text_by_key = {**cfg.ref_text_by_key, **ref_text_by_key}
        if design_temperature is not None:
            cfg.design_temperature = design_temperature

    _active_config = cfg
    lang = _map_lang(language)
    cache = _cache_dir(lang)
    cache.mkdir(parents=True, exist_ok=True)

    need_design = force or not all((cache / f"{vk}.wav").is_file() for vk in VOICE_KEYS)

    if need_design:
        print(f"  TTS: 8 голосов ({lang}), T={cfg.design_temperature}…")
        design = _get_design_model()
        import soundfile as sf

        for vk in VOICE_KEYS:
            gender, age = vk.split("_", 1)
            wav_path = cache / f"{vk}.wav"
            if wav_path.is_file() and not force:
                continue
            ref_line = cfg.ref_line(lang, vk)
            instruct = cfg.design_instruct(lang, gender, age, vk)
            wavs, sr = design.generate_voice_design(
                text=ref_line,
                language=lang,
                instruct=instruct,
                temperature=cfg.design_temperature,
                non_streaming_mode=True,
            )
            if not wavs:
                raise RuntimeError(f"VoiceDesign: нет аудио для {vk}")
            sf.write(wav_path, np.asarray(wavs[0], dtype=np.float32).squeeze(), sr)
            (cache / f"{vk}.txt").write_text(ref_line, encoding="utf-8")
            (cache / f"{vk}.instruct.txt").write_text(instruct, encoding="utf-8")
            _write_voice_meta(cache, vk, cfg, lang)
            print(f"    {vk}")

        _unload_design_only()

    if force:
        _clone_prompts.clear()

    print(f"  TTS: clone-prompts ({lang})…")
    base = _get_base_model()
    for vk in VOICE_KEYS:
        pkey = _prompt_cache_key(lang, vk)
        if pkey in _clone_prompts and not force:
            continue
        wav_path = cache / f"{vk}.wav"
        txt_path = cache / f"{vk}.txt"
        if not wav_path.is_file():
            raise FileNotFoundError(f"Нет {wav_path}")
        ref_line = txt_path.read_text(encoding="utf-8").strip() if txt_path.is_file() else cfg.ref_line(lang, vk)
        _clone_prompts[pkey] = base.create_voice_clone_prompt(
            ref_audio=str(wav_path),
            ref_text=ref_line,
            x_vector_only_mode=False,
        )


def dub_tts(
    text: str,
    language: str,
    *,
    gender: str,
    age: str,
    out_path: str | Path,
    config: VoiceBankConfig | None = None,
) -> str:
    if not (text or "").strip():
        raise ValueError("text must be non-empty")
    if config is not None:
        set_voice_bank_config(config)

    lang = _map_lang(language)
    voice_key = normalize_voice_key({"gender": gender, "age_group": age})
    ensure_voice_bank(lang)

    pkey = _prompt_cache_key(lang, voice_key)
    prompt = _clone_prompts.get(pkey)
    if prompt is None:
        ensure_voice_bank(lang, force=True)
        prompt = _clone_prompts[pkey]

    import soundfile as sf

    wavs, sr = _get_base_model().generate_voice_clone(
        text=text.strip(),
        language=lang,
        voice_clone_prompt=prompt,
        non_streaming_mode=True,
    )
    if not wavs:
        raise RuntimeError("generate_voice_clone returned no audio")

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(out_path, np.asarray(wavs[0], dtype=np.float32).squeeze(), sr)
    return str(out_path.resolve())


def dub_from_profile(
    text: str,
    language: str,
    profile: dict,
    out_path: str | Path,
    *,
    config: VoiceBankConfig | None = None,
) -> str:
    return dub_tts(
        text=text,
        language=language,
        gender=profile.get("gender", "male"),
        age=profile.get("age_group", "mature"),
        out_path=out_path,
        config=config,
    )


def dub_from_voice_param(
    text: str,
    language: str,
    sex: str | dict,
    emotion: str | dict,
    out_path: str | Path,
) -> str:
    s = json.loads(sex) if isinstance(sex, str) else sex
    return dub_from_profile(text, language, s, out_path)


def _unload_design_only() -> None:
    global _design_model
    import gc
    import torch

    _design_model = None
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def unload_model() -> None:
    global _design_model, _base_model, _clone_prompts
    import gc
    import torch

    _design_model = None
    _base_model = None
    _clone_prompts = {}
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
