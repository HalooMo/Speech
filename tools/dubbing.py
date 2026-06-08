"""Qwen TTS: 8 эталонов (пол×возраст) → voice clone."""
import gc
import json
import os
from pathlib import Path

import numpy as np

# --- модели и кэш ---
MODEL_DESIGN = os.environ.get("SPEECHLAB_TTS_DESIGN_MODEL", "Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign")
MODEL_BASE = os.environ.get("SPEECHLAB_TTS_BASE_MODEL", "Qwen/Qwen3-TTS-12Hz-1.7B-Base")
_env_cache = os.environ.get("SPEECHLAB_VOICE_CACHE", "")
CACHE_ROOT = Path(_env_cache).expanduser() if _env_cache else Path(__file__).resolve().parent.parent / ".speechlab_voice_bank"

VOICE_KEYS = [f"{g}_{a}" for g in ("male", "female") for a in ("child", "teenager", "mature", "elderly")]
MAP_LANG = {
    "ru": "Russian", "russian": "Russian", "en": "English", "english": "English",
    "de": "German", "german": "German", "es": "Spanish", "spanish": "Spanish",
    "fr": "French", "french": "French", "auto": "Auto",
}
GENDER_HINT = {"male": "masculine male voice, warm timbre", "female": "feminine female voice, warm timbre"}
AGE_HINT = {
    "child": "child 8-12, bright and spontaneous",
    "teenager": "teen 15-18, energetic, slightly uneven",
    "mature": "adult 28-45, relaxed and natural",
    "elderly": "senior 65-75, warm, unhurried",
}

# промпты VoiceDesign (редактируйте здесь)
REF_TEXT = {
    "Russian": "Знаешь, я сейчас скажу как есть — без занудства, просто по-человечески, как в жизни разговаривают.",
    "English": "Look, I'll just say it the way people actually talk — not stiff, not like a presenter, just natural.",
    "German": "Ich sag's einfach so, wie man im echten Gespräch redet — locker und natürlich.",
    "Spanish": "Te lo digo como en la vida real, sin tono de locutor, natural y cercano.",
    "French": "Je le dis comme dans la vraie vie, pas comme à la radio — naturel et vivant.",
}
DESIGN_TEMPLATE = "Film dubbing voice, native {lang}. {gender_hint}. {age_hint}. Sounds alive and human"
REF_TEXT_BY_KEY = {}
DESIGN_BY_KEY = {}
DESIGN_TEMP = 0.72

_design_inst = None
_base_inst = None
_clone_prompts = {}

# переопределения на один прогон run() (None → дефолты DESIGN_* / детекция по реплике)
_active = {
    "template": None, "by_key": None, "cache_root": None,
    "gender": None, "age": None, "design_temp": None,
}


def age_to_group(age):
    """Возраст (лет) → child | teenager | mature | elderly."""
    age = float(age)
    if age < 13:
        return "child"
    if age < 20:
        return "teenager"
    if age < 55:
        return "mature"
    return "elderly"


def _norm_gender(gender):
    if not gender:
        return None
    g = str(gender).strip().lower()
    if g in ("m", "male", "man"):
        return "male"
    if g in ("f", "female", "woman"):
        return "female"
    raise ValueError(f"voice_gender: ожидается male/female, получено {gender!r}")


def set_voice_prompts(
    template=None,
    by_key=None,
    cache_dir=None,
    *,
    gender=None,
    age=None,
    design_temperature=None,
):
    """Параметры голоса для текущего видео. cache_dir — папка проекта (voice_bank)."""
    _active["template"] = (template or "").strip() or None
    _active["by_key"] = dict(by_key) if by_key else None
    _active["cache_root"] = Path(cache_dir) if cache_dir else None
    _active["gender"] = _norm_gender(gender)
    _active["age"] = float(age) if age is not None else None
    _active["design_temp"] = float(design_temperature) if design_temperature is not None else None


def clear_voice_prompts():
    """Сброс после run()."""
    for k in _active:
        _active[k] = None


def uses_custom_voice():
    """Есть ли кастомные промпты/стиль в текущем прогоне."""
    return bool(
        _active["template"] or _active["by_key"] or _active["design_temp"] is not None,
    )


def has_voice_profile_override():
    """Заданы ли пол/возраст для всех реплик."""
    return bool(_active["gender"] or _active["age"] is not None)


def apply_voice_override(profile):
    """Подставить voice_gender / voice_age из run() в профиль casting."""
    p = dict(profile or {})
    if _active["gender"]:
        p["gender"] = _active["gender"]
    if _active["age"] is not None:
        age = float(_active["age"])
        p["age"] = round(age, 1)
        p["age_group"] = age_to_group(age)
    return p


def _design_temp():
    return _active["design_temp"] if _active["design_temp"] is not None else DESIGN_TEMP


def _bank_root():
    return _active["cache_root"] if _active["cache_root"] else CACHE_ROOT


def _prompts_meta_path():
    return _bank_root() / "voice_prompts.json"


def _current_prompts_meta():
    return {
        "template": _active["template"],
        "by_key": _active["by_key"] or {},
        "design_temperature": _design_temp(),
    }


def _prompts_changed():
    """Кастомные промпты изменились — нужна перегенерация банка."""
    if not uses_custom_voice():
        return False
    p = _prompts_meta_path()
    if not p.is_file():
        return True
    try:
        saved = json.loads(p.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return True
    return saved != _current_prompts_meta()


def _save_prompts_meta():
    if not uses_custom_voice():
        return
    _bank_root().mkdir(parents=True, exist_ok=True)
    _prompts_meta_path().write_text(
        json.dumps(_current_prompts_meta(), ensure_ascii=False, indent=2), encoding="utf-8",
    )


def map_lang(language):
    """Код языка → имя для Qwen."""
    return MAP_LANG.get((language or "").strip().lower(), language)


def ref_line(lang, voice_key):
    """Текст эталона для VoiceDesign и clone."""
    if voice_key in REF_TEXT_BY_KEY:
        return REF_TEXT_BY_KEY[voice_key]
    return REF_TEXT.get(lang, REF_TEXT["English"])


def design_instruct(lang, gender, age, voice_key):
    """Описание голоса для VoiceDesign."""
    by_key = {**DESIGN_BY_KEY, **(_active["by_key"] or {})}
    if voice_key in by_key:
        return by_key[voice_key]
    tmpl = _active["template"] or DESIGN_TEMPLATE
    gender_hint = GENDER_HINT.get(gender, gender)
    if isinstance(age, (int, float)) or (isinstance(age, str) and age.replace(".", "", 1).isdigit()):
        age_hint = f"speaker age {int(round(float(age)))}, natural delivery"
    else:
        age_hint = AGE_HINT.get(age, age)
    return tmpl.format(lang=lang, gender_hint=gender_hint, age_hint=age_hint)


def normalize_voice_key(profile):
    """casting profile → ключ male_mature и т.п."""
    gender = (profile.get("gender") or "male").strip().lower()
    age = (profile.get("age_group") or "mature").strip().lower()
    if gender == "child":
        probs = profile.get("gender_probs") or {}
        gender = "female" if probs.get("female", 0) > probs.get("male", 0) else "male"
    if gender not in ("male", "female"):
        gender = "male"
    if age not in ("child", "teenager", "mature", "elderly"):
        age = "mature"
    return f"{gender}_{age}"


def _cache_dir(lang):
    return _bank_root() / lang.lower().replace(" ", "_")


def _patch_talker():
    """Фикс pad_token_id в Qwen3."""
    from qwen_tts.core.models.configuration_qwen3_tts import Qwen3TTSTalkerConfig
    if getattr(Qwen3TTSTalkerConfig, "_speechlab_patched", False):
        return
    orig = Qwen3TTSTalkerConfig.__init__

    def _init(self, *a, **kw):
        orig(self, *a, **kw)
        if getattr(self, "pad_token_id", None) is None:
            self.pad_token_id = getattr(self, "codec_pad_id", 4196)

    Qwen3TTSTalkerConfig.__init__ = _init
    Qwen3TTSTalkerConfig._speechlab_patched = True


def _load_qwen(model_id):
    """Загрузить Qwen3TTSModel."""
    import torch
    from qwen_tts import Qwen3TTSModel
    _patch_talker()
    dev = "cuda:0" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    return Qwen3TTSModel.from_pretrained(model_id, device_map=dev, dtype=dtype, attn_implementation="sdpa")


def _design_model():
    global _design_inst
    if _design_inst is None:
        print("  TTS: VoiceDesign…")
        _design_inst = _load_qwen(MODEL_DESIGN)
    return _design_inst


def _base_model():
    global _base_inst
    if _base_inst is None:
        print("  TTS: Base clone…")
        _base_inst = _load_qwen(MODEL_BASE)
    return _base_inst


def _free_design():
    global _design_inst
    import torch
    _design_inst = None
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def ensure_voice_bank(language, force=False):
    """Создать 8 эталонов .wav + clone-prompts в кэше."""
    global _clone_prompts
    lang = map_lang(language)
    cache = _cache_dir(lang)
    cache.mkdir(parents=True, exist_ok=True)

    if uses_custom_voice() and _prompts_changed():
        force = True

    need = force or not all((cache / f"{vk}.wav").is_file() for vk in VOICE_KEYS)
    if need:
        import soundfile as sf
        print(f"  TTS: 8 голосов ({lang})…")
        design = _design_model()
        for vk in VOICE_KEYS:
            gender, age = vk.split("_", 1)
            wav_path = cache / f"{vk}.wav"
            if wav_path.is_file() and not force:
                continue
            line = ref_line(lang, vk)
            instr = design_instruct(lang, gender, age, vk)
            temp = _design_temp()
            wavs, sr = design.generate_voice_design(
                text=line, language=lang, instruct=instr,
                temperature=temp, non_streaming_mode=True,
            )
            if not wavs:
                raise RuntimeError(f"VoiceDesign: нет аудио для {vk}")
            sf.write(wav_path, np.asarray(wavs[0], dtype=np.float32).squeeze(), sr)
            (cache / f"{vk}.txt").write_text(line, encoding="utf-8")
            (cache / f"{vk}.instruct.txt").write_text(instr, encoding="utf-8")
            meta = {"voice_key": vk, "ref_text": line, "design_instruct": instr, "design_temperature": temp}
            (cache / f"{vk}.meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
            print(f"    {vk}")
        _free_design()
        _save_prompts_meta()

    if force:
        _clone_prompts.clear()
    print(f"  TTS: clone-prompts ({lang})…")
    base = _base_model()
    for vk in VOICE_KEYS:
        key = f"{lang}/{vk}"
        if key in _clone_prompts and not force:
            continue
        wav_path = cache / f"{vk}.wav"
        txt_path = cache / f"{vk}.txt"
        if not wav_path.is_file():
            raise FileNotFoundError(wav_path)
        line = txt_path.read_text(encoding="utf-8").strip() if txt_path.is_file() else ref_line(lang, vk)
        _clone_prompts[key] = base.create_voice_clone_prompt(
            ref_audio=str(wav_path), ref_text=line, x_vector_only_mode=False,
        )


def dub_tts(text, language, gender, age, out_path):
    """Озвучить одну реплику по полу/возрасту."""
    return dub_from_profile(
        text, language, {"gender": gender, "age_group": age}, out_path,
    )


def dub_from_profile(text, language, profile, out_path):
    """Озвучка по dict из casting.json."""
    if not (text or "").strip():
        raise ValueError("text пустой")
    profile = apply_voice_override(profile or {})
    lang = map_lang(language)
    vk = normalize_voice_key(profile)
    ensure_voice_bank(lang)
    key = f"{lang}/{vk}"
    prompt = _clone_prompts.get(key)
    if prompt is None:
        ensure_voice_bank(lang, force=True)
        prompt = _clone_prompts[key]
    import soundfile as sf
    wavs, sr = _base_model().generate_voice_clone(
        text=text.strip(), language=lang, voice_clone_prompt=prompt, non_streaming_mode=True,
    )
    if not wavs:
        raise RuntimeError("generate_voice_clone: нет аудио")
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(out_path, np.asarray(wavs[0], dtype=np.float32).squeeze(), sr)
    return str(out_path.resolve())


def unload_model():
    """Освободить VRAM."""
    global _design_inst, _base_inst, _clone_prompts
    import torch
    _design_inst = _base_inst = None
    _clone_prompts = {}
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
