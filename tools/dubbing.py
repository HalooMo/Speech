"""Qwen3-TTS: озвучка дубляжа (PRD шаг 6).

Схема:
  1) ensure_voice_bank — 8 эталонов (male/female × 4 возраста):
     VoiceDesign ИЛИ клон из пользовательского сэмпла → .wav в voice_bank
  2) Base model create_voice_clone_prompt — эмбеддинг голоса по эталону
  3) dub_from_profile — синтез реплики по тексту + профилю из casting.json

Кэш: .speechlab_voice_bank/ или voice_bank/ проекта при кастомных промптах/сэмплах.
"""
import gc
import json
import os
import subprocess
from pathlib import Path

import numpy as np

# =============================================================================
# Модели Qwen, кэш эталонов, ключи голосов (8 = 2 пола × 4 возраста)
# =============================================================================
MODEL_DESIGN = os.environ.get("SPEECHLAB_TTS_DESIGN_MODEL", "Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign")
MODEL_BASE = os.environ.get("SPEECHLAB_TTS_BASE_MODEL", "Qwen/Qwen3-TTS-12Hz-1.7B-Base")
_env_cache = os.environ.get("SPEECHLAB_VOICE_CACHE", "")
CACHE_ROOT = Path(_env_cache).expanduser() if _env_cache else Path(__file__).resolve().parent.parent / ".speechlab_voice_bank"

GENDERS = ("male", "female")
AGE_GROUPS = ("child", "teenager", "mature", "elderly")
VOICE_KEYS = [f"{g}_{a}" for g in GENDERS for a in AGE_GROUPS]
MAP_LANG = {
    "ru": "Russian", "russian": "Russian", "en": "English", "english": "English",
    "de": "German", "german": "German", "es": "Spanish", "spanish": "Spanish",
    "fr": "French", "french": "French", "auto": "Auto",
}
GENDER_HINT = {"male": "masculine male voice", "female": "feminine female voice, warm timbre"}
AGE_HINT = {
    "child": "child",
    "teenager": "teen",
    "mature": "adult",
    "elderly": "adult",
}

# Тексты и шаблоны VoiceDesign по умолчанию (переопределяются через API / set_voice_prompts)
REF_TEXT = {
    "Russian": "Знаешь, я сейчас скажу как есть — без занудства, просто по-человечески, как в жизни разговаривают.",
    "English": "Look, I'll just say it the way people actually talk — not stiff, not like a presenter, just natural.",
    "German": "Ich sag's einfach so, wie man im echten Gespräch redet — locker und natürlich.",
    "Spanish": "Te lo digo como en la vida real, sin tono de locutor, natural y cercano.",
    "French": "Je le dis comme dans la vraie vie, pas comme à la radio — naturel et vivant.",
}
DESIGN_TEMPLATE = "Natural {gender_hint}, {lang}. {age_hint}."
REF_TEXT_BY_KEY = {}
DESIGN_BY_KEY = {}
DESIGN_TEMP = 0.72

# Ленивая загрузка моделей в VRAM (один экземпляр Design + Base на процесс)
_design_inst = None
_base_inst = None
# clone-prompts: ключ "Russian/male_mature" → объект для generate_voice_clone
_clone_prompts = {}

# Параметры текущего run() из main.run() / API (сбрасываются в clear_voice_prompts)
_active = {
    "template": None, "by_key": None, "cache_root": None,
    "gender": None, "age": None, "design_temp": None,
    "clone_by_key": None,  # voice_key → {path, ref_text}
    "cast_by_id": None,  # cast_id → {path, ref_text} — встроенные голоса data/Cast
    "cast_mode": None,  # None | "speakers" — раздача cast по спикерам
}


# --- Маппинг возраста в годах → возрастная группа для voice_key ---
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
    """male/female из API; иначе ValueError."""
    if not gender:
        return None
    g = str(gender).strip().lower()
    if g in ("m", "male", "man"):
        return "male"
    if g in ("f", "female", "woman"):
        return "female"
    raise ValueError(f"voice_gender: ожидается male/female, получено {gender!r}")


# =============================================================================
# Настройка голоса на один прогон run() (вызывается из main.run / server API)
# =============================================================================
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


def set_cast_voices(
    voices: list[dict] | None = None,
    *,
    mode: str | None = None,
) -> None:
    """Встроенные cast-голоса (Локи / Том Харди / Тор).

    voices: [{"id", "path", "ref_text"}, ...] — эталоны cast_<id> в банке.
    mode="speakers" — в casting каждому спикеру назначается cast_id по кругу.
    """
    if not voices:
        _active["cast_by_id"] = None
    else:
        by_id: dict[str, dict] = {}
        for v in voices:
            cid = str(v.get("id") or "").strip()
            if not cid:
                raise ValueError("cast_voices: нужен id")
            src = Path(v["path"])
            if not src.is_file():
                raise FileNotFoundError(f"cast sample: {src}")
            by_id[cid] = {
                "path": src.resolve(),
                "ref_text": (v.get("ref_text") or "").strip() or None,
            }
        _active["cast_by_id"] = by_id or None
    m = (mode or "").strip().lower() or None
    if m and m not in ("speakers",):
        raise ValueError(f"cast_mode: ожидается speakers, получено {mode!r}")
    _active["cast_mode"] = m


def cast_mode() -> str | None:
    return _active["cast_mode"]


def cast_ids() -> list[str]:
    return list((_active["cast_by_id"] or {}).keys())


def _norm_age_groups(age_groups) -> list[str]:
    """Пусто → все 4 группы; иначе подмножество child|teenager|mature|elderly."""
    if not age_groups:
        return list(AGE_GROUPS)
    out: list[str] = []
    for a in age_groups:
        key = str(a).strip().lower()
        if key in AGE_GROUPS and key not in out:
            out.append(key)
    if not out:
        raise ValueError(f"age_groups: ожидается подмножество {AGE_GROUPS}, получено {age_groups!r}")
    return out


# --- Пользовательские аудио-сэмплы: один WAV/MP3 → несколько voice_key по age_groups ---
def set_voice_clone_samples(samples: list[dict] | None) -> None:
    """Аудио-сэмплы для клонирования: пол + возрастные группы → voice_key.

    samples: [{"gender": "male"|"female", "path": str|Path,
               "age_groups": list[str]|None, "ref_text": str|None}, ...]
    age_groups=None → все 4 группы. ref_text=None → x_vector_only_mode (ниже качество).
    """
    if not samples:
        _active["clone_by_key"] = None
        return
    expanded: dict[str, dict] = {}
    for spec in samples:
        gender = _norm_gender(spec.get("gender"))
        if not gender:
            raise ValueError("voice_clone_samples: нужен gender (male/female)")
        src = Path(spec["path"])
        if not src.is_file():
            raise FileNotFoundError(f"voice clone sample: {src}")
        ref_text = (spec.get("ref_text") or "").strip() or None
        for age in _norm_age_groups(spec.get("age_groups")):
            vk = f"{gender}_{age}"
            expanded[vk] = {"path": src.resolve(), "ref_text": ref_text}
    _active["clone_by_key"] = expanded or None


# --- Флаги кастомизации голоса в текущем run() ---
def has_custom_voice_bank() -> bool:
    """Нужен ли отдельный voice_bank проекта (промпты и/или clone-сэмплы)."""
    a = _active
    return bool(
        a["template"] or a["by_key"] or a["design_temp"] is not None
        or a["clone_by_key"] or a["cast_by_id"]
    )


def uses_custom_voice() -> bool:
    return bool(_active["template"] or _active["by_key"] or _active["design_temp"] is not None)


def uses_custom_clone() -> bool:
    return bool(_active["clone_by_key"] or _active["cast_by_id"])


def has_voice_profile_override() -> bool:
    return bool(_active["gender"] or _active["age"] is not None)


# --- Подстановка voice_gender/voice_age из API поверх детекции по WAV (casting) ---
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
    """Корень voice_bank: глобальный кэш или папка проекта."""
    return _active["cache_root"] if _active["cache_root"] else CACHE_ROOT


def _settings_meta_path():
    return _bank_root() / "voice_settings.json"


def _clone_samples_meta() -> dict:
    """Снимок путей/mtime сэмплов — для инвалидации банка при смене файла."""
    if not _active["clone_by_key"]:
        return {}
    meta = {}
    for vk, spec in _active["clone_by_key"].items():
        p = Path(spec["path"])
        meta[vk] = {
            "path": str(p.resolve()),
            "mtime": p.stat().st_mtime if p.is_file() else 0,
            "ref_text": spec.get("ref_text"),
        }
    return meta


def _cast_samples_meta() -> dict:
    """Снимок встроенных cast-сэмплов для voice_settings.json."""
    if not _active["cast_by_id"]:
        return {}
    meta = {}
    for cid, spec in _active["cast_by_id"].items():
        p = Path(spec["path"])
        meta[cid] = {
            "path": str(p.resolve()),
            "mtime": p.stat().st_mtime if p.is_file() else 0,
            "ref_text": spec.get("ref_text"),
        }
    return meta


def _current_settings_meta():
    """Текущие промпты/температура/сэмплы — сравнивается с voice_settings.json."""
    return {
        "prompts": {
            "template": _active["template"],
            "by_key": _active["by_key"] or {},
            "design_temperature": _design_temp(),
        },
        "clone_samples": _clone_samples_meta(),
        "cast_samples": _cast_samples_meta(),
        "cast_mode": _active["cast_mode"],
    }


def _settings_changed():
    """Кастомные настройки изменились — нужна перегенерация банка."""
    if not has_custom_voice_bank():
        return False
    p = _settings_meta_path()
    if not p.is_file():
        return True
    try:
        saved = json.loads(p.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return True
    return saved != _current_settings_meta()


def _save_settings_meta():
    """Записать voice_settings.json после пересборки банка."""
    if not has_custom_voice_bank():
        return
    _bank_root().mkdir(parents=True, exist_ok=True)
    _settings_meta_path().write_text(
        json.dumps(_current_settings_meta(), ensure_ascii=False, indent=2), encoding="utf-8",
    )


# =============================================================================
# Тексты для VoiceDesign и выбор voice_key из casting profile
# =============================================================================
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
    """casting profile → male_mature / cast_loki и т.п. для выбора эталона."""
    # Явный cast_id (режим speakers) имеет приоритет над пол×возраст
    cast_id = (profile.get("cast_id") or profile.get("cast_voice") or "").strip()
    if cast_id:
        from tools.cast_voices import voice_key_for_cast
        return voice_key_for_cast(cast_id)
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
    """Подкаталог банка по языку: .speechlab_voice_bank/russian/…"""
    return _bank_root() / lang.lower().replace(" ", "_")


def _clone_spec(voice_key: str) -> dict | None:
    """Спека пользовательского clone-сэмпла для voice_key или None."""
    by_key = _active["clone_by_key"]
    if by_key and voice_key in by_key:
        return by_key[voice_key]
    # cast_loki → запись в cast_by_id
    if voice_key.startswith("cast_") and _active["cast_by_id"]:
        cid = voice_key[len("cast_"):]
        return _active["cast_by_id"].get(cid)
    return None


def _bank_voice_keys() -> list[str]:
    """8 слотов пол×возраст + активные cast_*."""
    keys = list(VOICE_KEYS)
    for cid in (_active["cast_by_id"] or {}):
        vk = f"cast_{cid}"
        if vk not in keys:
            keys.append(vk)
    return keys


# =============================================================================
# Подготовка эталонных WAV и загрузка Qwen3-TTS
# =============================================================================
def _prepare_ref_wav(src: Path, dst: Path) -> None:
    """mp3/wav → mono WAV для Qwen clone (ffmpeg)."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg", "-y", "-i", str(src),
        "-vn", "-acodec", "pcm_s16le", "-ar", "24000", "-ac", "1",
        str(dst),
    ]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        tail = (r.stderr or "")[-2000:]
        raise RuntimeError(f"ffmpeg не конвертировал {src.name}: {tail}")


def _install_clone_sample(spec: dict, wav_path: Path, lang: str, voice_key: str) -> None:
    """Пользовательский сэмпл → эталон в voice_bank."""
    src = Path(spec["path"])
    _prepare_ref_wav(src, wav_path)
    ref_text = spec.get("ref_text")
    txt_path = wav_path.with_suffix(".txt")
    if ref_text:
        txt_path.write_text(ref_text, encoding="utf-8")
    elif txt_path.is_file():
        txt_path.unlink()
    meta = {
        "voice_key": voice_key,
        "source": str(src.resolve()),
        "ref_text": ref_text,
        "custom_clone": True,
    }
    (wav_path.parent / f"{voice_key}.meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8",
    )
    print(f"    {voice_key} ← clone sample ({src.name})")


def _patch_talker():
    """Фикс pad_token_id в Qwen3 (иначе generate падает на некоторых конфигах)."""
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
    """Загрузить Qwen3TTSModel на cuda:0 или CPU."""
    import torch
    from qwen_tts import Qwen3TTSModel
    _patch_talker()
    dev = "cuda:0" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    return Qwen3TTSModel.from_pretrained(model_id, device_map=dev, dtype=dtype, attn_implementation="sdpa")


def _design_model():
    """Синглтон VoiceDesign (создаёт эталонные WAV)."""
    global _design_inst
    if _design_inst is None:
        print("  TTS: VoiceDesign…")
        _design_inst = _load_qwen(MODEL_DESIGN)
    return _design_inst


def _base_model():
    """Синглтон Base (clone prompt + синтез реплик)."""
    global _base_inst
    if _base_inst is None:
        print("  TTS: Base clone…")
        _base_inst = _load_qwen(MODEL_BASE)
    return _base_inst


def _free_design():
    """Выгрузить VoiceDesign из VRAM после генерации эталонов."""
    global _design_inst
    import torch
    _design_inst = None
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# =============================================================================
# Сборка voice_bank: 8 эталонов + create_voice_clone_prompt для каждого
# =============================================================================
def ensure_voice_bank(language, force=False):
    """Создать эталоны .wav + clone-prompts в кэше (8 слотов + cast_*)."""
    global _clone_prompts
    lang = map_lang(language)
    cache = _cache_dir(lang)
    cache.mkdir(parents=True, exist_ok=True)
    bank_keys = _bank_voice_keys()

    # Смена промптов/сэмплов → пересоздать банк
    if has_custom_voice_bank() and _settings_changed():
        force = True

    need = force or not all((cache / f"{vk}.wav").is_file() for vk in bank_keys)
    if need:
        import soundfile as sf
        clone_keys = set(_active["clone_by_key"] or {})
        # Встроенные cast_* всегда ставятся из сэмпла, не через VoiceDesign
        for cid in (_active["cast_by_id"] or {}):
            clone_keys.add(f"cast_{cid}")
        # Ключи без пользовательского сэмпла — генерируем через VoiceDesign
        design_keys = [vk for vk in bank_keys if vk not in clone_keys]
        if clone_keys:
            print(f"  TTS: clone samples ({lang})…")
            for vk in bank_keys:
                spec = _clone_spec(vk)
                if not spec:
                    continue
                wav_path = cache / f"{vk}.wav"
                if wav_path.is_file() and not force:
                    continue
                _install_clone_sample(spec, wav_path, lang, vk)
        if design_keys:
            print(f"  TTS: VoiceDesign ({len(design_keys)} голосов, {lang})…")
            design = _design_model()
            for vk in design_keys:
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
                meta = {
                    "voice_key": vk, "ref_text": line,
                    "design_instruct": instr, "design_temperature": temp,
                }
                (cache / f"{vk}.meta.json").write_text(
                    json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8",
                )
                print(f"    {vk}")
            _free_design()
        _save_settings_meta()

    if force:
        _clone_prompts.clear()
    # Base model: из каждого .wav строим voice_clone_prompt (кэш в памяти)
    print(f"  TTS: clone-prompts ({lang})…")
    base = _base_model()
    for vk in bank_keys:
        key = f"{lang}/{vk}"
        if key in _clone_prompts and not force:
            continue
        wav_path = cache / f"{vk}.wav"
        txt_path = cache / f"{vk}.txt"
        if not wav_path.is_file():
            raise FileNotFoundError(wav_path)
        spec = _clone_spec(vk)
        # ref_text обязателен для качества; без него — x_vector_only_mode (хуже, но работает)
        if spec and spec.get("ref_text"):
            line = spec["ref_text"]
            x_vector = False
        elif txt_path.is_file():
            line = txt_path.read_text(encoding="utf-8").strip()
            x_vector = False
        else:
            line = ref_line(lang, vk)
            x_vector = bool(spec)
        _clone_prompts[key] = base.create_voice_clone_prompt(
            ref_audio=str(wav_path),
            ref_text=line,
            x_vector_only_mode=x_vector,
        )


# =============================================================================
# Синтез одной реплики (вызывается из main.dub_segments)
# =============================================================================
def dub_tts(text, language, gender, age, out_path):
    """Озвучить одну реплику по полу/возрасту."""
    return dub_from_profile(
        text, language, {"gender": gender, "age_group": age}, out_path,
    )


def dub_from_profile(text, language, profile, out_path, *, bank_ready=False):
    """Озвучка по casting.json через Qwen3-TTS."""
    if not (text or "").strip():
        raise ValueError("text пустой")
    profile = apply_voice_override(profile or {})
    lang = map_lang(language)

    vk = normalize_voice_key(profile)
    if not bank_ready:
        ensure_voice_bank(lang)
    key = f"{lang}/{vk}"
    prompt = _clone_prompts.get(key)
    if prompt is None:
        # Банк мог обновиться — пересобрать prompts
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
    """Освободить VRAM после этапа TTS (Qwen)."""
    global _design_inst, _base_inst, _clone_prompts
    import torch
    _design_inst = _base_inst = None
    _clone_prompts = {}
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
