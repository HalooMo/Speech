"""Fish Audio TTS: озвучка дубляжа через клонирование (PRD шаг 6).

Схема:
  1) ensure_voice_bank — Fish voices.create из сэмплов (clone / cast)
  2) dub_from_profile — s2.1-pro + reference_id; эмоции уже в тексте ([brackets])

Нужен voice_clone_samples и/или cast_voice / cast_mode.
"""
from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

# Модель TTS: самая мощная в линейке Fish (заголовок model)
FISH_MODEL = os.environ.get("SPEECHLAB_FISH_MODEL", "s2.1-pro")
_env_cache = os.environ.get("SPEECHLAB_VOICE_CACHE", "")
CACHE_ROOT = (
    Path(_env_cache).expanduser()
    if _env_cache
    else Path(__file__).resolve().parent.parent / ".speechlab_voice_bank"
)

AGE_GROUPS = ("child", "teenager", "mature", "elderly")

# reference_id по ключу "male_mature" / "cast_loki"
_voice_ids: dict[str, str] = {}
_client = None

_active = {
    "cache_root": None,
    "gender": None,
    "age": None,
    "clone_by_key": None,  # voice_key → {path, ref_text}
    "cast_by_id": None,  # cast_id → {path, ref_text}
    "cast_mode": None,  # None | "speakers"
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
    """male/female из API; иначе ValueError."""
    if not gender:
        return None
    g = str(gender).strip().lower()
    if g in ("m", "male", "man"):
        return "male"
    if g in ("f", "female", "woman"):
        return "female"
    raise ValueError(f"voice_gender: ожидается male/female, получено {gender!r}")


def get_fish_api_key() -> str:
    """Ключ Fish TTS из config/.env (FISH_TTS_API_KEY или FISH_API_KEY)."""
    from config.env_config import get_fish_tts_api_key
    return get_fish_tts_api_key()


def _fish():
    """Ленивый клиент Fish Audio."""
    global _client
    if _client is None:
        from fishaudio import FishAudio
        _client = FishAudio(api_key=get_fish_api_key())
    return _client


# =============================================================================
# Настройка голоса на один прогон run()
# =============================================================================
def set_voice_prompts(*, cache_dir=None, gender=None, age=None):
    """Параметры голоса для текущего видео (gender/age override + cache_dir)."""
    _active["cache_root"] = Path(cache_dir) if cache_dir else None
    _active["gender"] = _norm_gender(gender)
    _active["age"] = float(age) if age is not None else None


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

    voices: [{"id", "path", "ref_text"}, ...] — эталоны cast_<id>.
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


def set_voice_clone_samples(samples: list[dict] | None) -> None:
    """Аудио-сэмплы для клонирования: пол + возрастные группы → voice_key.

    samples: [{"gender": "male"|"female", "path": str|Path,
               "age_groups": list[str]|None, "ref_text": str|None}, ...]
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


def uses_custom_clone() -> bool:
    """Есть ли clone/cast сэмплы для Fish voices.create."""
    return bool(_active["clone_by_key"] or _active["cast_by_id"])


def require_clone_sources() -> None:
    """Без clone/cast озвучка невозможна."""
    if not uses_custom_clone():
        raise ValueError(
            "Нужен голос для Fish TTS: voice_clone_samples / voice_sample_* "
            "или cast_voice / cast_mode=speakers"
        )


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


def _bank_root():
    return _active["cache_root"] if _active["cache_root"] else CACHE_ROOT


def _settings_meta_path():
    return _bank_root() / "voice_settings.json"


def _clone_samples_meta() -> dict:
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
    return {
        "clone_samples": _clone_samples_meta(),
        "cast_samples": _cast_samples_meta(),
        "cast_mode": _active["cast_mode"],
        "fish_model": FISH_MODEL,
    }


def _settings_changed():
    if not uses_custom_clone():
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
    if not uses_custom_clone():
        return
    _bank_root().mkdir(parents=True, exist_ok=True)
    _settings_meta_path().write_text(
        json.dumps(_current_settings_meta(), ensure_ascii=False, indent=2), encoding="utf-8",
    )


def normalize_voice_key(profile):
    """casting profile → male_mature / cast_loki и т.п."""
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


def _clone_spec(voice_key: str) -> dict | None:
    by_key = _active["clone_by_key"]
    if by_key and voice_key in by_key:
        return by_key[voice_key]
    if voice_key.startswith("cast_") and _active["cast_by_id"]:
        return _active["cast_by_id"].get(voice_key[len("cast_"):])
    return None


def _bank_voice_keys() -> list[str]:
    """Только ключи, для которых есть сэмпл (clone или cast)."""
    keys: list[str] = []
    for vk in (_active["clone_by_key"] or {}):
        if vk not in keys:
            keys.append(vk)
    for cid in (_active["cast_by_id"] or {}):
        vk = f"cast_{cid}"
        if vk not in keys:
            keys.append(vk)
    return keys


def _meta_path(vk: str) -> Path:
    return _bank_root() / f"{vk}.meta.json"


def _prepare_ref_wav(src: Path, dst: Path) -> None:
    """mp3/wav → mono WAV 44.1 kHz для Fish clone."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg", "-y", "-i", str(src),
        "-vn", "-acodec", "pcm_s16le", "-ar", "44100", "-ac", "1",
        str(dst),
    ]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        tail = (r.stderr or "")[-2000:]
        raise RuntimeError(f"ffmpeg не конвертировал {src.name}: {tail}")


def _create_fish_voice(vk: str, wav_path: Path, ref_text: str | None) -> str:
    """Загрузить сэмпл в Fish → вернуть reference_id."""
    client = _fish()
    audio = wav_path.read_bytes()
    kwargs = {
        "title": f"speechlab_{vk}",
        "voices": [audio],
        "visibility": "private",
        "train_mode": "fast",
        "enhance_audio_quality": True,
    }
    if ref_text:
        kwargs["texts"] = [ref_text]
    voice = client.voices.create(**kwargs)
    return voice.id


# =============================================================================
# Сборка voice_bank: сэмплы → Fish reference_id
# =============================================================================
def ensure_voice_bank(force=False):
    """Клонировать голоса в Fish и закэшировать reference_id."""
    global _voice_ids
    require_clone_sources()
    bank_keys = _bank_voice_keys()
    if not bank_keys:
        raise ValueError("Нет сэмплов для клонирования (clone/cast пусты)")

    root = _bank_root()
    root.mkdir(parents=True, exist_ok=True)

    if uses_custom_clone() and _settings_changed():
        force = True

    if force:
        _voice_ids.clear()

    print(f"  TTS: Fish clone ({FISH_MODEL}, {len(bank_keys)} voices)...")
    for vk in bank_keys:
        meta_p = _meta_path(vk)
        if not force and meta_p.is_file() and vk in _voice_ids:
            continue
        if not force and meta_p.is_file() and vk not in _voice_ids:
            try:
                saved = json.loads(meta_p.read_text(encoding="utf-8"))
                rid = (saved.get("reference_id") or "").strip()
                if rid:
                    _voice_ids[vk] = rid
                    print(f"    {vk} <- cache {rid[:8]}...")
                    continue
            except json.JSONDecodeError:
                pass

        spec = _clone_spec(vk)
        if not spec:
            raise FileNotFoundError(f"нет сэмпла для {vk}")
        src = Path(spec["path"])
        wav_path = root / f"{vk}.wav"
        _prepare_ref_wav(src, wav_path)
        ref_text = spec.get("ref_text")
        if ref_text:
            wav_path.with_suffix(".txt").write_text(ref_text, encoding="utf-8")
        rid = _create_fish_voice(vk, wav_path, ref_text)
        _voice_ids[vk] = rid
        meta = {
            "voice_key": vk,
            "reference_id": rid,
            "source": str(src.resolve()),
            "ref_text": ref_text,
            "fish_model": FISH_MODEL,
        }
        meta_p.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"    {vk} <- fish {rid[:8]}...")

    _save_settings_meta()


def _resolve_reference_id(vk: str) -> str:
    if vk in _voice_ids:
        return _voice_ids[vk]
    meta_p = _meta_path(vk)
    if meta_p.is_file():
        try:
            rid = (json.loads(meta_p.read_text(encoding="utf-8")).get("reference_id") or "").strip()
            if rid:
                _voice_ids[vk] = rid
                return rid
        except json.JSONDecodeError:
            pass
    raise ValueError(
        f"Нет клонированного голоса для {vk!r}. "
        "Передайте voice_sample_* / voice_clone_samples или cast_voice / cast_mode "
        "с покрытием этого профиля (пол×возраст или cast)."
    )


# =============================================================================
# Синтез одной реплики
# =============================================================================
def dub_from_profile(text, language, profile, out_path, *, bank_ready=False):
    """Озвучка по casting.json через Fish TTS (clone + emo-теги в тексте)."""
    _ = language  # язык уже в тексте перевода; клон по сэмплу
    if not (text or "").strip():
        raise ValueError("text пустой")
    profile = apply_voice_override(profile or {})
    if not bank_ready:
        ensure_voice_bank()

    vk = normalize_voice_key(profile)
    rid = _resolve_reference_id(vk)

    from fishaudio.types import TTSConfig

    # model=str: SDK typing ещё без s2.1-pro, API принимает заголовок
    audio = _fish().tts.convert(
        text=text.strip(),
        reference_id=rid,
        format="wav",
        config=TTSConfig(format="wav", sample_rate=44100, latency="normal"),
        model=FISH_MODEL,  # type: ignore[arg-type]
    )
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_bytes(audio)
    return str(out_path.resolve())


def unload_model():
    """Сброс кэша reference_id и клиента после этапа TTS."""
    global _client, _voice_ids
    _client = None
    _voice_ids = {}
