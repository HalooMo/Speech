"""Встроенные голоса для Qwen3-TTS Base clone (data/Cast).

Использование:
  cast_voice=loki|tom_hardy|thor  — один голос на все мужские слоты
  cast_mode=speakers              — раздать все 3 голоса по спикерам (по кругу)
"""
from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CAST_DIR = ROOT / "data" / "Cast"

# id → отображаемое имя, файл, эталонный текст для create_voice_clone_prompt
_VOICES: list[dict] = [
    {
        "id": "loki",
        "name": "Локи",
        "file": "AlexGav.wav",
        "gender": "male",
        "ref_text": (
            "Если поможете пройти лабиринт? Я проведу вас. "
            "Потому что лишь один ключ отпирает дверь в лабиринт."
        ),
    },
    {
        "id": "tom_hardy",
        "name": "Том Харди",
        "file": "IlyaIsaev.wav",
        "gender": "male",
        "ref_text": (
            "Как же можно сказать человеку, что нечего делать? "
            "Я представить не могу положение, чтобы когда-нибудь было нечего делать."
        ),
    },
    {
        "id": "thor",
        "name": "Тор",
        "file": "IvanZhark.wav",
        "gender": "male",
        "ref_text": (
            "А так можно срезать дорогу к пруду. "
            "Или они развалили эту халабуду до конца? Вроде да. "
            "Старичок перестал читать мораль."
        ),
    },
]


def list_cast_voices() -> list[dict]:
    """Публичный список пресетов (для API и документации)."""
    out = []
    for v in _VOICES:
        path = CAST_DIR / v["file"]
        out.append({
            "id": v["id"],
            "name": v["name"],
            "gender": v["gender"],
            "sample_file": v["file"],
            "sample_path": str(path.resolve()),
            "ref_text": v["ref_text"],
            "available": path.is_file(),
        })
    return out


def _norm_query(q: str) -> str:
    return " ".join((q or "").strip().lower().replace("ё", "е").split())


def resolve_cast_voice(query: str) -> dict:
    """Найти пресет по id или имени (Локи / loki / Том Харди)."""
    key = _norm_query(query)
    if not key:
        raise ValueError("cast_voice: пустое значение")
    for v in _VOICES:
        aliases = {_norm_query(v["id"]), _norm_query(v["name"]), _norm_query(v["id"].replace("_", " "))}
        if key in aliases:
            path = CAST_DIR / v["file"]
            if not path.is_file():
                raise FileNotFoundError(f"cast_voice {v['id']}: нет файла {path}")
            return {
                "id": v["id"],
                "name": v["name"],
                "gender": v["gender"],
                "path": path.resolve(),
                "ref_text": v["ref_text"],
            }
    known = ", ".join(f"{v['id']} ({v['name']})" for v in _VOICES)
    raise ValueError(f"cast_voice: неизвестно {query!r}. Доступно: {known}")


def cast_voice_ids() -> list[str]:
    return [v["id"] for v in _VOICES]


def voice_key_for_cast(cast_id: str) -> str:
    """Ключ эталона в voice_bank: cast_loki, cast_tom_hardy, …"""
    return f"cast_{cast_id}"


def to_clone_sample(voice: dict, age_groups: list[str] | None = None) -> dict:
    """Формат для dubbing.set_voice_clone_samples (один пол × age_groups)."""
    return {
        "gender": voice["gender"],
        "path": voice["path"],
        "ref_text": voice["ref_text"],
        "age_groups": age_groups,  # None → все 4 группы
    }


def assign_cast_to_speakers(speakers: list[str]) -> dict[str, str]:
    """Спикер → cast_id по кругу (порядок каталога)."""
    ids = cast_voice_ids()
    if not ids:
        return {}
    ordered = sorted(speakers, key=lambda s: str(s))
    return {spk: ids[i % len(ids)] for i, spk in enumerate(ordered)}
