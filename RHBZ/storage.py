"""Хранение списка людей (ФИО, звание) и настроек рабочей директории."""

import json
import uuid
from pathlib import Path

# Служебные файлы приложения (рядом с кодом)
APP_DIR = Path(__file__).resolve().parent
DATA_DIR = APP_DIR / "data"
CONFIG_FILE = DATA_DIR / "config.json"
DEFAULT_PEOPLE_FILE = DATA_DIR / "people.json"


def _ensure_data_dir() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)


def load_config() -> dict:
    _ensure_data_dir()
    if not CONFIG_FILE.exists():
        return {"work_dir": ""}
    with CONFIG_FILE.open(encoding="utf-8") as f:
        return json.load(f)


def save_config(config: dict) -> None:
    _ensure_data_dir()
    with CONFIG_FILE.open("w", encoding="utf-8") as f:
        json.dump(config, f, ensure_ascii=False, indent=2)


def get_work_dir() -> Path | None:
    path = load_config().get("work_dir", "").strip()
    if not path:
        return None
    p = Path(path)
    return p if p.is_dir() else None


def set_work_dir(path: str) -> tuple[bool, str]:
    p = Path(path.strip())
    if not p.exists():
        return False, "Папка не существует."
    if not p.is_dir():
        return False, "Укажите папку, а не файл."
    cfg = load_config()
    cfg["work_dir"] = str(p.resolve())
    save_config(cfg)
    return True, str(p.resolve())


def people_file() -> Path:
    """Файл списка: в рабочей директории или в data/ по умолчанию."""
    work = get_work_dir()
    if work:
        work.mkdir(parents=True, exist_ok=True)
        return work / "people.json"
    _ensure_data_dir()
    return DEFAULT_PEOPLE_FILE


def _ensure_people_file() -> None:
    pf = people_file()
    pf.parent.mkdir(parents=True, exist_ok=True)
    if not pf.exists():
        pf.write_text("[]", encoding="utf-8")


def _normalize(person: dict) -> dict:
    """Поддержка старых записей (unit/note -> rank)."""
    return {
        "id": person["id"],
        "fio": str(person.get("fio", "")).strip(),
        "rank": str(
            person.get("rank", person.get("zvanie", person.get("unit", "")))
        ).strip(),
    }


def load_people() -> list[dict]:
    _ensure_people_file()
    with people_file().open(encoding="utf-8") as f:
        raw = json.load(f)
    return [_normalize(p) for p in raw]


def save_people(people: list[dict]) -> None:
    _ensure_people_file()
    with people_file().open("w", encoding="utf-8") as f:
        json.dump(people, f, ensure_ascii=False, indent=2)


def add_person(fio: str, rank: str) -> dict:
    person = {
        "id": str(uuid.uuid4()),
        "fio": fio.strip(),
        "rank": rank.strip(),
    }
    people = load_people()
    people.append(person)
    save_people(people)
    return person


def find_person(person_id: str) -> dict | None:
    for p in load_people():
        if p["id"] == person_id:
            return p
    return None


def delete_person(person_id: str) -> bool:
    people = load_people()
    new_list = [p for p in people if p["id"] != person_id]
    if len(new_list) == len(people):
        return False
    save_people(new_list)
    return True


def people_count() -> int:
    return len(load_people())
