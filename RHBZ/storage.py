"""Хранение списка людей и расхода по причинам. Все данные — только в рабочей директории."""

import json
import uuid
from pathlib import Path

APP_DIR = Path(__file__).resolve().parent
DATA_DIR = APP_DIR / "data"
CONFIG_FILE = DATA_DIR / "config.json"
LS_DOCX_FILE = APP_DIR / "Pattern" / "ЛС  1125 ЦРХБЗ!!!.docx"
PEOPLE_FILENAME = "people.json"

# Причины расхода (отсутствия)
REASONS = [
    "наряд",
    "отпуск",
    "командировка",
    "40 РЦПС",
    "больничный",
    "госпиталь",
    "прочие причины",
]


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
    """Рабочая директория: программа читает и пишет только здесь."""
    path = load_config().get("work_dir", "").strip()
    if not path:
        return None
    p = Path(path)
    return p.resolve() if p.is_dir() else None


def set_work_dir(path: str) -> tuple[bool, str]:
    p = Path(path.strip())
    if not p.exists():
        return False, "Папка не существует."
    if not p.is_dir():
        return False, "Укажите папку, а не файл."
    p = p.resolve()
    cfg = load_config()
    cfg["work_dir"] = str(p)
    save_config(cfg)
    p.mkdir(parents=True, exist_ok=True)
    _ensure_people_file()
    return True, str(p)


def clear_work_dir() -> None:
    cfg = load_config()
    cfg["work_dir"] = ""
    save_config(cfg)


def require_work_dir() -> tuple[bool, str]:
    if get_work_dir():
        return True, ""
    return False, "Сначала укажите рабочую директорию."


def people_file() -> Path:
    work = get_work_dir()
    if not work:
        raise RuntimeError("Рабочая директория не задана.")
    return work / PEOPLE_FILENAME


def _ensure_people_file() -> None:
    pf = people_file()
    pf.parent.mkdir(parents=True, exist_ok=True)
    if not pf.exists():
        pf.write_text("[]", encoding="utf-8")


def _normalize(person: dict) -> dict:
    status = str(person.get("status", "")).strip()
    if status not in REASONS:
        status = "" if status else status  # пусто = в общем списке (работает)
    return {
        "id": person["id"],
        "fio": str(person.get("fio", "")).strip(),
        "rank": str(
            person.get("rank", person.get("zvanie", person.get("unit", "")))
        ).strip(),
        "status": status if status in REASONS else "",
    }


def load_people() -> list[dict]:
    ok, msg = require_work_dir()
    if not ok:
        return []
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
        "status": "",
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


def set_person_status(person_id: str, status: str) -> tuple[bool, str]:
    if status and status not in REASONS:
        return False, "Неизвестная причина."
    people = load_people()
    for p in people:
        if p["id"] != person_id:
            continue
        p["status"] = status
        save_people(people)
        return True, ""
    return False, "Запись не найдена."


def return_to_work(person_id: str) -> tuple[bool, str]:
    return set_person_status(person_id, "")


def status_counts(people: list[dict] | None = None) -> dict[str, int]:
    rows = people if people is not None else load_people()
    counts = {r: 0 for r in REASONS}
    working = 0
    for p in rows:
        st = p.get("status", "")
        if st in counts:
            counts[st] += 1
        else:
            working += 1
    counts["работают"] = working
    counts["всего"] = len(rows)
    return counts


def people_count() -> int:
    return len(load_people())


def parse_ls_docx(path: Path | None = None) -> list[dict]:
    from docx import Document

    docx_path = Path(path) if path else LS_DOCX_FILE
    if not docx_path.is_file():
        raise FileNotFoundError(f"Файл не найден: {docx_path}")

    table = Document(docx_path).tables[0]
    rows = []
    for i, row in enumerate(table.rows):
        if i == 0:
            continue
        rank = row.cells[1].text.strip()
        fio = row.cells[2].text.strip()
        if not fio:
            continue
        rows.append({"rank": rank, "fio": fio})
    return rows


def _person_key(fio: str, rank: str) -> tuple[str, str]:
    return fio.strip().lower(), rank.strip().lower()


def import_from_ls_docx(path: Path | None = None, replace: bool = False) -> dict:
    ok, msg = require_work_dir()
    if not ok:
        return {"error": msg}

    parsed = parse_ls_docx(path)
    new_people = [
        {"id": str(uuid.uuid4()), "fio": p["fio"], "rank": p["rank"], "status": ""}
        for p in parsed
    ]

    if replace:
        save_people(new_people)
        added = len(new_people)
    else:
        existing = load_people()
        keys = {_person_key(p["fio"], p["rank"]) for p in existing}
        added = 0
        for p in new_people:
            key = _person_key(p["fio"], p["rank"])
            if key in keys:
                continue
            existing.append(p)
            keys.add(key)
            added += 1
        save_people(existing)

    return {
        "added": added,
        "from_docx": len(parsed),
        "total": people_count(),
        "people_file": str(people_file()),
    }
