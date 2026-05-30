"""Проверка логики хранения без GUI."""

import tempfile
from pathlib import Path

import storage


def run_tests():
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        storage.DATA_DIR = tmp_path / "data"
        storage.CONFIG_FILE = storage.DATA_DIR / "config.json"
        storage.DEFAULT_PEOPLE_FILE = storage.DATA_DIR / "people.json"

        p = storage.add_person("Иванов И.И.", "майор")
        assert p["fio"] == "Иванов И.И."
        assert p["rank"] == "майор"
        assert storage.people_count() == 1

        work = tmp_path / "work"
        work.mkdir()
        ok, _ = storage.set_work_dir(str(work))
        assert ok
        assert storage.get_work_dir() == work.resolve()

        storage.add_person("Петров П.П.", "капитан")
        assert storage.people_file() == work / "people.json"
        assert storage.people_count() == 1  # новый файл в work

        assert storage.delete_person(p["id"]) is False  # старый id в другом файле
        person = storage.load_people()[0]
        assert storage.delete_person(person["id"])
        assert storage.load_people() == []

        cfg = storage.load_config()
        cfg["work_dir"] = ""
        storage.save_config(cfg)
        assert storage.get_work_dir() is None

    print("test_storage: OK")


if __name__ == "__main__":
    run_tests()
