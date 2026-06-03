"""Проверка логики хранения без GUI."""

import tempfile
from pathlib import Path

import storage


def run_tests():
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        storage.DATA_DIR = tmp_path / "cfg"
        storage.CONFIG_FILE = storage.DATA_DIR / "config.json"
        work = tmp_path / "work"
        work.mkdir()
        storage.set_work_dir(str(work))

        p = storage.add_person("Иванов И.И.", "майор")
        assert p["status"] == ""
        storage.set_person_status(p["id"], "отпуск")
        assert storage.find_person(p["id"])["status"] == "отпуск"
        c = storage.status_counts()
        assert c["отпуск"] == 1
        assert c["работают"] == 0
        storage.return_to_work(p["id"])
        assert storage.find_person(p["id"])["status"] == ""
        assert storage.delete_person(p["id"])
        assert storage.load_people() == []

    if storage.LS_DOCX_FILE.is_file():
        rows = storage.parse_ls_docx()
        assert len(rows) >= 40

    print("test_storage: OK")


if __name__ == "__main__":
    run_tests()
