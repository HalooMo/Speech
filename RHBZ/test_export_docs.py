"""Проверка генерации документов (без Word COM для ШДС — опционально)."""

import tempfile
from pathlib import Path

import storage
import export_docs


def run_tests():
    with tempfile.TemporaryDirectory() as tmp:
        work = Path(tmp) / "work"
        work.mkdir()
        storage.DATA_DIR = Path(tmp) / "cfg"
        storage.CONFIG_FILE = storage.DATA_DIR / "config.json"
        storage.set_work_dir(str(work))

        storage.add_person("Иванов Иван Иванович", "майор")
        storage.add_person("Петров Пётр Петрович", "капитан")
        storage.set_person_status(
            storage.load_people()[0]["id"], "отпуск"
        )

        groups = export_docs.group_by_status(storage.load_people())
        assert len(groups["отпуск"]) == 1
        assert len(groups[""]) == 1

        if not export_docs.TEMPLATE_STROEVAYA_DOCX.is_file():
            print("test_export_docs: skip (no Pattern templates)")
            return

        result = export_docs.generate_all()
        assert result["ok"], result
        folder = Path(result["folder"])
        assert folder.is_dir()
        assert len(list(folder.glob("*"))) == 4
        print("test_export_docs: OK", folder)


if __name__ == "__main__":
    run_tests()
