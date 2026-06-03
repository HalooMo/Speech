"""Создаёт Word-документ с заголовком «Как ты»."""

from pathlib import Path

from docx import Document

OUT = Path(__file__).resolve().parent / "Как_ты.docx"


def main():
    doc = Document()
    doc.add_heading("Как ты", level=0)
    doc.save(OUT)
    print(f"Создан: {OUT}")


if __name__ == "__main__":
    main()
