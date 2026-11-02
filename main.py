"""Точка входа дня 1: список документов базы знаний и их длина."""

from __future__ import annotations

import sys
from pathlib import Path

from loader import load_documents

KNOWLEDGE_BASE = Path(__file__).resolve().parent / "knowledge_base"


def main() -> int:
    if not KNOWLEDGE_BASE.is_dir():
        print(f"Папка базы знаний не найдена: {KNOWLEDGE_BASE}", file=sys.stderr)
        return 1

    try:
        documents = load_documents(KNOWLEDGE_BASE)
    except ValueError as exc:
        print(exc, file=sys.stderr)
        return 1

    if not documents:
        print("Документы не найдены.")
        print(
            "Положите минимум 5 файлов .txt или .md в кодировке UTF-8 "
            f"в папку {KNOWLEDGE_BASE}"
        )
        return 1

    print(f"Загружено документов: {len(documents)}")
    for document in documents:
        print(f"- {document.source} | {document.char_count} символов")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
