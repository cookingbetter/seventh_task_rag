"""Точка входа: загрузка базы знаний и нарезка документов на чанки."""

from __future__ import annotations

import sys
from pathlib import Path

from chunking import CHUNK_OVERLAP, CHUNK_SIZE, Chunk, chunk_document
from loader import Document, load_documents

KNOWLEDGE_BASE = Path(__file__).resolve().parent / "knowledge_base"
EXAMPLE_COUNT = 3


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

    chunked = [(document, chunk_document(document)) for document in documents]
    print(f"Загружено документов: {len(documents)}")
    print(f"Размер чанка: {CHUNK_SIZE} символов, перекрытие: {CHUNK_OVERLAP}")
    for document, chunks in chunked:
        print(
            f"- {document.source} | {document.char_count} символов | "
            f"{len(chunks)} чанков"
        )

    _print_examples(chunked)
    return 0


def _print_examples(chunked: list[tuple[Document, list[Chunk]]]) -> None:
    document, chunks = max(chunked, key=lambda item: (len(item[1]), item[0].char_count))
    examples = chunks[:EXAMPLE_COUNT]
    print()
    print(f"Примеры чанков ({document.source}):")
    for chunk in examples:
        print(f"chunk_id: {chunk.chunk_id}")
        print(f"document_id: {chunk.document_id}")
        print(f"source_name: {chunk.source_name}")
        print(chunk.text)
        print("---")


if __name__ == "__main__":
    raise SystemExit(main())
