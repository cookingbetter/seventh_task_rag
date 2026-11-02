"""Загрузка и очистка текстовых документов базы знаний."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

SUPPORTED_SUFFIXES = {".txt", ".md"}
_REPEATED_SPACES = re.compile(r"[ \t]+")


@dataclass(frozen=True)
class Document:
    """Один документ после очистки."""

    source: str
    path: Path
    text: str
    char_count: int


def clean_text(text: str) -> str:
    """Нормализует переводы строк, пробелы и пустые строки."""
    normalized = text.replace("\r\n", "\n").replace("\r", "\n")
    lines: list[str] = []
    for line in normalized.split("\n"):
        collapsed = _REPEATED_SPACES.sub(" ", line.strip())
        if collapsed:
            lines.append(collapsed)
    return "\n".join(lines)


def load_documents(directory: Path) -> list[Document]:
    """Читает .txt и .md только из указанной папки, без вложенных каталогов."""
    if not directory.is_dir():
        raise FileNotFoundError(f"Папка базы знаний не найдена: {directory}")

    documents: list[Document] = []
    for path in sorted(directory.iterdir()):
        if not path.is_file():
            continue
        if path.name.startswith("."):
            continue
        if path.suffix.lower() not in SUPPORTED_SUFFIXES:
            continue
        try:
            raw = path.read_text(encoding="utf-8")
        except UnicodeDecodeError as exc:
            raise ValueError(
                f"Файл {path.name} должен быть в кодировке UTF-8"
            ) from exc
        text = clean_text(raw)
        documents.append(
            Document(
                source=path.name,
                path=path,
                text=text,
                char_count=len(text),
            )
        )
    return documents
