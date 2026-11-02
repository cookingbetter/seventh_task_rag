"""Нарезка очищенных документов на чанки для retrieval."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from loader import Document

CHUNK_SIZE = 500
CHUNK_OVERLAP = 80
_BREAK_CHARS = " \n"


@dataclass(frozen=True)
class Chunk:
    """Фрагмент документа с метаданными для поиска."""

    document_id: str
    source_name: str
    chunk_id: str
    text: str


def split_text(
    text: str,
    chunk_size: int = CHUNK_SIZE,
    overlap: int = CHUNK_OVERLAP,
) -> list[str]:
    """Режет текст на фрагменты, не разрывая слова."""
    if overlap >= chunk_size:
        raise ValueError("CHUNK_OVERLAP должен быть меньше CHUNK_SIZE")

    stripped = text.strip()
    if not stripped:
        return []
    if len(stripped) <= chunk_size:
        return [stripped]

    chunks: list[str] = []
    start = 0
    length = len(stripped)
    while start < length:
        hard_end = min(start + chunk_size, length)
        end = _break_before(stripped, start, hard_end)
        piece = stripped[start:end].strip()
        if piece:
            chunks.append(piece)
        if end >= length:
            break
        start = _next_start(stripped, start, end, overlap)
        if start >= length:
            break
    return chunks


def chunk_document(document: Document) -> list[Chunk]:
    """Нарезает один документ и проставляет метаданные чанков."""
    document_id = Path(document.source).stem
    return [
        Chunk(
            document_id=document_id,
            source_name=document.source,
            chunk_id=f"{document_id}-{index}",
            text=part,
        )
        for index, part in enumerate(split_text(document.text))
    ]


def chunk_documents(documents: list[Document]) -> list[Chunk]:
    """Нарезает все документы в исходном порядке."""
    chunks: list[Chunk] = []
    for document in documents:
        chunks.extend(chunk_document(document))
    return chunks


def _break_before(text: str, start: int, hard_end: int) -> int:
    """Сдвигает конец чанка к переводу строки или пробелу во второй половине окна."""
    if hard_end >= len(text):
        return len(text)
    window_start = start + (hard_end - start) // 2
    for index in range(hard_end - 1, window_start - 1, -1):
        if text[index] in _BREAK_CHARS:
            return index
    return hard_end


def _next_start(text: str, start: int, end: int, overlap: int) -> int:
    """Начинает следующий чанк так, чтобы хвост предыдущего попал в перекрытие."""
    next_start = end - overlap
    if next_start <= start:
        next_start = end
    elif next_start > 0 and text[next_start - 1] not in _BREAK_CHARS and text[next_start] not in _BREAK_CHARS:
        snapped = next_start
        while snapped < end and text[snapped] not in _BREAK_CHARS:
            snapped += 1
        while snapped < end and text[snapped] in _BREAK_CHARS:
            snapped += 1
        if snapped < end:
            next_start = snapped
    while next_start < len(text) and text[next_start] in _BREAK_CHARS:
        next_start += 1
    if next_start <= start:
        next_start = end
        while next_start < len(text) and text[next_start] in _BREAK_CHARS:
            next_start += 1
    return next_start
