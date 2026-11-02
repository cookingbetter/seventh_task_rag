"""Схемы ответа RAG."""

from pydantic import BaseModel, Field


class SourceRef(BaseModel):
    """Источник, на который опирается ответ."""

    chunk_id: str
    source_name: str


class RagAnswer(BaseModel):
    """Ответ, построенный по найденным фрагментам."""

    answer: str
    context_sufficient: bool
    sources: list[SourceRef] = Field(default_factory=list)


class PlainAnswer(BaseModel):
    """Ответ модели без retrieved context."""

    answer: str
