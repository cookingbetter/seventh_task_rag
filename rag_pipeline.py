"""Сборка ответа: retrieval, промпт и вызов модели."""

from __future__ import annotations

import os

from dotenv import load_dotenv
from openai import OpenAI

from embedder import ENV_PATH
from prompts import GROUNDED_SYSTEM, UNGROUNDED_SYSTEM
from retriever import RetrievedChunk, search
from schemas import PlainAnswer, RagAnswer, SourceRef

ANSWER_MODEL = "gpt-4.1-mini"


def answer(query: str) -> RagAnswer:
    """Отвечает на вопрос только по top-3 чанкам из Weaviate."""
    text = _require_query(query)
    chunks = search(text)
    if not chunks:
        return RagAnswer(
            answer="По найденным фрагментам ответить нельзя: поиск ничего не вернул.",
            context_sufficient=False,
            sources=[],
        )
    parsed = _complete(
        GROUNDED_SYSTEM,
        _grounded_user_message(text, chunks),
        RagAnswer,
    )
    return parsed.model_copy(update={"sources": _known_sources(parsed.sources, chunks)})


def answer_without_retrieval(query: str) -> str:
    """Отвечает на тот же вопрос без фрагментов базы."""
    text = _require_query(query)
    parsed = _complete(UNGROUNDED_SYSTEM, text, PlainAnswer)
    return parsed.answer


def _require_query(query: str) -> str:
    text = query.strip()
    if not text:
        raise ValueError("Пустой вопрос")
    return text


def _grounded_user_message(query: str, chunks: list[RetrievedChunk]) -> str:
    blocks = []
    for chunk in chunks:
        blocks.append(f"[{chunk.chunk_id}] {chunk.source_name}\n{chunk.text}")
    context = "\n\n".join(blocks)
    return f"Вопрос:\n{query}\n\nКонтекст:\n{context}"


def _known_sources(sources: list[SourceRef], chunks: list[RetrievedChunk]) -> list[SourceRef]:
    by_id = {chunk.chunk_id: chunk for chunk in chunks}
    kept: list[SourceRef] = []
    seen: set[str] = set()
    for source in sources:
        chunk = by_id.get(source.chunk_id)
        if chunk is None or source.chunk_id in seen:
            continue
        seen.add(source.chunk_id)
        kept.append(SourceRef(chunk_id=chunk.chunk_id, source_name=chunk.source_name))
    return kept


def _complete(system: str, user: str, schema: type[RagAnswer] | type[PlainAnswer]):
    load_dotenv(ENV_PATH)
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("Не задан OPENAI_API_KEY. Добавьте его в файл .env")
    completion = OpenAI(api_key=api_key).chat.completions.parse(
        model=ANSWER_MODEL,
        temperature=0,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        response_format=schema,
    )
    message = completion.choices[0].message
    if message.parsed is None:
        refusal = message.refusal or "Модель не вернула ответ"
        raise RuntimeError(refusal)
    return message.parsed
