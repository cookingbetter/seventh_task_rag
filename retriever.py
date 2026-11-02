"""Поиск релевантных чанков в Weaviate."""

from __future__ import annotations

from dataclasses import dataclass

import weaviate
from weaviate.classes.query import MetadataQuery

from embedder import embed_texts
from weaviate_store import COLLECTION_NAME

TOP_K = 3
MAX_DISTANCE = 0.55


@dataclass(frozen=True)
class RetrievedChunk:
    """Чанк, найденный по вопросу."""

    chunk_id: str
    source_name: str
    document_id: str
    text: str
    distance: float | None = None
    score: float | None = None


def is_weak_context(chunks: list[RetrievedChunk]) -> bool:
    """Контекст слабый, если ближайший чанк дальше порога или его нет."""
    distances = [chunk.distance for chunk in chunks if chunk.distance is not None]
    if not distances:
        return True
    return min(distances) > MAX_DISTANCE


def search(query: str, limit: int = TOP_K) -> list[RetrievedChunk]:
    """Semantic search: ближе к нулю distance — релевантнее чанк."""
    vector = _query_vector(query)
    client = _connect()
    try:
        response = client.collections.get(COLLECTION_NAME).query.near_vector(
            near_vector=vector,
            limit=limit,
            return_metadata=MetadataQuery(distance=True),
        )
        return [_from_object(obj, distance=obj.metadata.distance) for obj in response.objects]
    finally:
        client.close()


def search_hybrid(query: str, limit: int = TOP_K) -> list[RetrievedChunk]:
    """Hybrid search: выше score — релевантнее чанк."""
    vector = _query_vector(query)
    client = _connect()
    try:
        response = client.collections.get(COLLECTION_NAME).query.hybrid(
            query=query,
            vector=vector,
            alpha=0.5,
            limit=limit,
            query_properties=["text"],
            return_metadata=MetadataQuery(score=True),
        )
        return [_from_object(obj, score=obj.metadata.score) for obj in response.objects]
    finally:
        client.close()


def _query_vector(query: str) -> list[float]:
    text = query.strip()
    if not text:
        raise ValueError("Пустой вопрос")
    vectors = embed_texts([text])
    if len(vectors) != 1:
        raise RuntimeError("OpenAI не вернул embedding вопроса")
    return vectors[0]


def _connect() -> weaviate.WeaviateClient:
    try:
        client = weaviate.connect_to_local()
    except Exception as exc:
        raise ConnectionError(
            "Weaviate недоступен. Запустите его командой docker compose up -d"
        ) from exc
    if not client.is_ready():
        client.close()
        raise ConnectionError(
            "Weaviate недоступен. Запустите его командой docker compose up -d"
        )
    return client


def _from_object(obj, distance: float | None = None, score: float | None = None) -> RetrievedChunk:
    properties = obj.properties
    return RetrievedChunk(
        chunk_id=str(properties["chunk_id"]),
        source_name=str(properties["source_name"]),
        document_id=str(properties["document_id"]),
        text=str(properties["text"]),
        distance=distance,
        score=score,
    )
