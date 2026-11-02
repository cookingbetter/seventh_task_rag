"""Схема Weaviate и загрузка чанков с векторами."""

from __future__ import annotations

import uuid
from dataclasses import dataclass

import weaviate
from weaviate.classes.config import Configure, DataType, Property
from weaviate.classes.data import DataObject

from chunking import Chunk

COLLECTION_NAME = "KnowledgeChunk"
_CHUNK_NAMESPACE = uuid.UUID("a3f1c2d4-5e6b-4789-8abc-def012345678")


@dataclass(frozen=True)
class StoredChunk:
    """Объект, прочитанный обратно из Weaviate."""

    chunk_id: str
    document_id: str
    source_name: str
    text: str


@dataclass(frozen=True)
class IndexSummary:
    """Результат загрузки базы знаний."""

    count: int
    sample: StoredChunk


def chunk_uuid(chunk_id: str) -> uuid.UUID:
    """Стабильный идентификатор объекта по chunk_id."""
    return uuid.uuid5(_CHUNK_NAMESPACE, chunk_id)


def upload_chunks(chunks: list[Chunk], vectors: list[list[float]]) -> IndexSummary:
    """Пересоздаёт коллекцию и загружает текущие чанки без дублей."""
    if len(chunks) != len(vectors):
        raise ValueError("Число чанков и embeddings не совпадает")
    if not chunks:
        raise ValueError("Нет чанков для загрузки")

    try:
        client = weaviate.connect_to_local()
    except Exception as exc:
        raise ConnectionError(
            "Weaviate недоступен. Запустите его командой docker compose up -d"
        ) from exc
    try:
        if not client.is_ready():
            raise ConnectionError(
                "Weaviate недоступен. Запустите его командой docker compose up -d"
            )
        _recreate_collection(client)
        collection = client.collections.get(COLLECTION_NAME)
        result = collection.data.insert_many(_data_objects(chunks, vectors))
        if result.has_errors:
            first_error = next(iter(result.errors.values()))
            raise RuntimeError(f"Weaviate не принял объекты: {first_error.message}")

        total = collection.aggregate.over_all(total_count=True).total_count
        if total != len(chunks):
            raise RuntimeError(
                f"В коллекции {total} объектов, ожидалось {len(chunks)}"
            )
        sample = _read_sample(collection, chunks[0].chunk_id)
        return IndexSummary(count=total, sample=sample)
    finally:
        client.close()


def _recreate_collection(client: weaviate.WeaviateClient) -> None:
    if client.collections.exists(COLLECTION_NAME):
        client.collections.delete(COLLECTION_NAME)
    client.collections.create(
        name=COLLECTION_NAME,
        vector_config=Configure.Vectors.self_provided(),
        properties=[
            Property(name="document_id", data_type=DataType.TEXT),
            Property(name="source_name", data_type=DataType.TEXT),
            Property(name="chunk_id", data_type=DataType.TEXT),
            Property(name="text", data_type=DataType.TEXT),
        ],
    )


def _data_objects(chunks: list[Chunk], vectors: list[list[float]]) -> list[DataObject]:
    objects: list[DataObject] = []
    for chunk, vector in zip(chunks, vectors):
        objects.append(
            DataObject(
                properties={
                    "document_id": chunk.document_id,
                    "source_name": chunk.source_name,
                    "chunk_id": chunk.chunk_id,
                    "text": chunk.text,
                },
                uuid=chunk_uuid(chunk.chunk_id),
                vector=vector,
            )
        )
    return objects


def _read_sample(collection, chunk_id: str) -> StoredChunk:
    fetched = collection.query.fetch_object_by_id(chunk_uuid(chunk_id))
    if fetched is None:
        raise RuntimeError(f"Объект {chunk_id} не найден после загрузки")
    properties = fetched.properties
    return StoredChunk(
        chunk_id=str(properties["chunk_id"]),
        document_id=str(properties["document_id"]),
        source_name=str(properties["source_name"]),
        text=str(properties["text"]),
    )
