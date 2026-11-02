"""Embeddings чанков через OpenAI."""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

EMBEDDING_MODEL = "text-embedding-3-small"
ENV_PATH = Path(__file__).resolve().parent / ".env"


def embed_texts(texts: list[str]) -> list[list[float]]:
    """Возвращает вектор для каждого текста в том же порядке."""
    if not texts:
        return []

    load_dotenv(ENV_PATH)
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("Не задан OPENAI_API_KEY. Добавьте его в файл .env")

    response = OpenAI(api_key=api_key).embeddings.create(
        model=EMBEDDING_MODEL,
        input=texts,
    )
    ordered = sorted(response.data, key=lambda item: item.index)
    vectors = [item.embedding for item in ordered]
    if len(vectors) != len(texts):
        raise RuntimeError("OpenAI вернул неверное число embeddings")
    return vectors
