"""Точка входа: просмотр базы знаний и загрузка чанков в Weaviate."""

from __future__ import annotations

import sys
from pathlib import Path

from chunking import CHUNK_OVERLAP, CHUNK_SIZE, Chunk, chunk_document, chunk_documents
from loader import Document, load_documents

KNOWLEDGE_BASE = Path(__file__).resolve().parent / "knowledge_base"
EXAMPLE_COUNT = 3
SAMPLE_TEXT_LIMIT = 180


def main(argv: list[str] | None = None) -> int:
    args = sys.argv[1:] if argv is None else argv
    if not args:
        return show_knowledge_base()
    if args == ["index"]:
        return index_knowledge_base()
    if args == ["demo"]:
        return demo_questions()
    if args == ["compare"]:
        return compare_answers()
    if args == ["evaluate"]:
        return evaluate_questions()
    if args[0] in {"ask", "answer"}:
        query = " ".join(args[1:]).strip()
        if not query:
            print(f'Укажите вопрос: python3 main.py {args[0]} "вопрос"', file=sys.stderr)
            return 1
        if args[0] == "ask":
            return ask_question(query)
        return answer_question(query)
    print(
        "Неизвестная команда. Используйте: "
        'python3 main.py [index|demo|compare|evaluate|ask "вопрос"|answer "вопрос"]',
        file=sys.stderr,
    )
    return 1


def show_knowledge_base() -> int:
    documents, error_code = _load_documents()
    if error_code != 0 or documents is None:
        return error_code

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


def index_knowledge_base() -> int:
    documents, error_code = _load_documents()
    if error_code != 0 or documents is None:
        return error_code

    chunks = chunk_documents(documents)
    if not chunks:
        print("Нет чанков для загрузки.", file=sys.stderr)
        return 1

    try:
        from embedder import embed_texts
        from weaviate_store import upload_chunks

        vectors = embed_texts([chunk.text for chunk in chunks])
        summary = upload_chunks(chunks, vectors)
    except Exception as exc:
        print(f"Не удалось загрузить базу в Weaviate: {exc}", file=sys.stderr)
        return 1

    sample = summary.sample
    preview = sample.text.replace("\n", " ")
    if len(preview) > SAMPLE_TEXT_LIMIT:
        preview = preview[:SAMPLE_TEXT_LIMIT] + "..."
    print(f"Объектов в коллекции: {summary.count}")
    print(f"chunk_id: {sample.chunk_id}")
    print(f"document_id: {sample.document_id}")
    print(f"source_name: {sample.source_name}")
    print(f"text: {preview}")
    return 0


WEAK_QUESTIONS = [
    "Какая столица Франции?",
    "Как приготовить борщ?",
    "Какой курс доллара сегодня?",
    "Как получить визу в Японию?",
    "Сколько планет в Солнечной системе?",
]


DEMO_QUESTIONS = [
    "В какое время нельзя проводить шумные ремонтные работы в Москве?",
    "Как зарегистрироваться в личном кабинете ПИК-Комфорт?",
    "Как подключить автоплатёж?",
    "Нужно ли передавать показания счётчиков, если не живу в квартире?",
    "Чем опасна незаконная перепланировка?",
]


def evaluate_questions() -> int:
    print("Вопросы, на которые база отвечает")
    for query in DEMO_QUESTIONS:
        if _evaluate_one(query) != 0:
            return 1
    print()
    print("Вопросы вне базы")
    for query in WEAK_QUESTIONS:
        if _evaluate_one(query) != 0:
            return 1
    return 0


def _evaluate_one(query: str) -> int:
    try:
        from rag_pipeline import answer
        from retriever import search

        hits = search(query)
        result = answer(query)
    except Exception as exc:
        print(f"Не удалось оценить вопрос: {exc}", file=sys.stderr)
        return 1
    best = hits[0] if hits else None
    distance = f"{best.distance:.4f}" if best and best.distance is not None else "—"
    source = best.source_name if best else "—"
    decision = "ответ" if result.context_sufficient else "отказ"
    print(f"Вопрос: {query}")
    print(f"distance: {distance}")
    print(f"source_name: {source}")
    print(f"решение: {decision}")
    print(result.answer)
    print()
    return 0


def answer_question(query: str) -> int:
    try:
        from rag_pipeline import answer

        result = answer(query)
    except Exception as exc:
        print(f"Не удалось собрать ответ: {exc}", file=sys.stderr)
        return 1
    _print_rag_answer(result)
    return 0


def compare_answers() -> int:
    try:
        from rag_pipeline import answer, answer_without_retrieval
    except Exception as exc:
        print(f"Не удалось собрать ответ: {exc}", file=sys.stderr)
        return 1

    for query in DEMO_QUESTIONS:
        print(f"Вопрос: {query}")
        try:
            plain = answer_without_retrieval(query)
            grounded = answer(query)
        except Exception as exc:
            print(f"Не удалось собрать ответ: {exc}", file=sys.stderr)
            return 1
        print("Без retrieval:")
        print(plain)
        print("С retrieval:")
        _print_rag_answer(grounded)
        print()
    return 0


def _print_rag_answer(result) -> None:
    print(result.answer)
    sufficient = "да" if result.context_sufficient else "нет"
    print(f"Контекст достаточен: {sufficient}")
    print("Источники:")
    if not result.sources:
        print("- нет")
        return
    for source in result.sources:
        print(f"- {source.chunk_id} | {source.source_name}")


def ask_question(query: str) -> int:
    try:
        from retriever import search

        hits = search(query)
    except Exception as exc:
        print(f"Не удалось выполнить поиск: {exc}", file=sys.stderr)
        return 1
    _print_hits("Semantic search", hits)
    return 0


def demo_questions() -> int:
    try:
        from retriever import search, search_hybrid
    except Exception as exc:
        print(f"Не удалось выполнить поиск: {exc}", file=sys.stderr)
        return 1

    for query in DEMO_QUESTIONS:
        print(f"Вопрос: {query}")
        try:
            semantic = search(query)
            hybrid = search_hybrid(query)
        except Exception as exc:
            print(f"Не удалось выполнить поиск: {exc}", file=sys.stderr)
            return 1
        _print_hits("Semantic search", semantic)
        _print_hits("Hybrid search", hybrid)
        print()
    return 0


def _print_hits(title: str, hits: list) -> None:
    print(title)
    if not hits:
        print("Ничего не найдено.")
        return
    for index, hit in enumerate(hits, start=1):
        if hit.distance is not None:
            metric = f"distance: {hit.distance:.4f}"
        elif hit.score is not None:
            metric = f"score: {hit.score:.4f}"
        else:
            metric = "score: —"
        preview = hit.text.replace("\n", " ")
        if len(preview) > SAMPLE_TEXT_LIMIT:
            preview = preview[:SAMPLE_TEXT_LIMIT] + "..."
        print(f"{index}. {metric}")
        print(f"source_name: {hit.source_name}")
        print(f"chunk_id: {hit.chunk_id}")
        print(preview)


def _load_documents() -> tuple[list[Document] | None, int]:
    if not KNOWLEDGE_BASE.is_dir():
        print(f"Папка базы знаний не найдена: {KNOWLEDGE_BASE}", file=sys.stderr)
        return None, 1

    try:
        documents = load_documents(KNOWLEDGE_BASE)
    except ValueError as exc:
        print(exc, file=sys.stderr)
        return None, 1

    if not documents:
        print("Документы не найдены.")
        print(
            "Положите минимум 5 файлов .txt или .md в кодировке UTF-8 "
            f"в папку {KNOWLEDGE_BASE}"
        )
        return None, 1
    return documents, 0


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
