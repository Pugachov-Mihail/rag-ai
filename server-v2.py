import json
import os
import threading
import time
from dataclasses import dataclass
from typing import Dict, List, Optional

import chromadb
import requests
from flask import Flask, Response, request
from qdrant_client import QdrantClient

from reader_book import async_knowledge_crawler, rebuild_vector_db

app = Flask(__name__)

CHROMA_PATH = "./my_library_db"
EMBED_MODEL = "nomic-embed-text"
PLANNER_MODEL = "qwen2.5-coder:14b"
ANSWER_MODEL = "qwen2.5-coder:14b"
REVIEW_MODEL = "gemma4:26b"
OLLAMA_CHAT_URL = "http://localhost:11434/api/chat"
OLLAMA_EMBED_URL = "http://localhost:11434/api/embeddings"
MAX_CONTEXT_TOKENS = 16384
BOOKS_TOP_K = 4
PROJECT_TOP_K = 6
MEMORY_TOP_K = 2
BOOKS_MAX_CHARS = 4000
PROJECT_MAX_CHARS = 8000
MEMORY_MAX_CHARS = 1200
RECENT_MESSAGES_LIMIT = 4
CPU_HEAVY_THRESHOLD = 75.0

chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)
memory_collection = chroma_client.get_or_create_collection(name="chat_memory")
qdrant_client = QdrantClient("localhost", port=6333)


@dataclass
class QueryPlan:
    task_type: str
    response_mode: str
    books_top_k: int
    project_top_k: int
    needs_review: bool
    need_diff: bool
    need_architecture: bool
    ultra_short: bool


@dataclass
class RetrievedContext:
    books: str
    project: str
    memory: str


def safe_json_response(payload: Dict, status: int = 200) -> Response:
    return Response(json.dumps(payload, ensure_ascii=False), status=status, mimetype="application/json")


def safe_yield(text_content: str) -> bytes:
    payload = {"message": {"role": "assistant", "content": text_content}, "done": False}
    return json.dumps(payload, ensure_ascii=False).encode("utf-8") + b"\n"


def trim_text(text: str, limit: int) -> str:
    if not text or len(text) <= limit:
        return text
    return text[:limit] + "\n...[truncated]"


def estimate_tokens(text: str) -> int:
    if not text:
        return 0
    return max(1, len(text) // 4)


def get_cpu_load_macos() -> float:
    try:
        num_cores = int(os.popen("sysctl -n hw.ncpu").read().strip())
        load_1min = os.getloadavg()[0]
        return (load_1min / max(num_cores, 1)) * 100.0
    except Exception:
        return 20.0


def ollama_chat(model_name: str, messages: List[Dict], stream: bool = False, timeout: int = 240):
    return requests.post(
        OLLAMA_CHAT_URL,
        json={
            "model": model_name,
            "messages": messages,
            "stream": stream,
            "options": {"num_ctx": MAX_CONTEXT_TOKENS},
        },
        stream=stream,
        timeout=timeout,
    )


def call_ollama_text(model_name: str, messages: List[Dict], timeout: int = 240) -> str:
    try:
        response = ollama_chat(model_name, messages, stream=False, timeout=timeout)
        if response.status_code != 200:
            return f"Ошибка Ollama {response.status_code}: {response.text}"
        data = response.json()
        return data.get("message", {}).get("content", "")
    except Exception as exc:
        return f"Ошибка вызова модели {model_name}: {exc}"


def get_query_embedding(text: str) -> Optional[List[float]]:
    try:
        response = requests.post(
            OLLAMA_EMBED_URL,
            json={"model": EMBED_MODEL, "prompt": text},
            timeout=20,
        )
        if response.status_code != 200:
            return None
        data = response.json()
        return data.get("embedding") or data.get("embeddings")
    except Exception:
        return None


def infer_ultra_short(user_query: str, mode: str) -> bool:
    text = (user_query or "").lower()
    return mode == "ultra-short" or any(marker in text for marker in [
        "ultra-short", "ultrashort", "коротко", "в 5-8 строк", "без воды", "кратко", "в двух словах"
    ])


def build_query_plan(user_query: str, mode: str) -> QueryPlan:
    text = (user_query or "").lower()
    ultra_short = infer_ultra_short(user_query, mode)
    architecture_markers = [
        "архитект", "architecture", "refactor", "рефактор", "масштаб", "ddd", "cqrs", "event sourcing",
        "bounded context", "декомпоз", "границ", "слой", "модул"
    ]
    bugfix_markers = ["fix", "исправ", "ошиб", "bug", "panic", "не работает", "race", "goroutine", "deadlock", "channel"]
    explain_markers = ["почему", "как работает", "объясни", "explain"]

    need_architecture = mode == "consult" or any(marker in text for marker in architecture_markers)
    need_diff = any(marker in text for marker in bugfix_markers)
    task_type = "architecture" if need_architecture else "implementation"
    if not need_architecture and any(marker in text for marker in explain_markers):
        task_type = "explain"

    if task_type == "architecture":
        response_mode = "ultra-short" if ultra_short else "plan"
        return QueryPlan(task_type, response_mode, 4, 5, True, False, True, ultra_short)
    if need_diff:
        response_mode = "ultra-short" if ultra_short else "diff"
        return QueryPlan(task_type, response_mode, 3, 6, True, True, False, ultra_short)
    response_mode = "ultra-short" if ultra_short else "standard"
    return QueryPlan(task_type, response_mode, 3, 4, False, False, False, ultra_short)


def planner_refine_plan(user_query: str, base_plan: QueryPlan, recent_messages: List[Dict]) -> QueryPlan:
    planner_prompt = (
        "Ты планировщик RAG для Go-помощника внутри IDE.\n"
        "Твоя задача — не отвечать на вопрос, а только уточнить план retrieval/answering.\n"
        "Верни JSON с полями: task_type, response_mode, books_top_k, project_top_k, needs_review, need_diff, need_architecture, ultra_short.\n"
        "Не усложняй. Если запрос локальный, оставь минимальный retrieval. Если архитектурный — можно усилить review."
    )
    messages = [{"role": "system", "content": planner_prompt}]
    for msg in recent_messages[-2:]:
        role = msg.get("role")
        content = (msg.get("content") or "").strip()
        if role in {"user", "assistant"} and content:
            messages.append({"role": role, "content": trim_text(content, 500)})
    messages.append({
        "role": "user",
        "content": (
            f"Исходный запрос: {user_query}\n"
            f"Текущий базовый план: {json.dumps(base_plan.__dict__, ensure_ascii=False)}\n"
            "Верни только JSON."
        )
    })
    raw = call_ollama_text(PLANNER_MODEL, messages, timeout=90).strip()
    try:
        start = raw.find("{")
        end = raw.rfind("}")
        if start == -1 or end == -1:
            return base_plan
        data = json.loads(raw[start:end + 1])
        return QueryPlan(
            task_type=data.get("task_type", base_plan.task_type),
            response_mode=data.get("response_mode", base_plan.response_mode),
            books_top_k=max(1, min(int(data.get("books_top_k", base_plan.books_top_k)), 6)),
            project_top_k=max(1, min(int(data.get("project_top_k", base_plan.project_top_k)), 8)),
            needs_review=bool(data.get("needs_review", base_plan.needs_review)),
            need_diff=bool(data.get("need_diff", base_plan.need_diff)),
            need_architecture=bool(data.get("need_architecture", base_plan.need_architecture)),
            ultra_short=bool(data.get("ultra_short", base_plan.ultra_short)),
        )
    except Exception:
        return base_plan


def query_qdrant(collection_name: str, query_vector: List[float], limit: int) -> List[Dict]:
    try:
        result = qdrant_client.query_points(collection_name=collection_name, query=query_vector, limit=limit)
        return [point.payload or {} for point in result.points]
    except Exception as exc:
        print(f"[!] Ошибка поиска в Qdrant ({collection_name}): {exc}")
        return []


def render_book_blocks(items: List[Dict]) -> str:
    blocks = []
    for item in items:
        source = item.get("source", "Книга")
        content = item.get("content", "")
        blocks.append(f"Источник: {source}\nФрагмент:\n{content}")
    return "\n---\n".join(blocks)


def render_project_blocks(items: List[Dict]) -> str:
    blocks = []
    for item in items:
        filepath = item.get("file_path", "unknown")
        component = item.get("component", "unknown")
        related = item.get("related_modules", [])
        content = item.get("content", "")
        blocks.append(
            f"Файл: {filepath}\nКомпонент: {component}\nRelated modules: {related}\nКод:\n{content}"
        )
    return "\n---\n".join(blocks)


def get_relevant_chat_memory(query_vector: List[float]) -> str:
    try:
        if memory_collection.count() == 0:
            return ""
        results = memory_collection.query(query_embeddings=[query_vector], n_results=MEMORY_TOP_K)
        docs = results.get("documents", [[]])
        metas = results.get("metadatas", [[]])
        ids = results.get("ids", [[]])
        if not docs or not docs[0]:
            return ""
        current_time = time.time()
        chunks: List[str] = []
        for doc, meta, doc_id in zip(docs[0], metas[0], ids[0]):
            score = float(meta.get("score", 1.0))
            timestamp = float(meta.get("timestamp", current_time))
            days_passed = (current_time - timestamp) / 86400
            decayed = score * (0.95 ** days_passed)
            memory_collection.update(ids=[doc_id], metadatas=[{"score": min(score + 0.1, 3.0), "timestamp": current_time}])
            if decayed >= 0.45:
                chunks.append(doc)
        return "\n---\n".join(chunks)
    except Exception as exc:
        print(f"[!] Ошибка чтения памяти: {exc}")
        return ""


def save_to_long_term_memory(user_query: str, ai_answer: str, query_vector: Optional[List[float]]) -> None:
    if not query_vector or not ai_answer.strip():
        return
    try:
        memory_id = f"mem_{int(time.time() * 1000)}"
        compact_answer = trim_text(ai_answer, 1200)
        combined = f"Вопрос пользователя: {user_query}\nКраткий ответ: {compact_answer}"
        memory_collection.add(
            embeddings=[query_vector],
            documents=[combined],
            ids=[memory_id],
            metadatas=[{"score": 1.0, "timestamp": time.time()}],
        )
    except Exception as exc:
        print(f"[!] Ошибка сохранения памяти: {exc}")


def build_context(query_vector: Optional[List[float]], plan: QueryPlan) -> RetrievedContext:
    if not query_vector:
        return RetrievedContext("", "", "")
    book_items = query_qdrant("books_collection", query_vector, plan.books_top_k)
    project_items = query_qdrant("go_project_context", query_vector, plan.project_top_k)
    memory_text = get_relevant_chat_memory(query_vector)
    return RetrievedContext(
        books=trim_text(render_book_blocks(book_items), BOOKS_MAX_CHARS),
        project=trim_text(render_project_blocks(project_items), PROJECT_MAX_CHARS),
        memory=trim_text(memory_text, MEMORY_MAX_CHARS),
    )


def build_system_prompt(plan: QueryPlan, context: RetrievedContext) -> str:
    if plan.response_mode == "ultra-short":
        answer_contract = (
            "Формат ответа: 5-8 строк максимум, без вводных фраз и без воды.\n"
            "Структура: диагноз -> что делать -> если нужен diff, покажи только ключевой фрагмент."
        )
    elif plan.response_mode == "diff":
        answer_contract = (
            "Формат ответа:\n"
            "1. Короткий диагноз.\n"
            "2. Минимальный Go diff или точечный код.\n"
            "3. Краткое объяснение.\n"
            "4. Что проверить после изменения."
        )
    elif plan.response_mode == "plan":
        answer_contract = (
            "Формат ответа:\n"
            "1. Диагноз текущего решения.\n"
            "2. Целевая архитектура без лишних паттернов.\n"
            "3. Пошаговый план миграции.\n"
            "4. Риски и критерии успеха."
        )
    else:
        answer_contract = (
            "Формат ответа:\n"
            "1. Короткий ответ по сути.\n"
            "2. Причина проблемы.\n"
            "3. Минимальное изменение.\n"
            "4. Что проверить."
        )

    return (
        "Ты Go-помощник внутри IDE с RAG по книгам и коду проекта.\n"
        "Сначала определи, чего хочет пользователь. Затем опирайся на код проекта. После этого применяй правила из книг к найденному коду.\n"
        "Приоритет источников: код проекта > книги > память.\n\n"
        "Жесткие правила:\n"
        "- Не отвечай на другой вопрос, кроме исходного.\n"
        "- Не придумывай пакеты, файлы, функции и зависимости, которых нет в проектном контексте.\n"
        "- Не пересказывай книги абстрактно: используй их как проверку и объяснение для конкретного кода.\n"
        "- Не навязывай DDD/CQRS/Event Sourcing без явного запроса.\n"
        "- Если контекста не хватает, скажи этого прямо.\n"
        "- Предпочитай минимальное изменение существующего решения.\n\n"
        f"{answer_contract}\n\n"
        f"КНИГИ:\n{context.books or 'Нет данных.'}\n\n"
        f"КОД ПРОЕКТА:\n{context.project or 'Нет данных.'}\n\n"
        f"ПАМЯТЬ:\n{context.memory or 'Нет данных.'}"
    )


def build_messages(system_prompt: str, incoming_messages: List[Dict], user_query: str) -> List[Dict]:
    recent = incoming_messages[-(RECENT_MESSAGES_LIMIT + 1):-1] if len(incoming_messages) > 1 else []
    cleaned_recent = []
    for msg in recent:
        role = msg.get("role")
        content = (msg.get("content") or "").strip()
        if role in {"user", "assistant"} and content:
            cleaned_recent.append({"role": role, "content": trim_text(content, 1000)})
    return [{"role": "system", "content": system_prompt}] + cleaned_recent + [{"role": "user", "content": user_query}]


def build_reviewer_prompt(user_query: str, plan: QueryPlan) -> str:
    extra = "Проверь, не стал ли ответ длиннее 8 строк." if plan.ultra_short else "Проверь, не стал ли ответ излишне многословным."
    return (
        f"Проведи короткий аудит ответа на исходный запрос: '{user_query}'.\n"
        "Проверь только следующее:\n"
        "1. Есть ли уход в сторону от вопроса пользователя.\n"
        "2. Есть ли выдуманные сущности, которых нет в коде проекта.\n"
        "3. Есть ли плохие советы для Go: race, deadlock, goroutine leak, channel misuse, context misuse.\n"
        f"4. {extra}\n"
        "Верни только список замечаний. Если всё хорошо, верни OK."
    )


def build_final_prompt(user_query: str, review_feedback: str, plan: QueryPlan) -> str:
    brevity = "Ответ должен остаться в 5-8 строк." if plan.ultra_short else "Сохрани ответ коротким и прикладным."
    return (
        f"Исходный запрос пользователя: '{user_query}'.\n"
        f"Замечания ревьюера:\n{review_feedback}\n\n"
        "Пересобери финальный ответ.\n"
        "Исправь только реальные замечания.\n"
        "Не добавляй новые идеи, если они не нужны для решения вопроса.\n"
        f"{brevity}"
    )


@app.route("/api/tags", methods=["GET"])
def mock_tags():
    return safe_json_response({
        "models": [
            {"name": ANSWER_MODEL, "model": ANSWER_MODEL},
            {"name": REVIEW_MODEL, "model": REVIEW_MODEL},
        ]
    })


@app.route("/api/show", methods=["POST"])
def mock_show():
    return safe_json_response({"modelfile": f"FROM {ANSWER_MODEL}"})


@app.after_request
def add_cors_headers(response):
    response.headers["Access-Control-Allow-Origin"] = "*"
    response.headers["Access-Control-Allow-Headers"] = "Content-Type, Authorization"
    return response


@app.route("/api/rebuild-db", methods=["POST"])
def manual_rebuild_db():
    thread = threading.Thread(target=rebuild_vector_db, daemon=True)
    thread.start()
    return {"status": "rebuilding", "message": "Пересборка БД Qdrant запущена в фоне"}


@app.route("/api/chat", methods=["POST"])
def api_chat():
    data = request.json or {}
    incoming_messages = data.get("messages", [])
    mode = data.get("mode", "default")

    if not incoming_messages:
        return safe_json_response({"message": {"role": "assistant", "content": "Нет сообщений."}})

    user_query = (incoming_messages[-1].get("content") or "").strip()
    if not user_query:
        return safe_json_response({"message": {"role": "assistant", "content": "Пустой запрос."}})

    cpu_load = get_cpu_load_macos()
    silent_mode = cpu_load > CPU_HEAVY_THRESHOLD
    print(f"\n[📥] Запрос из IDE | CPU load: {cpu_load:.1f}% | mode={mode}")

    query_vector = get_query_embedding(user_query)
    base_plan = build_query_plan(user_query, mode)
    plan = planner_refine_plan(user_query, base_plan, incoming_messages) if not silent_mode else base_plan
    context = build_context(query_vector, plan)
    system_prompt = build_system_prompt(plan, context)
    base_messages = build_messages(system_prompt, incoming_messages, user_query)

    def generate():
        full_answer = ""
        try:
            if not silent_mode:
                yield safe_yield(f"[Plan] task={plan.task_type}, mode={plan.response_mode}, books={plan.books_top_k}, code={plan.project_top_k}\n\n")
                yield safe_yield("[RAG] Сначала собрал план, затем достал код, потом применил знания из книг.\n\n")

            draft_answer = call_ollama_text(ANSWER_MODEL, base_messages)
            if not draft_answer.strip():
                yield safe_yield("Пустой ответ от основной модели.")
                yield b'{"done": true}\n'
                return

            final_messages = base_messages + [{"role": "assistant", "content": draft_answer}]

            if plan.needs_review and not silent_mode:
                yield safe_yield("[Review] Проверяю ответ на уход в сторону, выдумки и лишнюю сложность.\n\n")
                review_prompt = build_reviewer_prompt(user_query, plan)
                review_feedback = call_ollama_text(
                    REVIEW_MODEL,
                    final_messages + [{"role": "user", "content": review_prompt}],
                )
                if review_feedback.strip() and review_feedback.strip().upper() != "OK":
                    final_prompt = build_final_prompt(user_query, review_feedback, plan)
                    final_messages = final_messages + [
                        {"role": "user", "content": review_prompt},
                        {"role": "assistant", "content": review_feedback},
                        {"role": "user", "content": final_prompt},
                    ]

            stream_model = REVIEW_MODEL if plan.needs_review and not silent_mode else ANSWER_MODEL
            response = ollama_chat(stream_model, final_messages, stream=True, timeout=360)
            if response.status_code != 200:
                error_text = f"Ollama ошибка {response.status_code}: {response.text}"
                yield safe_yield(error_text)
                yield b'{"done": true}\n'
                return

            for line in response.iter_lines():
                if not line:
                    continue
                yield line + b"\n"
                try:
                    chunk_json = json.loads(line.decode("utf-8"))
                    content = chunk_json.get("message", {}).get("content", "")
                    full_answer += content
                except json.JSONDecodeError:
                    continue

            if not full_answer.strip():
                yield safe_yield("[⚠️] Модель вернула пустой ответ. Проверь длину контекста или память.")
            else:
                save_to_long_term_memory(user_query, full_answer, query_vector)
        except Exception as exc:
            yield safe_yield(f"[Ошибка]: {exc}")
            yield b'{"done": true}\n'

    return Response(generate(), mimetype="application/x-ndjson")


if __name__ == "__main__":
    crawler = threading.Thread(target=async_knowledge_crawler, daemon=True)
    crawler.start()
    print("[🚀] RAG server запущен на http://localhost:5001")
    app.run(host="localhost", port=5001, use_reloader=False)
