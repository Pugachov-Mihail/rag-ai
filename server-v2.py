import json
import os
import threading
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import chromadb
import requests
from flask import Flask, Response, jsonify, request
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
CPU_HEAVY_THRESHOLD = 75.0
MEMORY_TOP_K = 2
RECENT_MESSAGES_LIMIT = 4

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
    code_query: str
    books_query: str
    memory_query: str


@dataclass
class RetrievedContext:
    books: str
    project: str
    memory: str
    meta: Dict


def get_cpu_load_macos() -> float:
    try:
        num_cores = int(os.popen("sysctl -n hw.ncpu").read().strip())
        load_1min = os.getloadavg()[0]
        return (load_1min / max(num_cores, 1)) * 100.0
    except Exception:
        return 20.0


def trim_text(text: str, limit: int) -> str:
    if not text or len(text) <= limit:
        return text
    return text[:limit] + "\n...[truncated]"


def to_json_line(payload: Dict) -> bytes:
    return (json.dumps(payload, ensure_ascii=False) + "\n").encode("utf-8")


def make_ollama_chunk(text: str, done: bool = False) -> bytes:
    return to_json_line({"message": {"role": "assistant", "content": text}, "done": done})


def get_query_embedding(text: str) -> Optional[List[float]]:
    try:
        response = requests.post(OLLAMA_EMBED_URL, json={"model": EMBED_MODEL, "prompt": text}, timeout=20)
        if response.status_code != 200:
            return None
        data = response.json()
        return data.get("embedding") or data.get("embeddings")
    except Exception:
        return None


def ollama_chat(model_name: str, messages: List[Dict], stream: bool = False, timeout: int = 240):
    return requests.post(
        OLLAMA_CHAT_URL,
        json={"model": model_name, "messages": messages, "stream": stream, "options": {"num_ctx": MAX_CONTEXT_TOKENS}},
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


def infer_ultra_short(user_query: str, mode: str) -> bool:
    text = (user_query or "").lower()
    return mode == "ultra-short" or any(marker in text for marker in ["ultra-short", "коротко", "без воды", "в 5-8 строк", "кратко"])


def build_base_plan(user_query: str, mode: str) -> QueryPlan:
    text = (user_query or "").lower()
    ultra = infer_ultra_short(user_query, mode)
    need_arch = mode == "consult" or any(x in text for x in ["архитект", "рефактор", "ddd", "cqrs", "слой", "модул", "bounded context"])
    need_diff = any(x in text for x in ["fix", "исправ", "ошиб", "bug", "не работает", "panic", "race", "deadlock", "утеч"])
    task_type = "architecture" if need_arch else "implementation"
    response_mode = "ultra-short" if ultra else ("plan" if need_arch else ("diff" if need_diff else "standard"))
    return QueryPlan(
        task_type=task_type,
        response_mode=response_mode,
        books_top_k=4 if need_arch else 3,
        project_top_k=6 if need_diff or need_arch else 4,
        needs_review=True if need_arch or need_diff else False,
        need_diff=need_diff,
        need_architecture=need_arch,
        ultra_short=ultra,
        code_query=user_query,
        books_query=user_query,
        memory_query=user_query,
    )


def planner_refine(user_query: str, base_plan: QueryPlan, incoming_messages: List[Dict]) -> QueryPlan:
    planner_prompt = (
        "Ты planner для IDE coding assistant с RAG. Не отвечай на вопрос пользователя.\n"
        "Верни JSON с полями: task_type, response_mode, books_top_k, project_top_k, needs_review, need_diff, need_architecture, ultra_short, code_query, books_query, memory_query.\n"
        "code_query должен быть оптимизирован под поиск по коду, books_query — под поиск в книгах, memory_query — под память.\n"
        "Не усложняй. Дай короткие retrieval-запросы."
    )
    msgs = [{"role": "system", "content": planner_prompt}]
    for msg in incoming_messages[-2:]:
        role = msg.get("role")
        content = (msg.get("content") or "").strip()
        if role in {"user", "assistant"} and content:
            msgs.append({"role": role, "content": trim_text(content, 500)})
    msgs.append({"role": "user", "content": f"Запрос: {user_query}\nБазовый план: {json.dumps(base_plan.__dict__, ensure_ascii=False)}\nВерни только JSON."})
    raw = call_ollama_text(PLANNER_MODEL, msgs, timeout=90).strip()
    try:
        start, end = raw.find("{"), raw.rfind("}")
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
            code_query=(data.get("code_query") or base_plan.code_query)[:200],
            books_query=(data.get("books_query") or base_plan.books_query)[:200],
            memory_query=(data.get("memory_query") or base_plan.memory_query)[:200],
        )
    except Exception:
        return base_plan


def query_qdrant(collection_name: str, query_text: str, limit: int) -> List[Dict]:
    vector = get_query_embedding(query_text)
    if not vector:
        return []
    try:
        res = qdrant_client.query_points(
            collection_name=collection_name,
            query=vector,          # сюда идёт list[float]
            limit=limit,
            with_payload=True,
            with_vectors=False,
        )
        return [p.payload or {} for p in res.points]
    except Exception as exc:
        print(f"[!] Qdrant query_points error in {collection_name}: {exc}")
        return []


def get_relevant_chat_memory(query_text: str) -> str:
    try:
        vector = get_query_embedding(query_text)
        if not vector or memory_collection.count() == 0:
            return ""
        results = memory_collection.query(query_embeddings=[vector], n_results=MEMORY_TOP_K)
        docs = results.get("documents", [[]])
        metas = results.get("metadatas", [[]])
        ids = results.get("ids", [[]])
        if not docs or not docs[0]:
            return ""
        current_time = time.time()
        chunks = []
        for doc, meta, doc_id in zip(docs[0], metas[0], ids[0]):
            score = float(meta.get("score", 1.0))
            timestamp = float(meta.get("timestamp", current_time))
            decayed = score * (0.95 ** ((current_time - timestamp) / 86400))
            memory_collection.update(ids=[doc_id], metadatas=[{"score": min(score + 0.1, 3.0), "timestamp": current_time}])
            if decayed >= 0.45:
                chunks.append(doc)
        return "\n---\n".join(chunks)
    except Exception as exc:
        print(f"[!] Memory error: {exc}")
        return ""


def render_blocks(items: List[Dict], kind: str) -> Tuple[str, List[str]]:
    blocks, refs = [], []
    for item in items:
        if kind == "code":
            file_path = item.get("file_path", "unknown")
            component = item.get("component", "unknown")
            related = item.get("related_modules", [])
            refs.append(file_path)
            blocks.append(f"Файл: {file_path}\nКомпонент: {component}\nRelated: {related}\nКод:\n{item.get('content', '')}")
        else:
            source = item.get("source", "Книга")
            refs.append(source)
            blocks.append(f"Источник: {source}\nФрагмент:\n{item.get('content', '')}")
    return "\n---\n".join(blocks), refs


def build_context(plan: QueryPlan) -> RetrievedContext:
    code_items = query_qdrant("go_project_context", plan.code_query, plan.project_top_k)
    book_items = query_qdrant("books_collection", plan.books_query, plan.books_top_k)
    memory_text = get_relevant_chat_memory(plan.memory_query)
    code_text, code_refs = render_blocks(code_items, "code")
    book_text, book_refs = render_blocks(book_items, "book")
    return RetrievedContext(
        books=trim_text(book_text, 4000),
        project=trim_text(code_text, 9000),
        memory=trim_text(memory_text, 1200),
        meta={
            "code_query": plan.code_query,
            "books_query": plan.books_query,
            "memory_query": plan.memory_query,
            "code_refs": code_refs,
            "book_refs": book_refs,
        },
    )


def build_system_prompt(plan: QueryPlan, context: RetrievedContext) -> str:
    if plan.ultra_short:
        contract = "Ответ строго 5-8 строк, без воды, сначала диагноз, затем действие, потом проверка."
    elif plan.response_mode == "diff":
        contract = "Дай короткий диагноз, затем минимальный diff/patch, затем 2-3 проверки после фикса."
    elif plan.response_mode == "plan":
        contract = "Дай короткий диагноз, целевую архитектуру, шаги миграции и риски."
    else:
        contract = "Дай прикладной ответ по существу, без длинной теории."
    return (
        "Ты coding assistant для IDE.\n"
        "Порядок: пойми вопрос -> опирайся на код проекта -> примени знания из книг -> сформируй ответ.\n"
        "Приоритет источников: код > книги > память.\n"
        "Не выдумывай файлы, функции и пакеты.\n"
        "Если контекста не хватает — скажи это явно.\n"
        f"{contract}\n\n"
        f"КОД ПРОЕКТА:\n{context.project or 'Нет данных'}\n\n"
        f"КНИГИ:\n{context.books or 'Нет данных'}\n\n"
        f"ПАМЯТЬ:\n{context.memory or 'Нет данных'}"
    )


def build_messages(system_prompt: str, incoming_messages: List[Dict], user_query: str) -> List[Dict]:
    recent = incoming_messages[-(RECENT_MESSAGES_LIMIT + 1):-1] if len(incoming_messages) > 1 else []
    cleaned = []
    for msg in recent:
        role = msg.get("role")
        content = (msg.get("content") or "").strip()
        if role in {"user", "assistant"} and content:
            cleaned.append({"role": role, "content": trim_text(content, 900)})
    return [{"role": "system", "content": system_prompt}] + cleaned + [{"role": "user", "content": user_query}]


def build_reviewer_prompt(user_query: str, plan: QueryPlan) -> str:
    length_rule = "Проверь, что ответ реально помещается в 5-8 строк." if plan.ultra_short else "Проверь, что ответ не расползся без пользы."
    return (
        f"Проведи короткий аудит ответа на запрос: '{user_query}'.\n"
        "Проверь: уход в сторону, выдуманные сущности, опасные советы по Go, лишнюю сложность.\n"
        f"{length_rule}\n"
        "Верни только замечания. Если всё хорошо, верни OK."
    )


def build_repair_prompt(user_query: str, review_feedback: str, plan: QueryPlan) -> str:
    brevity = "Оставь ответ в 5-8 строк." if plan.ultra_short else "Оставь ответ коротким."
    return (
        f"Исходный запрос: {user_query}\n"
        f"Замечания ревьюера:\n{review_feedback}\n\n"
        f"Пересобери финальный ответ. {brevity} Не добавляй новые идеи без необходимости."
    )


def save_memory(user_query: str, answer: str) -> None:
    vector = get_query_embedding(user_query)
    if not vector or not answer.strip():
        return
    try:
        memory_collection.add(
            embeddings=[vector],
            documents=[f"Вопрос пользователя: {user_query}\nКраткий ответ: {trim_text(answer, 1200)}"],
            ids=[f"mem_{int(time.time() * 1000)}"],
            metadatas=[{"score": 1.0, "timestamp": time.time()}],
        )
    except Exception as exc:
        print(f"[!] Save memory error: {exc}")


def run_pipeline(user_query: str, incoming_messages: List[Dict], mode: str, include_debug: bool) -> Tuple[str, Dict]:
    cpu_load = get_cpu_load_macos()
    silent_mode = cpu_load > CPU_HEAVY_THRESHOLD
    base_plan = build_base_plan(user_query, mode)
    plan = base_plan if silent_mode else planner_refine(user_query, base_plan, incoming_messages)
    context = build_context(plan)
    system_prompt = build_system_prompt(plan, context)
    messages = build_messages(system_prompt, incoming_messages, user_query)
    draft = call_ollama_text(ANSWER_MODEL, messages)
    final_answer = draft
    review_feedback = ""
    if plan.needs_review and not silent_mode and draft.strip():
        review_feedback = call_ollama_text(REVIEW_MODEL, messages + [{"role": "assistant", "content": draft}, {"role": "user", "content": build_reviewer_prompt(user_query, plan)}])
        if review_feedback.strip() and review_feedback.strip().upper() != "OK":
            final_answer = call_ollama_text(
                ANSWER_MODEL,
                messages
                + [{"role": "assistant", "content": draft}]
                + [{"role": "user", "content": build_repair_prompt(user_query, review_feedback, plan)}],
            )
    save_memory(user_query, final_answer)
    debug = {
        "cpu_load": cpu_load,
        "silent_mode": silent_mode,
        "plan": plan.__dict__,
        "retrieval": context.meta,
        "review_feedback": review_feedback,
    }
    if include_debug:
        return final_answer, debug
    return final_answer, {}


@app.after_request
def add_cors_headers(response):
    response.headers["Access-Control-Allow-Origin"] = "*"
    response.headers["Access-Control-Allow-Headers"] = "Content-Type, Authorization"
    return response


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "service": "multi-ide-rag"})


@app.route("/api/tags", methods=["GET"])
def ollama_tags():
    return jsonify({"models": [{"name": ANSWER_MODEL, "model": ANSWER_MODEL}, {"name": REVIEW_MODEL, "model": REVIEW_MODEL}]})


@app.route("/api/show", methods=["POST"])
def ollama_show():
    return jsonify({"modelfile": f"FROM {ANSWER_MODEL}"})


@app.route("/api/chat", methods=["POST"])
def ollama_chat_proxy():
    data = request.json or {}
    incoming_messages = data.get("messages", [])
    mode = data.get("mode", "default")
    include_debug = bool(data.get("debug", False))
    if not incoming_messages:
        return jsonify({"message": {"role": "assistant", "content": "Нет сообщений."}, "done": True})
    user_query = (incoming_messages[-1].get("content") or "").strip()
    if not user_query:
        return jsonify({"message": {"role": "assistant", "content": "Пустой запрос."}, "done": True})

    def generate():
        answer, debug = run_pipeline(user_query, incoming_messages, mode, include_debug)
        if include_debug:
            yield make_ollama_chunk(f"[debug]\n{json.dumps(debug, ensure_ascii=False, indent=2)}\n\n")
        yield make_ollama_chunk(answer)
        yield make_ollama_chunk("", done=True)

    return Response(generate(), mimetype="application/x-ndjson")


@app.route("/v1/models", methods=["GET"])
def openai_models():
    return jsonify({
        "object": "list",
        "data": [
            {"id": ANSWER_MODEL, "object": "model", "owned_by": "local"},
            {"id": REVIEW_MODEL, "object": "model", "owned_by": "local"},
        ],
    })


@app.route("/v1/chat/completions", methods=["POST"])
def openai_chat_completions():
    data = request.json or {}
    incoming_messages = data.get("messages", [])
    stream = bool(data.get("stream", True))
    mode = data.get("metadata", {}).get("mode", "default") if isinstance(data.get("metadata"), dict) else "default"
    include_debug = bool((data.get("metadata") or {}).get("debug", False)) if isinstance(data.get("metadata"), dict) else False

    if not incoming_messages:
        return jsonify({"error": {"message": "Нет сообщений."}}), 400
    user_query = (incoming_messages[-1].get("content") or "").strip()
    if not user_query:
        return jsonify({"error": {"message": "Пустой запрос."}}), 400

    answer, debug = run_pipeline(user_query, incoming_messages, mode, include_debug)
    full_text = answer if not include_debug else f"[debug]\n{json.dumps(debug, ensure_ascii=False, indent=2)}\n\n{answer}"

    if not stream:
        return jsonify({
            "id": f"chatcmpl-{int(time.time())}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": ANSWER_MODEL,
            "choices": [{"index": 0, "message": {"role": "assistant", "content": full_text}, "finish_reason": "stop"}],
        })

    def sse_stream():
        chunk = {
            "id": f"chatcmpl-{int(time.time())}",
            "object": "chat.completion.chunk",
            "created": int(time.time()),
            "model": ANSWER_MODEL,
            "choices": [{"index": 0, "delta": {"role": "assistant", "content": full_text}, "finish_reason": None}],
        }
        yield f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n"
        end_chunk = {
            "id": f"chatcmpl-{int(time.time())}",
            "object": "chat.completion.chunk",
            "created": int(time.time()),
            "model": ANSWER_MODEL,
            "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
        }
        yield f"data: {json.dumps(end_chunk, ensure_ascii=False)}\n\n"
        yield "data: [DONE]\n\n"

    return Response(sse_stream(), mimetype="text/event-stream")


@app.route("/v1/responses", methods=["POST"])
def openai_responses():
    data = request.json or {}
    input_items = data.get("input", [])
    metadata = data.get("metadata", {}) if isinstance(data.get("metadata"), dict) else {}
    mode = metadata.get("mode", "default")
    include_debug = bool(metadata.get("debug", False))

    incoming_messages = []
    for item in input_items:
        if item.get("role") in {"user", "assistant", "system"}:
            content = item.get("content")
            if isinstance(content, list):
                text_parts = [x.get("text", "") for x in content if isinstance(x, dict) and x.get("type") in {"input_text", "output_text", "text"}]
                content = "\n".join([x for x in text_parts if x])
            incoming_messages.append({"role": item.get("role"), "content": content or ""})
    if not incoming_messages:
        return jsonify({"error": {"message": "Нет input."}}), 400
    user_query = (incoming_messages[-1].get("content") or "").strip()
    answer, debug = run_pipeline(user_query, incoming_messages, mode, include_debug)
    output_text = answer if not include_debug else f"[debug]\n{json.dumps(debug, ensure_ascii=False, indent=2)}\n\n{answer}"
    return jsonify({
        "id": f"resp_{int(time.time())}",
        "object": "response",
        "created_at": int(time.time()),
        "status": "completed",
        "model": ANSWER_MODEL,
        "output": [{"type": "message", "role": "assistant", "content": [{"type": "output_text", "text": output_text}]}],
        "output_text": output_text,
    })


@app.route("/api/rebuild-db", methods=["POST"])
def manual_rebuild_db():
    thread = threading.Thread(target=rebuild_vector_db, daemon=True)
    thread.start()
    return jsonify({"status": "rebuilding", "message": "Пересборка БД Qdrant запущена в фоне"})


if __name__ == "__main__":
    crawler = threading.Thread(target=async_knowledge_crawler, daemon=True)
    crawler.start()
    print("[🚀] Multi-IDE RAG server запущен на http://localhost:5001")
    app.run(host="localhost", port=5001, use_reloader=False)
