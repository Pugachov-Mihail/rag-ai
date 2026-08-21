import os
import re
import threading
import chromadb
import requests
import time
import subprocess
import hashlib
import json
from flask import Flask, request, Response
from qdrant_client import QdrantClient

# Импортируем ваши воркеры (их тоже нужно перевести на Qdrant для сохранения)
from reader_book import sync_books, sync_go_project, async_knowledge_crawler, rebuild_vector_db

app = Flask(__name__)

## 1. Конфигурация моделей и путей
CHROMA_PATH = "./my_library_db"
EMBED_MODEL = "nomic-embed-text"

MODEL_CREATOR = "deepseek-coder-v2:16b"
MODEL_CRITIC = "gemma4:26b"  # <-- Интегрирована тяжелая модель для аудита
LLM_MODEL = "qwen2.5-coder:7b"
ADVISOR_MODEL = "qwen2.5-coder:14b"

OPENCODE_BIN = "/Users/pugachev.mihail9/.opencode/bin/opencode"
GO_PROJECT_PATH = "/Users/pugachev.mihail9/Desktop/work_app/calculator-v2"
HASH_CACHE_FILE = "./go_project_hashes.json"

OLLAMA_CHAT_URL = "http://localhost:11434/api/chat"
OLLAMA_EMBED_URL = "http://localhost:11434/api/embeddings"

MAX_CONTEXT_TOKENS = 16384
CONTEXT_WARN_THRESHOLD = 0.85
CPU_HEAVY_THRESHOLD = 65.0

## 2. Инициализация баз данных
# ChromaDB - только для памяти (логика затухания)
chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)
memory_collection = chroma_client.get_or_create_collection(name="chat_memory")

# Qdrant - для книг и структуры проекта (AST)
qdrant_client = QdrantClient("localhost", port=6333)


def get_cpu_load_macos():
    try:
        num_cores = int(os.popen("sysctl -n hw.ncpu").read().strip())
        load_1min = os.getloadavg()[0]
        return (load_1min / num_cores) * 100.0
    except:
        return 20.0


def get_query_embedding(text):
    try:
        response = requests.post(OLLAMA_EMBED_URL, json={"model": EMBED_MODEL, "prompt": text}, timeout=10)
        return response.json().get("embedding") or response.json().get("embeddings")
    except:
        return None


def estimate_tokens(text):
    if not text: return 0
    return len(text) // 3


def should_skip_consultation(messages, max_tokens=MAX_CONTEXT_TOKENS):
    total_tokens = sum(estimate_tokens(msg.get("content", "")) for msg in messages)
    ratio = total_tokens / max_tokens
    if ratio > CONTEXT_WARN_THRESHOLD:
        print(f"[🔍 КОНТЕКСТ] Занято {ratio * 100:.0f}% — консультация пропускается")
        return True
    return False


## 3. Функции поиска по Qdrant
def get_qdrant_context(query_vector, collection_name, limit=3):
    """Ищет архитектурные блоки в Qdrant и извлекает связи related_modules"""
    try:
        results = qdrant_client.query_points(
            collection_name= collection_name, #"books_collection",
            query=query_vector,
            limit=limit
        )

        context_blocks = []
        for r in results.points:
            meta = r.payload
            if collection_name == "go_project_context":
                component = meta.get('component', 'unknown')
                dependencies = meta.get('related_modules', [])
                filepath = meta.get('file_path', 'unknown')

                block = (
                    f"Файл: {filepath} | Компонент: {component}\n"
                    f"Вызывает зависимости (важно для графа маршрутизации): {dependencies}\n"
                    f"Код:\n{meta.get('content')}"
                )
                context_blocks.append(block)
            else:
                block = f"Источник: {meta.get('source', 'Книга')}\n{meta.get('content')}"
                context_blocks.append(block)

        return "\n---\n".join(context_blocks)
    except Exception as e:
        print(f"[!] Ошибка поиска в Qdrant ({collection_name}): {e}")
        return ""


## 4. Память и OpenCode
def get_relevant_chat_memory(user_query, query_vector):
    if memory_collection.count() == 0: return ""
    results = memory_collection.query(query_embeddings=[query_vector], n_results=3)
    if not results or not results.get('documents') or not results['documents']: return ""
    memory_context = []
    current_time = time.time()
    for doc, meta, doc_id in zip(results['documents'][0], results['metadatas'][0], results['ids'][0]):
        score = meta.get('score', 1.0)
        timestamp = meta.get('timestamp', current_time)
        days_passed = (current_time - timestamp) / 86400
        decayed_score = score * (0.95 ** days_passed)
        memory_collection.update(ids=[doc_id], metadatas=[{"score": score + 0.3, "timestamp": current_time}])
        if decayed_score > 0.4:
            memory_context.append(f"[Из прошлого опыта]:\n{doc}")
    return "\n---\n".join(memory_context)


def save_to_long_term_memory(user_query, ai_answer, query_vector):
    memory_id = f"mem_{int(time.time() * 1000)}"
    combined_text = f"Вопрос пользователя: {user_query}\nОтвет ИИ: {ai_answer}"
    memory_collection.add(embeddings=[query_vector], documents=[combined_text], ids=[memory_id],
                          metadatas=[{"score": 1.0, "timestamp": time.time()}])


def call_ollama(model_name, messages):
    try:
        response = requests.post(
            OLLAMA_CHAT_URL,
            json={
                "model": model_name,
                "messages": messages,
                "stream": False,
                "options": {"num_ctx": MAX_CONTEXT_TOKENS}
            },
            timeout=180
        )
        res_json = response.json()
        if "message" in res_json:
            return res_json["message"]["content"]
        elif "error" in res_json:
            return f"Ошибка Ollama: {res_json['error']}"
        return f"Неизвестный формат: {res_json}"
    except Exception as e:
        return f"Ошибка вызова модели {model_name}: {e}"


def consult_advisor(model_critic_draft, user_query, context_info, go_project_path=GO_PROJECT_PATH):
    # ЖЕСТКАЯ ОБРЕЗКА ДЛЯ ВНЕШНЕЙ ИИ (Защита от лимитов RPC 16k)
    safe_context = context_info[:3000]
    safe_draft = model_critic_draft[:3000]

    advisor_prompt = (
        f"Ты — внешний консультант Go-архитектора.\nВопрос пользователя: '{user_query}'\n\n"
        f"Справочная информация (выжимка):\n{safe_context}\n\n"
        f"Черновик решения (выжимка):\n{safe_draft}\n\n"
        f"Укажи 3-5 конкретных багов или узких мест конкурентности. Только нумерованный список."
    )

    try:
        # Отправка во внешний агент
        result = subprocess.run([OPENCODE_BIN, "run", "--agent", "plan", "--dir", go_project_path, advisor_prompt],
                                capture_output=True, text=True, encoding='utf-8', timeout=90)
        if result.returncode == 0 and result.stdout.strip(): return result.stdout
    except:
        pass

    # Фолбэк на внешнюю модель
    return call_ollama(ADVISOR_MODEL, [{"role": "user", "content": advisor_prompt}])


## 5. Flask Endpoints
@app.route('/api/tags', methods=['GET'])
def mock_tags(): return Response(json.dumps({"models": [{"name": MODEL_CRITIC, "model": MODEL_CRITIC}]}),
                                 mimetype='application/json')


@app.route('/api/show', methods=['POST'])
def mock_show(): return Response(json.dumps({"modelfile": f"FROM {MODEL_CRITIC}"}), mimetype='application/json')


@app.after_request
def add_cors_headers(response):
    response.headers['Access-Control-Allow-Origin'] = '*'
    response.headers['Access-Control-Allow-Headers'] = 'Content-Type, Authorization'
    return response


def safe_yield(text_content):
    payload = {"message": {"role": "assistant", "content": text_content}, "done": False}
    return json.dumps(payload).encode('utf-8') + b'\n'


@app.route('/api/chat', methods=['POST'])
def mock_chat():
    data = request.json or {}
    messages = data.get("messages", [])
    if not messages:
        return json.dumps({"message": {"role": "assistant", "content": "Нет сообщений."}})

    user_query = messages[-1]["content"]
    current_cpu_load = get_cpu_load_macos()
    silent_mode = current_cpu_load > CPU_HEAVY_THRESHOLD

    print(f"\n[📥 ВХОДЯЩИЙ ЗАПРОС ИЗ GOLAND] Загрузка CPU: {current_cpu_load:.1f}%")

    query_vector = get_query_embedding(user_query)

    # Ищем контекст в Qdrant
    books_context = get_qdrant_context(query_vector, "books_collection", limit=2) if query_vector else ""
    project_context = get_qdrant_context(query_vector, "go_project_context", limit=4) if query_vector else ""
    long_term_memory_context = get_relevant_chat_memory(user_query, query_vector) if query_vector else ""

    system_instruction = (
        "Ты — ведущий ИИ-архитектор систем на Go.\n"
        f"1. КНИГИ:\n{books_context}\n\n"
        f"2. ИСХОДНЫЙ КОД ПРОЕКТА И ЗАВИСИМОСТИ:\n{project_context}\n\n"
        f"3. ИСТОРИЯ ДИАЛОГОВ:\n{long_term_memory_context}\n\n"
        "🚨 ПРАВИЛА: Не выдумывай пакеты, которых нет в проекте. Опирайся на related_modules. Выводи точечные Go-диффы."
    )

    agent_messages = [{"role": "system", "content": system_instruction}] + messages[:-1] + [
        {"role": "user", "content": user_query}]

    def generate():
        nonlocal agent_messages
        full_answer = ""
        try:
            if not silent_mode:
                yield safe_yield("[🧠 Шаг 1/3: Проектирование архитектурного черновика...]\n\n")

                draft_messages = agent_messages + [
                    {"role": "user", "content": "Напиши подробный технический черновик ответа."}]
                draft_content = call_ollama(LLM_MODEL, draft_messages)

                yield safe_yield(f"[🕵️ Шаг 2/3: Аудит ревьюера ({MODEL_CRITIC}) на утечки и связи...]\n\n")

                critic_prompt = (
                    f"Вот черновик ответа:\n{draft_content}\n\n"
                    "Проведи аудит. Проверь на утечки каналов/горутин. Убедись, что предложенные функции существуют в графе related_modules контекста. Выпиши замечания."
                )
                critic_feedback = call_ollama(MODEL_CRITIC,
                                              agent_messages + [{"role": "assistant", "content": draft_content},
                                                                {"role": "user", "content": critic_prompt}])

                context_for_consultation = f"Проект (суть): {project_context[:1000]}\nАудит: {critic_feedback[:1000]}"

                # Игнорируем проверку токенов для советника, отправляем укороченный запрос
                yield safe_yield("[💬] Запрос совета у внешней ИИ (с защитой от RPC лимита)...\n")
                advisor_feedback = consult_advisor(draft_content, user_query, context_for_consultation)

                final_prompt = (
                    f"Вопрос: '{user_query}'.\nЧерновик, аудит и советник переданы ранее.\n"
                    "Сформируй итоговый ответ НА РУССКОМ ЯЗЫКЕ.\n"
                    "1. РАЗБОР ПО ЗАПРОСУ: Четкий ответ.\n"
                    "2. СКРЫТЫЕ ОШИБКИ И УТЕЧКИ: Анализ багов конкурентности.\n"
                    "3. ИСПРАВЛЕННЫЙ GO-КОД: Готовый дифф.\n"
                )

                # Собираем ПОЛНЫЙ контекст исключительно для локальной Gemma4:26b
                final_messages = agent_messages + [
                    {"role": "assistant", "content": draft_content},
                    {"role": "user", "content": critic_prompt},
                    {"role": "assistant", "content": critic_feedback},
                    {"role": "user", "content": "Советник:\n" + advisor_feedback},
                    {"role": "assistant", "content": "Принято."},
                    {"role": "user", "content": final_prompt}
                ]
            else:
                final_messages = agent_messages

            yield safe_yield(f"\n[🧠 Шаг 3/3] Финальный анализатор: {MODEL_CRITIC}...\n\n")
            print(f"\n==================== ОТВЕТ ИИ ({MODEL_CRITIC}) ====================")

            # Финальный потоковый вывод через тяжелую модель
            response = requests.post(
                OLLAMA_CHAT_URL,
                json={"model": MODEL_CRITIC, "messages": final_messages, "stream": True,
                      "options": {"num_ctx": MAX_CONTEXT_TOKENS}},
                stream=True, timeout=180
            )

            # ПРОВЕРКА НА ОШИБКИ OLLAMA
            if response.status_code != 200:
                error_text = f"Ollama вернула ошибку {response.status_code}: {response.text}"
                print(f"\n[!] КРИТИЧЕСКАЯ ОШИБКА: {error_text}")
                yield safe_yield(f"\n[Критическая ошибка]: {error_text}")
                yield b'{"done": True}\n'
                return

            # ПОТОКОВОЕ ЧТЕНИЕ И ВЫВОД В ТЕРМИНАЛ
            for line in response.iter_lines():
                if line:
                    # Отправляем строку в IDE (Goland)
                    yield line + b'\n'

                    # Парсим строку для вывода в консоль
                    try:
                        chunk_json = json.loads(line.decode('utf-8'))
                        if "message" in chunk_json and "content" in chunk_json["message"]:
                            content = chunk_json["message"]["content"]
                            # Печатаем прямо в терминал без переноса строки (flush=True обязательно)
                            print(content, end="", flush=True)
                            full_answer += content
                    except json.JSONDecodeError:
                        pass

            print("\n===================================================================\n")

            if query_vector and full_answer:
                save_to_long_term_memory(user_query, full_answer, query_vector)
                print("[📢] Ответ сохранен в долгосрочную память ChromaDB.")

        except Exception as e:
            error_msg = f"\n[Ошибка консорциума]: {str(e)}"
            print(error_msg)
            yield safe_yield(error_msg)
            yield b'{"done": True}\n'

    return Response(generate(), mimetype='application/x-ndjson')

@app.route('/api/rebuild-db', methods=['POST'])
def manual_rebuild_db():
    """Ручной запуск полной пересборки базы данных"""
    # Запускаем в отдельном потоке, чтобы не блокировать ответ сервера
    thread = threading.Thread(target=rebuild_vector_db)
    thread.start()
    return {"status": "rebuilding", "message": "Пересборка БД Qdrant запущена в фоне"}

if __name__ == "__main__":
    crawler = threading.Thread(target=async_knowledge_crawler, daemon=True)
    crawler.start()
    print("[🚀] Сервер Суперагента запущен на http://localhost:5001")
    app.run(host='localhost', port=5001, use_reloader=False)