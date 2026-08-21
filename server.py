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
from reader_book import sync_books, sync_go_project, async_knowledge_crawler, rebuild_vector_db

app = Flask(__name__)

# 1. Конфигурация
CHROMA_PATH = "./my_library_db"
EMBED_MODEL = "nomic-embed-text"

MODEL_CREATOR = "deepseek-coder-v2:16b"
MODEL_CRITIC = "codestral"
LLM_MODEL = "qwen2.5-coder:7b"
OPENCODE_BIN = "/Users/pugachev.mihail9/.opencode/bin/opencode"  # путь к opencode
ADVISOR_MODEL = "qwen2.5-coder:14b"  # фолбэк, если opencode недоступен
MAX_CONTEXT_TOKENS = 16384  # максимальный контекст модели
CONTEXT_WARN_THRESHOLD = 0.85  # предупреждение если занято >85%

# Имя файла, куда будут сохраняться хеши файлов вашего проекта
HASH_CACHE_FILE = "./go_project_hashes.json"

OLLAMA_CHAT_URL = "http://localhost:11434/api/chat"
OLLAMA_EMBED_URL = "http://localhost:11434/api/embeddings"
GO_PROJECT_PATH = "/Users/pugachev.mihail9/Desktop/work_app/calculator-v2"  # <-- УКАЖИТЕ СВОЙ ПУТЬ

CPU_HEAVY_THRESHOLD = 65.0

# Подключение к ChromaDB
chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)
books_collection = chroma_client.get_or_create_collection(name="books_collection")
project_collection = chroma_client.get_or_create_collection(name="project_collection")


# try:
#     chroma_client.delete_collection(name="chat_memory")
#     print("Сброс кеша контекста")
# except Exception as e:
#     print(f"Ошибка сброса кеша {e}")
#     pass

memory_collection = chroma_client.get_or_create_collection(name="chat_memory")


def get_cpu_load_macos():
    """Сверхбыстрый замер средней загрузки CPU на macOS без сторонних библиотек"""
    try:
        # Получаем количество ядер Mac M5
        num_cores = int(os.popen("sysctl -n hw.ncpu").read().strip())
        # Получаем среднюю загрузку за последнюю 1 минуту
        load_1min = os.getloadavg()[0]
        # Переводим в понятные проценты нагрузки
        cpu_percentage = (load_1min / num_cores) * 100.0
        return cpu_percentage
    except:
        return 20.0 # Дефолтное безопасное значение, если замер не удался


def get_query_embedding(text):
    try:
        response = requests.post(OLLAMA_EMBED_URL, json={"model": EMBED_MODEL, "prompt": text})
        return response.json().get("embedding") or response.json().get("embeddings")
    except:
        return None


def estimate_tokens(text):
    """Оценка количества токенов через символы (базовый эвристика)"""
    if not text:
        return 0
    # Средняя оценка: 1 токен ~ 3-4 символа для английского, ~2 для русского/кода
    return len(text) // 3


def should_skip_consultation(messages, max_tokens=MAX_CONTEXT_TOKENS):
    """Проверяет, достаточно ли места в контексте для консультации"""
    total_tokens = 0
    for msg in messages:
        total_tokens += estimate_tokens(msg.get("content", ""))

    ratio = total_tokens / max_tokens
    if ratio > CONTEXT_WARN_THRESHOLD:
        print(f"[🔍 КОНТЕКСТ] Занято {ratio*100:.0f}% ({total_tokens}/{max_tokens} токенов) — консультация пропускается")
        return True
    else:
        print(f"[🔍 КОНТЕКСТ] Занято {ratio*100:.0f}% ({total_tokens}/{max_tokens} токенов) — консультация допустима")
        return False


def consult_advisor(model_critic_draft, user_query, context_info, go_project_path=GO_PROJECT_PATH):
    """Запрашивает совета у opencode или фолбэк на модель"""
    advisor_prompt = (
        f"Ты — внешний консультант Go-архитектора.\n"
        f"Вопрос пользователя: '{user_query}'\n\n"
        f"Справочная информация:\n{context_info}\n\n"
        f"Черновик решения (для анализа):\n{model_critic_draft}\n\n"
        f"Задача: укажи 3-5 конкретных улучшений, возможных багов или уязвимостей в этом черновике. "
        f"Будь предельно лаконичен — только суть. Формат: нумерованный список."
    )

    # Сначала пробуем opencode
    try:
        print(f"[💬] Вызов opencode для консультации...")
        result = subprocess.run(
            [OPENCODE_BIN, "run", "--agent", "plan", "--dir", go_project_path, advisor_prompt],
            capture_output=True, text=True, encoding='utf-8', timeout=90
        )
        if result.returncode == 0 and result.stdout.strip():
            print(f"[✅] Opencode вернул совет: {len(result.stdout)} символов")
            return result.stdout
    except Exception as e:
        print(f"[⚠️] Opencode недоступен ({e}), фолбэк на Ollama...")

    # Фолбэк: Ollama
    advisor_messages = [{"role": "user", "content": advisor_prompt}]
    result = call_ollama(ADVISOR_MODEL, advisor_messages)
    return result


def get_file_hash(file_path):
    """Сверхбыстро считает SHA-256 хеш файла"""
    hasher = hashlib.sha256()
    try:
        with open(file_path, 'rb') as f:
            buf = f.read()
            hasher.update(buf)
        return hasher.hexdigest()
    except:
        return ""


def read_hash_cache():
    """Читает кэш хешей из локального JSON-файла"""
    if not os.path.exists(HASH_CACHE_FILE):
        return {}
    try:
        with open(HASH_CACHE_FILE, 'r', encoding='utf-8') as f:
            return json.load(f)
    except:
        return {}


def write_hash_cache(cache_data):
    """Записывает обновленные хеши в локальный JSON-файл"""
    try:
        with open(HASH_CACHE_FILE, 'w', encoding='utf-8') as f:
            json.dump(cache_data, f, indent=2, ensure_ascii=False)
    except Exception as e:
        print(f"[!] Не удалось обновить файл кэша хешей: {e}")


def get_go_project_context(project_path, user_query):
    """
    Интеллектуальный сбор контекста через локальный JSON-кэш.
    Если хеш Go-файла совпадает с сохраненным — блокирует вызов OpenCode
    и сообщает моделям, что кодовая база стабильна.
    """
    if not os.path.exists(project_path):
        return "Папка проекта не найдена."

    # Шаг 1: Парсим имя компонента (например, FinalTimeComponent) из вопроса пользователя
    import re
    potential_components = re.findall(r'[A-Z][a-zA-Z0-9]+', user_query)

    target_file_path = None
    target_rel_path = None

    if potential_components:
        # Берем самый первый найденный компонент
        comp_name = potential_components[0].lower().replace("component", "")
        for root, dirs, files in os.walk(project_path):
            dirs[:] = [d for d in dirs if d not in ['.git', 'vendor', 'mocks']]
            for file in files:
                if file.endswith('.go') and (comp_name in file.lower()):
                    target_file_path = os.path.join(root, file)
                    target_rel_path = os.path.relpath(target_file_path, project_path)
                    break

    # Шаг 2: Проверка хеша через локальный JSON файл
    if target_file_path and target_rel_path:
        current_hash = get_file_hash(target_file_path)
        hash_cache = read_hash_cache()

        last_saved_hash = hash_cache.get(target_rel_path, "")
        print(
            f"[🔍 Локальный Кэш] Файл: {target_rel_path} | Текущий хеш: {current_hash[:10]}... | Прошлый: {last_saved_hash[:10]}...")

        # ИДЕАЛЬНОЕ ПОПАДАНИЕ В КЭШ: Если код на диске не менялся!
        if last_saved_hash == current_hash:
            print(
                f"[🎯 КЭШ ПОПАДАНИЕ] Код файла {target_rel_path} НЕ менялся с прошлого запроса. OpenCode заблокирован!")
            return (
                f"ВНИМАНИЕ КОНСОРЦИУМУ: Исходный код файла {target_rel_path} не претерпел изменений. "
                "Опирайтесь на контекст текущей беседы и прошлый сохраненный опыт из памяти, "
                "не требуя новой выгрузки файлов."
            )

        # Если хеш изменился или файла не было в кэше — обновляем JSON
        hash_cache[target_rel_path] = current_hash
        write_hash_cache(hash_cache)
        print(f"[💾 КЭШ ОБНОВЛЕН] Записан новый хеш для {target_rel_path}")

    # Шаг 3: Код изменился -> Запускаем OpenCode для свежего анализа
    print(f"[🤖 OpenCode Agent] Код изменен или запрос новый. Запуск OpenCode для: '{user_query}'...")

    opencode_task = (
        f"Найди в Go-проекте файлы, структуры или интерфейсы, относящиеся к: '{user_query}'. "
        "🚨 ВЫВЕДИ ТОЛЬКО: Сигнатуры функций, интерфейсы и структуры данных (каркас проекта). "
        "Не выводи длинную реализацию методов, чтобы беречь контекст."
    )

    try:
        # Вызов OpenCode через абсолютный путь с флагом --dir
        result = subprocess.run(
            [
                "/Users/pugachev.mihail9/.opencode/bin/opencode", "run",  # Укажите ваш путь из which opencode
                "--agent", "plan",
                "--dir", project_path,
                opencode_task
            ],
            capture_output=True,
            text=True,
            encoding='utf-8',
            timeout=180
        )

        if result.returncode == 0 and result.stdout.strip():
            print("[🎯 OpenCode Agent] Свежий контекст успешно собран!")
            return f"ОБНОВЛЕННЫЙ АКТУАЛЬНЫЙ КОД ПРОЕКТА (СОБРАНО OPENCODE):\n{result.stdout}"
        else:
            print(f"[⚠️ OpenCode Agent] Сбой сбора: {result.stderr}")
            return "Контекст пуст. Опирайся на общие архитектурные знания Go."

    except Exception as e:
        print(f"[!] Критическая ошибка вызова OpenCode: {e}")
        return "Ошибка работы локального агента контекста."

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
        # Если это тяжелый DeepSeek на Шаге 1, даем ему 16К контекста. Для остальных — 8К.
        context_limit = 16384

        try:
            response = requests.post(
                OLLAMA_CHAT_URL,
                json={
                    "model": model_name,
                    "messages": messages,
                    "stream": False,
                    "options": {"num_ctx": context_limit}  # <-- АДАПТИВНЫЙ ЛИМИТ
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


# --- СЛУЖЕБНЫЕ ЭНДПОИНТЫ (Притворяемся моделью, которую выбрал Multi) ---
@app.route('/api/tags', methods=['GET'])
def mock_tags():
    mock_response = {
        "models": [{"name": MODEL_CRITIC, "model": MODEL_CRITIC, "details": {"format": "gguf", "family": "llama"}}]}
    return Response(json.dumps(mock_response), mimetype='application/json')


@app.route('/api/show', methods=['POST'])
def mock_show():
    mock_response = {"modelfile": f"FROM {MODEL_CRITIC}", "parameters": "", "template": "{{ .Prompt }}",
                     "details": {"format": "gguf", "family": "llama"}}
    return Response(json.dumps(mock_response), mimetype='application/json')


@app.route('/api/version', methods=['GET'])
def mock_version(): return Response(json.dumps({"version": "0.1.48"}), mimetype='application/json')


@app.route('/api/ps', methods=['GET'])
def mock_ps(): return Response(json.dumps({"models": []}), mimetype='application/json')


@app.after_request
def add_cors_headers(response):
    response.headers['Access-Control-Allow-Origin'] = '*'
    response.headers['Access-Control-Allow-Headers'] = 'Content-Type, Authorization'
    response.headers['Access-Control-Allow-Methods'] = 'POST, GET, OPTIONS'
    return response


def safe_yield(text_content):
    payload = {
        "message": {
            "role": "assistant",
            "content": text_content
        },
        "done": False
    }
    return json.dumps(payload).encode('utf-8') + b'\n'

@app.route('/api/chat', methods=['POST'])
def mock_chat():
    data = request.json or {}
    messages = data.get("messages", [])
    if not messages:
        return json.dumps({"message": {"role": "assistant", "content": "Нет сообщений."}})

    user_query = messages[-1]["content"]

    # --- ТЕРМОКОНТРОЛЬ ---
    current_cpu_load = get_cpu_load_macos()
    silent_mode = current_cpu_load > CPU_HEAVY_THRESHOLD

    print(f"\n========================================================")
    print(f"[📥 ВХОДЯЩИЙ ЗАПРОС ИЗ GOLAND] Загрузка CPU: {current_cpu_load:.1f}%")
    print(f"========================================================")

    query_vector = get_query_embedding(user_query)

    # Моментальная выборка из БД
    books_context = ""
    if query_vector and books_collection.count() > 0:
        book_results = books_collection.query(query_embeddings=[query_vector], n_results=2)
        if book_results and book_results.get('documents') and book_results['documents']:
            for doc in book_results['documents']:
                books_context = "\n---\n".join(doc)

    project_context = ""
    if query_vector and project_collection.count() > 0:
        project_results = project_collection.query(query_embeddings=[query_vector], n_results=4)
        if project_results and project_results.get('documents') and project_results['documents']:
            for doc in project_results['documents']:
                project_context = "\n---\n".join(doc)

    long_term_memory_context = get_relevant_chat_memory(user_query, query_vector) if query_vector else ""

    # Системный промпт
    system_instruction = (
        "Ты — ведущий ИИ-архитектор систем на Go (Clean Architecture, SOLID, gRPC, DDD).\n"
        f"1. ТЕОРЕТИЧЕСКИЙ КОНТЕКСТ ИЗ ВАШИХ КНИГ:\n{books_context}\n\n"
        f"2. ТЕКУЩИЙ ИСХОДНЫЙ КОД ПРОЕКТА ИЗ ВЕКТОРНОЙ БД:\n{project_context}\n\n"
        f" Ранее обсуждались данные действия \n{long_term_memory_context}\n\n"
        "🚨 СТРОЖАЙШИЕ ПРАВИЛА ЗАЩИТЫ ОТ ГАЛЛЮЦИНАЦИЙ:\n"
        "1. Запрещено выдумывать пути к файлам, названия пакетов, имена функций или структур, которых нет в предоставленном выше коде.\n"
        "2. Если пользователь спрашивает про компонент, а в векторной БД проекта его исходный код отсутствует — прямо ответь, что компонент не найден в базе проекта.\n"
        "3. Выводи только точечные, рабочие Go-диффы изменений."
    )

    agent_messages = [{"role": "system", "content": system_instruction}]
    for msg in messages[:-1]: agent_messages.append(msg)
    agent_messages.append({"role": "user", "content": user_query})

    # --- ЗАЩИТА ОТ ТАЙМАУТОВ IDE ЧЕРЕЗ СТРИМИНГ-ГЕНЕРАТОР ---
    def generate():
        nonlocal agent_messages
        full_answer = ""

        try:
            if silent_mode:
                final_messages = agent_messages
            else:
                # 📢 ОТПРАВЛЯЕМ ИМПУЛЬС В IDE
                pulse_msg = {"message": {"role": "assistant",
                                         "content": "[🔍 Шаг 0/4: Сверка с векторной базой данных знаний...]\n\n"},
                             "done": False}
                yield json.dumps(pulse_msg).encode('utf-8') + b'\n'

                # --- ШАГ 0: ПРЕДВАРИТЕЛЬНАЯ ВЕРИФИКАЦИЯ БАЗЫ ---
                verify_prompt = (
                    f"Проверь предоставленный 'project_context' и исходный вопрос пользователя: '{user_query}'.\n"
                    "Ответь строго в формате JSON с ключами:\n"
                    "1. 'status': строка 'OK' (если в контексте есть реальный код нужного компонента/файла) или 'INCOMPLETE' (если нужного кода в базе нет).\n"
                    "2. 'reason': короткое объяснение, почему данных хватает или чего именно не хватает.\n"
                    "Выведи ТОЛЬКО чистый JSON, без лишнего текста."
                )
                verify_messages = agent_messages + [{"role": "user", "content": verify_prompt}]
                verify_res = call_ollama(MODEL_CREATOR, verify_messages)

                # Парсим вердикт верификации
                is_data_ok = True
                verification_reason = ""
                try:
                    # Очищаем ответ от возможных markdown-тегов ```json
                    clean_json = re.sub(r'```json\s*|```', '', verify_res).strip()
                    verify_data = json.loads(clean_json)
                    if verify_data.get("status") == "INCOMPLETE":
                        is_data_ok = False
                        verification_reason = verify_data.get("reason", "Недостаточно данных в базе.")
                except:
                    # Если модель не смогла вернуть JSON, по умолчанию считаем, что всё ок, чтобы не ломать поток
                    pass

                # Если верификация провалилась — пробуем живой поиск по проекту
                if not is_data_ok:
                    print(f"[⚠️ ВЕРИФИКАЦИЯ ПРОВАЛЕНА]: {verification_reason}")
                    print(f"[🔄] Попытка живого поиска по проекту вместо прерывания...")

                    project_context_from_files = get_go_project_context(GO_PROJECT_PATH, user_query)

                    # ИСПРАВЛЕНО: Безопасный yield
                    yield safe_yield(project_context_from_files + "\n")

                    # Пересобираем системный промпт с живым контекстом
                    system_instruction_with_live = (
                        "Ты — ведущий ИИ-архитектор систем на Go (Clean Architecture, SOLID, gRPC, DDD).\n"
                        f"1. ТЕОРЕТИЧЕСКИЙ КОНТЕКСТ ИЗ ВАШИХ КНИГ:\n{books_context}\n\n"
                        f"2. ЖИВОЙ КОНТЕКСТ ИЗ ФАЙЛОВ ПРОЕКТА (поиск без векторной БД):\n{project_context_from_files}\n\n"
                        "🚨 СТРОЖАЙШИЕ ПРАВИЛА ЗАЩИТЫ ОТ ГАЛЛЮЦИНАЦИЙ:\n"
                        "1. Запрещено выдумывать пути к файлам, названия пакетов, имена функций или структур, которых нет в предоставленном выше коде.\n"
                        "2. Если пользователь спрашивает про компонент, а в контексте его исходный код отсутствует — прямо ответь, что компонент не найден.\n"
                        "3. Выводи только точечные, рабочие Go-диффы изменений."
                    )

                    agent_messages_live = [{"role": "system", "content": system_instruction_with_live}]
                    for msg in messages[:-1]: agent_messages_live.append(msg)
                    agent_messages_live.append({"role": "user", "content": user_query})

                    # Стримим ответ с живым контекстом
                    final_prompt_live = (
                        f"Перед тобой исходный вопрос пользователя: '{user_query}'.\n"
                        f"У тебя есть контекст из книг и живой контекст из файлов проекта.\n\n"
                        "🚨 СТРОГАЯ СТРУКТУРА ОТВЕТА:\n"
                        f"1. РАЗБОР ПО ЗАПРОСУ: Ответь на вопрос, основываясь на контексте.\n"
                        "2. ИСПРАВЛЕННЫЙ GO-КОД: Выведи точный, готовый к интеграции Go-код.\n\n"
                        "Выведи только результат."
                    )
                    final_messages = agent_messages_live + [
                        {"role": "user", "content": final_prompt_live}]

                    yield safe_yield("\n[🧠 Шаг 3/3] Трансляция ответа из живого контекста проекта...\n")

                    # В экстренном режиме поиска оставляем MODEL_CRITIC
                    response = requests.post(
                        OLLAMA_CHAT_URL,
                        json={
                            "model": MODEL_CRITIC,
                            "messages": final_messages,
                            "stream": True,
                            "options": {"num_ctx": 16384}
                        },
                        stream=True,
                        timeout=120
                    )

                    if response.status_code != 200:
                        err_text = f"Ollama вернула ошибку {response.status_code}"
                        yield json.dumps({"message": {"role": "assistant", "content": err_text}, "done": True}).encode(
                            'utf-8') + b'\n'
                        return

                    for line in response.iter_lines():
                        if line:
                            yield line + b'\n'
                            try:
                                chunk_json = json.loads(line.decode('utf-8'))
                                if "message" in chunk_json and "content" in chunk_json["message"]:
                                    content = chunk_json["message"]["content"]
                                    print(content, end="", flush=True)
                                    full_answer += content
                            except:
                                pass

                    print("\n========================================================")

                    if query_vector and full_answer:
                        save_to_long_term_memory(user_query, full_answer, query_vector)
                        print("[📢] Ответ сохранен в долгосрочную память.")

                    return

                # --- ЕСЛИ ВСЁ ОК -> ЗАПУСКАЕМ НАШ СТАНДАРТНЫЙ КОНСОРЦИУМ ---
                print("[🎯 ВЕРИФИКАЦИЯ СУПЕР]: Данных в ChromaDB достаточно. Запускаю консорциум...")

                pulse_msg1 = {"message": {"role": "assistant",
                                          "content": "[🧠 Шаг 1/4: Проектирование архитектурного черновика ответа...]\n\n"},
                              "done": False}
                yield json.dumps(pulse_msg1).encode('utf-8') + b'\n'

                # Выполняем Шаг 1 (Черновик)
                draft_messages = agent_messages + [
                    {"role": "user", "content": "Напиши подробный технический черновик ответа."}]
                draft_content = call_ollama(MODEL_CRITIC,draft_messages)

                pulse_msg2 = {"message": {"role": "assistant",
                                          "content": "[🕵️ Шаг 2/4: Глубокий аудит ревьюера на утечки памяти и SOLID...]\n\n"},
                              "done": False}
                yield json.dumps(pulse_msg2).encode('utf-8') + b'\n'

                # Выполняем Шаг 2 (Критика)
                critic_prompt = (
                    f"Вот черновик ответа:\n{draft_content}\n\n"
                    "Проведи аудит. Проверь на УТЕЧКИ ПАМЯТИ (незакрытые body, каналы, горутины). Исправь синтаксические ошибки Go. Выпиши замечания."
                )
                critic_messages = agent_messages + [{"role": "assistant", "content": draft_content},
                                                    {"role": "user", "content": critic_prompt}]
                critic_feedback = call_ollama(MODEL_CRITIC, critic_messages)

                # --- ШАГ 2.5: КОНСУЛЬТАЦИЯ С АНАЛОГОМ (с проверкой контекста) ---
                context_for_consultation = (
                    f"Книги (выжимка): {books_context[:500]}\n"
                    f"Проект (выжимка): {project_context[:800]}\n"
                    f"Аудит ревьюера: {critic_feedback[:1500]}"  # Ограничиваем критику
                )

                pulse_msg_consult = {"message": {"role": "assistant",
                                                "content": "[🤝 Шаг 2.5/4: Консультация с аналогом...]\n\n"},
                                     "done": False}
                yield json.dumps(pulse_msg_consult).encode('utf-8') + b'\n'

                advisor_feedback = ""
                if not should_skip_consultation(agent_messages + [{"role": "assistant", "content": draft_content},
                                                                  {"role": "user", "content": critic_prompt}]):
                    yield safe_yield("[💬] Архитектор запрашивает совет у OpenCode...\n")
                    advisor_feedback = consult_advisor(draft_content, user_query, context_for_consultation)
                    yield safe_yield(f"[✅] Совет получен: {len(advisor_feedback)} символов\n")
                else:
                    advisor_feedback = "КОНСУЛЬТАЦИЯ НЕ ПРОИЗВЕДЕНА — контекст близок к пределу."
                    yield safe_yield("[⏭️] Консультация пропущена из-за лимита контекста\n")


                # Выполняем Шаг 3 (Финальная сборка)
                final_prompt = (
                    f"Перед тобой исходный вопрос пользователя: '{user_query}'.\n"
                    "Также у тебя есть черновик решения, жесткие замечания критика-ревьюера по коду "
                    "и мнение советника-архитектора.\n\n"
                    "🚨 ТВОЯ ЗАДАЧА — СФОРМИРОВАТЬ ИТОГОВЫЙ СТРУКТУРИРОВАННЫЙ ВЕРДИКТ СТРОГО НА РУССКОМ ЯЗЫКЕ.\n"
                    "Запрещено использовать английский язык для текстовых описаний, пояснений и заголовков. "
                    "Весь текст, кроме названий Go-структур, пакетов и самого Go-кода, должен быть написан на чистом, техническом русском языке.\n\n"
                    "Запрещено пересказывать сам процесс работы консорциума. "
                    "Отвечай так, будто ты сразу выдаешь финальное, глубоко проверенное решение. "
                    "Используй исключительно реальный контекст кода вашего Go-проекта и книг.\n\n"
                    "🚨 СТРОГАЯ СТРУКТУРА ФИНАЛЬНОГО ОТВЕТА (ПИШИ ЗАГОЛОВКИ И ТЕКСТ НА РУССКОМ):\n"
                    f"1. РАЗБОР ПО ЗАПРОСУ: Четко и по существу ответь на исходный вопрос пользователя ('{user_query}'), основываясь на коде из базы.\n"
                    "2. СКРЫТЫЕ ОШИБКИ И УТЕЧКИ ПАМЯТИ: С опорой на аудит ревьюера выпиши только реальные баги, race conditions, утечки горутин/каналов или памяти, которые присутствуют в коде проекта. Если их нет — напиши, что критических утечек не обнаружено.\n"
                    "3. РЕКОМЕНДАЦИИ СОВЕТНИКА: Кратко опиши ключевые замечания советника, если консультация была проведена.\n"
                    "4. ИСПРАВЛЕННЫЙ GO-КОД: Выведи точный, готовый к интеграции в IDE Go-код (дифф или инкремент логики), исправляющий уязвимости или расширяющий компонент. Пиши рабочий код без текстовых заглушек.\n\n"
                    "Выведи только результат по этой структуре."
                )

                # 2. Упаковываем все предыдущие шаги в единый контекст для финальной модели
                final_messages = agent_messages + [
                    {"role": "assistant", "content": draft_content},  # Черновик от Codestral
                    {"role": "user", "content": critic_prompt},  # Запрос на критику
                    {"role": "assistant", "content": critic_feedback},  # Ответ критика от Codestral
                    {"role": "user", "content": "Комментарии советника:\n" + advisor_feedback},  # Совет от OpenCode
                    {"role": "assistant", "content": "Принял замечания советника."},
                    {"role": "user", "content": final_prompt}  # Тот самый финальный промпт
                ]

            yield safe_yield("\n[🧠 Шаг 3/3] Начинаю сборку итогового ответа...\n")

            # Финальный стриминг ответа в IDE
            active_model = MODEL_CRITIC if not silent_mode else LLM_MODEL
            yield json.dumps(
                f"==========================ACTIVE MODEL: {active_model}============================== "
            ).encode('utf-8') + b'\n'


            final_synthesis_model = LLM_MODEL
            yield safe_yield(
                f"\n========================== ФИНАЛЬНЫЙ АНАЛИЗАТОР: {final_synthesis_model} ==============================\n\n")

            # 4. Отправляем финальный запрос
            response = requests.post(
                OLLAMA_CHAT_URL,
                json={
                    "model": final_synthesis_model,
                    "messages": final_messages,
                    "stream": True,
                    "options": {"num_ctx": 16384}
                },
                stream=True,
                timeout=120
            )
            pulse_msg3 = {"message": {"role": "assistant",
                         "content": "[🕵️ АРХИТЕКТОР: Анализ структуры ответа и корректность]\n\n"},
             "done": False}
            yield json.dumps(pulse_msg3).encode('utf-8') + b'\n'

            if response.status_code != 200:
                err_text = f"Ollama вернула ошибку {response.status_code}"
                yield json.dumps({"message": {"role": "assistant", "content": err_text}, "done": True}).encode(
                    'utf-8') + b'\n'
                return

            for line in response.iter_lines():
                if line:
                    yield line + b'\n'

                    try:
                        chunk_json = json.loads(line.decode('utf-8'))
                        if "message" in chunk_json and "content" in chunk_json["message"]:
                            content = chunk_json["message"]["content"]
                            print(content, end="", flush=True)
                            full_answer += content
                    except:
                        pass

                if query_vector and full_answer:
                        save_to_long_term_memory(user_query, full_answer, query_vector)
                        print("\n[📢] Архитектурный ответ успешно сохранен в долгосрочную память.")

        except Exception as e:
            print(f"\n[!] Критическая ошибка при стриминге консорциума: {e}")
            err_payload = {"message": {"role": "assistant", "content": f"\n[Ошибка консорциума]: {str(e)}"},
                                       "done": True}
            yield json.dumps(err_payload).encode('utf-8') + b'\n'

    return Response(generate(), mimetype='application/x-ndjson')


@app.route('/api/rebuild-db', methods=['POST'])
def manual_rebuild_db():
    """Ручной запуск пересборки БД"""
    thread = threading.Thread(target=rebuild_vector_db)
    thread.start()
    return {"status": "rebuilding", "message": "Пересборка БД запущена в фоне"}

if __name__ == "__main__":
    crawler = threading.Thread(target=async_knowledge_crawler, daemon=True)
    crawler.start()

    # Запускаем наш прокси-сервер на порту 5001
    print("[🚀] Сервер Суперагента запущен на http://localhost:5001")
    print("Он слушает запросы от плагина Multi и перенаправляет их в Ollama с полной предобработкой.")
    app.run(host='localhost', port=5001, use_reloader=False)