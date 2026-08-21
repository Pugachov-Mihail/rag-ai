import os
import chromadb
import requests
import json
import time

from qdrant_client import QdrantClient

# 1. Конфигурация
CHROMA_PATH = "./my_library_db"
EMBED_MODEL = "nomic-embed-text"
LLM_MODEL = "qwen2.5-coder:14b"
OLLAMA_CHAT_URL = "http://localhost:11434/api/chat"
OLLAMA_EMBED_URL = "http://localhost:11434/api/embeddings"

GO_PROJECT_PATH = "/Users/pugachev.mihail9/Desktop/work_app/calculator-v2"  # <-- Укажите свой путь
CRITIC_MODEL = "gemma4:26b"

# Qdrant для кода и книг (как обсуждалось в RAG архитектуре)
qdrant_client = QdrantClient("localhost", port=6333)

# Подключение к ChromaDB
chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)
# Коллекция 1: Ваши книги (статичные знания)
books_collection = chroma_client.get_collection(name="books_collection")
# Коллекция 2: Динамическая память чата (создается автоматически)
memory_collection = chroma_client.get_or_create_collection(name="chat_memory")

# Краткосрочная память для удержания текущей нити разговора (последние 5 реплик)
short_term_history = []


def get_query_embedding(text):
    try:
        response = requests.post(OLLAMA_EMBED_URL, json={"model": EMBED_MODEL, "prompt": text})
        return response.json().get("embedding") or response.json().get("embeddings")
    except:
        return None


def get_qdrant_context(query_vector, collection_name, limit=3):
    """Ищет релевантные куски кода или книг в Qdrant"""
    try:
        results = qdrant_client.search(
            collection_name=collection_name,
            query_vector=query_vector,
            limit=limit
        )

        context_blocks = []
        for r in results:
            meta = r.payload

            # Если ищем по коду, добавляем связи графа зависимостей
            if collection_name == "go_project_context":
                component = meta.get('component', 'unknown')
                dependencies = meta.get('related_modules', [])
                filepath = meta.get('file_path', 'unknown')

                block = (
                    f"Файл: {filepath} | Компонент: {component}\n"
                    f"Вызывает зависимости: {dependencies}\n"
                    f"Код:\n{meta.get('content')}"
                )
                context_blocks.append(block)
            else:
                # Если ищем по книгам
                block = f"Источник: {meta.get('source')}\n{meta.get('content')}"
                context_blocks.append(block)

        return "\n---\n".join(context_blocks)
    except Exception as e:
        print(f"[!] Ошибка поиска в Qdrant ({collection_name}): {e}")
        return ""

def get_relevant_chat_memory(user_query, query_vector):
    """Ищет прошлые разговоры, повышает приоритет частым темам и занижает редким"""
    if memory_collection.count() == 0:
        return ""

    # Ищем до 5 похожих прошлых диалогов
    results = memory_collection.query(query_embeddings=[query_vector], n_results=5)

    memory_context = []
    current_time = time.time()

    for doc, meta, doc_id in zip(results['documents'][0], results['metadatas'][0], results['ids'][0]):
        score = meta['score']
        timestamp = meta['timestamp']

        # Эвристика затухания: уменьшаем важность на 5% за каждый день простоя (86400 сек)
        days_passed = (current_time - timestamp) / 86400
        decayed_score = score * (0.95 ** days_passed)

        # Если контекст всплывает снова, прокачиваем его базовый score в БД для следующих раз
        new_score = score + 0.5  # Повышаем приоритет частой теме
        memory_collection.update(ids=[doc_id], metadatas=[{"score": new_score, "timestamp": current_time}])

        # Если тема совсем устарела и забылась (score упал ниже порога), игнорируем её
        if decayed_score > 0.4:
            memory_context.append(f"[Из прошлого опыта (Приоритет: {decayed_score:.2f})]:\n{doc}")

    return "\n---\n".join(memory_context)


def save_to_long_term_memory(user_query, ai_answer, query_vector):
    """Сохраняет пару вопрос-ответ в базу долгосрочной памяти"""
    memory_id = f"mem_{int(time.time() * 1000)}"
    combined_text = f"Вопрос пользователя: {user_query}\nОтвет ИИ: {ai_answer}"

    memory_collection.add(
        embeddings=[query_vector],
        documents=[combined_text],
        ids=[memory_id],
        metadatas=[{"score": 1.0, "timestamp": time.time()}]  # Изначальный приоритет = 1.0
    )


def call_ollama(messages, stream=False, model_name=LLM_MODEL):
    try:
        response = requests.post(OLLAMA_CHAT_URL, json={"model": model_name, "messages": messages, "stream": stream},
                                 stream=stream)
        return response
    except:
        return None


def ask_super_agent(user_query):
    global short_term_history
    query_vector = get_query_embedding(user_query)
    if not query_vector: return

    print("[🔍] Шаг 1: Поиск в книгах и извлечение памяти...")
    # Извлекаем знания из книг через Qdrant
    books_context = get_qdrant_context(query_vector, "books_collection", limit=3)
    long_term_memory_context = get_relevant_chat_memory(user_query, query_vector)

    print("[💻] Шаг 2: Семантический поиск по AST-структуре Go-проекта...")
    # Ищем только те функции и структуры графа маршрутизации, которые нужны для ответа
    project_context = get_qdrant_context(query_vector, "go_project_context", limit=5)

    system_instruction = (
        "Ты — ведущий ИИ-архитектор. Реши задачу пользователя.\n"
        f"У тебя есть данные из трех источников:\n"
        f"1. ТЕОРЕТИЧЕСКИЙ КОНТЕКСТ ИЗ КНИГ:\n{books_context}\n\n"
        f"2. ПРОШЛЫЙ ОПЫТ (ОТФИЛЬТРОВАН):\n{long_term_memory_context}\n\n"
        f"3. ПРАКТИЧЕСКИЙ КОНТЕКСТ ТЕКУЩЕГО КОДА:\n{project_context}\n\n"
        "Отвечай технически точно. Строго соблюдай архитектуру проекта и используй указанные зависимости."
    )

    messages = [{"role": "system", "content": system_instruction}]
    # Добавляем краткосрочную нить текущей беседы
    for msg in short_term_history: messages.append(msg)
    messages.append({"role": "user", "content": user_query})

    print("[🧠] Шаг 3: Самопроверка и аудит черновика ответа (Gemma 4 26B)...")

    # 3.1 Скрытый шаг создания черновика (быстрая модель)
    draft_messages = messages + [{"role": "user", "content": "Напиши подробный технический черновик ответа."}]
    res = call_ollama(draft_messages, model_name=LLM_MODEL)
    if not res: return
    draft_content = res.json()["message"]["content"]

    # 3.2 Скрытый шаг жесткой критики (Умная модель)
    critic_prompt = (
        f"Вот предварительный черновик ответа:\n{draft_content}\n\n"
        "Проведи жесткий архитектурный аудит.\n"
        "1. Зависимости: Проверь, не нарушает ли код существующий граф зависимостей. Убедись, что используются функции, перечисленные в related_modules, и не изобретаются дубликаты.\n"
        "2. Контекст: Соответствует ли предложенная логика правилам из книг и истории диалога?\n"
        "3. Качество: Исправь синтаксические ошибки Go, горутины, утечки памяти и логические нестыковки.\n"
        "Выпиши конкретные замечания."
    )
    critic_messages = messages + [{"role": "assistant", "content": draft_content},
                                  {"role": "user", "content": critic_prompt}]

    # Вызываем Gemma 4 26B для критики
    res = call_ollama(critic_messages, model_name=CRITIC_MODEL)
    if not res: return
    critic_feedback = res.json()["message"]["content"]

    # 3.3 Финальная сборка идеального ответа (Gemma 4 26B)
    final_prompt = "Используя черновик и свои замечания аудитора, сформируй финальный идеальный ответ. Выведи только готовое решение и пояснения к нему."
    final_messages = messages + [
        {"role": "assistant", "content": draft_content},
        {"role": "user", "content": critic_prompt},
        {"role": "assistant", "content": critic_feedback},
        {"role": "user", "content": final_prompt}
    ]

    print("\n[🎉 Ответ проверен по Книгам, Коду и Истории]")
    print("ИИ: ", end="", flush=True)

    # Финальный стриминг через Gemma
    res = call_ollama(final_messages, stream=True, model_name=CRITIC_MODEL)
    full_answer = ""
    for line in res.iter_lines():
        if line:
            chunk_json = json.loads(line.decode('utf-8'))
            if "message" in chunk_json and "content" in chunk_json["message"]:
                content = chunk_json["message"]["content"]
                print(content, end="", flush=True)
                full_answer += content
    print()

    # Сохраняем в краткосрочную память текущей сессии
    short_term_history.append({"role": "user", "content": user_query})
    short_term_history.append({"role": "assistant", "content": full_answer})
    if len(short_term_history) > 10: short_term_history = short_term_history[-10:]

    # Сохраняем в векторную базу данных для долгосрочной памяти (с базовым весом 1.0)
    save_to_long_term_memory(user_query, full_answer, query_vector)


if __name__ == "__main__":
    print(f"\n[🚀] Агент с адаптивной памятью запущен.")
    print("Он запоминает частые темы, забывает редкие, сверяется с кодом проекта и книгами.")
    while True:
        try:
            user_input = input("\nВы: ").strip()
            if user_input.lower() in ['exit', 'выход']: break
            if user_input: ask_super_agent(user_input)
        except KeyboardInterrupt:
            break