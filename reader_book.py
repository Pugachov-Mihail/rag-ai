import hashlib
import json
import os
import glob
import time

import fitz  # PyMuPDF
import chromadb
import requests

# 1. Настройки конфигурации
OLLAMA_URL = "http://localhost:11434/api/embeddings"
EMBED_MODEL = "nomic-embed-text"  # <-- ПОМЕНЯЛИ МОДЕЛЬ НА СПЕЦИАЛЬНУЮ
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 150

# Имя файла, куда будут сохраняться хеши файлов вашего проекта
HASH_CACHE_FILE = "./go_project_hashes.json"
GO_PROJECT_PATH = "/Users/pugachev.mihail9/Desktop/work_app/calculator-v2"

# 2. Инициализация векторной БД Chroma
chroma_client = chromadb.PersistentClient(path="./my_library_db")
collection = chroma_client.get_or_create_collection(name="books_collection")
project_collection = chroma_client.get_or_create_collection(name="project_collection")

# Замените верхнюю часть настроек в вашем файле fast_rag_pipeline.py на эту:


# А функцию получения эмбеддинга обновите вот так (с исправленным ключом):
def get_ollama_embedding(text):
    """Быстрое получение вектора от локальной Ollama"""
    try:
        response = requests.post(OLLAMA_URL, json={"model": EMBED_MODEL, "prompt": text})
        json_data = response.json()

        # Ollama в зависимости от версии может возвращать ключ 'embedding' или список 'embeddings'
        if "embedding" in json_data:
            return json_data["embedding"]
        elif "embeddings" in json_data:
            return json_data["embeddings"][0]  # Берем первый вектор из батча
        else:
            print(f"[!] Структура ответа Ollama изменилась: {json_data}")
            return None
    except Exception as e:
        print(f"[!] Ошибка связи с Ollama: {e}")
        return None


def parse_and_index_pdf(pdf_path):
    filename = os.path.basename(pdf_path)
    print(f"\n[*] Начинаю обработку: {filename}")

    try:
        doc = fitz.open(pdf_path)  # Сверхбыстрое чтение C++ движком
        full_text = ""
        for page in doc:
            full_text += page.get_text() + "\n"
        doc.close()  # Обязательно закрываем файл, чтобы macOS разрешила его удалить

        # Если книга оказалась сканом (картинкой) и текста нет
        if not full_text.strip():
            print(f"[!] Внимание: В {filename} не найден текстовый слой. Пропускаю.")
            return False

        # Нарезка на куски
        chunks = []
        start = 0
        while start < len(full_text):
            end = start + CHUNK_SIZE
            chunks.append(full_text[start:end])
            start += CHUNK_SIZE - CHUNK_OVERLAP

        print(f"[+] Книга разбита на {len(chunks)} кусков. Векторизация...")

        # Заливка в ChromaDB
        for i, chunk in enumerate(chunks):
            vector = get_ollama_embedding(chunk)
            if vector is None:
                return False  # Если упала Ollama, останавливаем процесс

            chunk_id = f"{filename}_chunk_{i}"
            collection.add(
                embeddings=[vector],
                documents=[chunk],
                ids=[chunk_id],
                metadatas=[{"source": filename, "chunk_index": i}]
            )

        print(f"[SUCCESS] {filename} полностью добавлена в ИИ-память!")
        return True  # Сигнал, что всё прошло успешно и файл можно удалить

    except Exception as e:
        print(f"[!] Ошибка при обработке файла {filename}: {e}")
        return False


def process_folder(folder_path):
    """Ищет все PDF в папке, обрабатывает их и удаляет оригиналы"""
    if not os.path.exists(folder_path):
        print(f"[!] Ошибка: Папка {folder_path} не существует!")
        return

    # Находит все .pdf файлы в указанной папке (включая вложенные подпапки)
    pdf_files = glob.glob(os.path.join(folder_path, "**", "*.pdf"), recursive=True)

    total_files = len(pdf_files)
    print(f"[🏠] Найдено файлов для обработки: {total_files}")

    if total_files == 0:
        print("[*] Обрабатывать нечего. Папка пуста.")
        return

    for idx, pdf_path in enumerate(pdf_files, 1):
        print(f"\nProgress: [{idx}/{total_files}]")

        # Запускаем парсинг
        success = parse_and_index_pdf(pdf_path)

        # Если всё записалось в базу без ошибок — удаляем оригинал книги
        if success:
            try:
                os.remove(pdf_path)
                print(f"[🗑] Оригинальный файл удален: {os.path.basename(pdf_path)}")
            except Exception as e:
                print(f"[!] Не удалось удалить файл {pdf_path}: {e}")
        else:
            print(f"[⚠️] Файл {os.path.basename(pdf_path)} НЕ удален из-за ошибки в процессе.")

    print("\n[🎉] Все доступные книги успешно перенесены в векторную память!")

def get_file_hash(file_path):
    hasher = hashlib.sha256()
    try:
        with open(file_path, 'rb') as f:
            hasher.update(f.read())
        return hasher.hexdigest()
    except: return ""

def sync_go_project():
    print(f"[💻] Анализ изменений в Go-проекте...")
    if not os.path.exists(GO_PROJECT_PATH):
        print("[!] Ошибка: Путь к проекту не найден.")
        return

    # Читаем старый кэш хешей
    hash_cache = {}
    if os.path.exists(HASH_CACHE_FILE):
        with open(HASH_CACHE_FILE, 'r', encoding='utf-8') as f:
            hash_cache = json.load(f) if os.stat(HASH_CACHE_FILE).st_size != 0 else {}

    has_changes = False

    for root, dirs, files in os.walk(GO_PROJECT_PATH):
        dirs[:] = [d for d in dirs if d not in ['.git', 'vendor', 'mocks', '.venv']]
        for file in files:
            if file.endswith('.go') or file == 'go.mod':
                full_path = os.path.join(root, file)
                rel_path = os.path.relpath(full_path, GO_PROJECT_PATH)

                current_hash = get_file_hash(full_path)
                old_hash = hash_cache.get(rel_path, "")

                # Если файл изменился или он новый
                if current_hash != old_hash:
                    print(f"[🔥 ИЗМЕНЕНИЕ] Файл обновлен: {rel_path}")
                    has_changes = True

                    try:
                        with open(full_path, 'r', encoding='utf-8') as f:
                            content = f.read()

                        # Удаляем старые векторы этого файла из БД, чтобы не дублировать
                        try:
                            project_collection.delete(where={"source": rel_path})
                        except:
                            pass

                        # Если файл слишком большой, разбиваем его на структуры/функции
                        chunks = [content[i:i + 2000] for i in range(0, len(content), 1700)]
                        for i, chunk in enumerate(chunks):
                            vector = get_ollama_embedding(chunk)
                            if vector:
                                project_collection.add(
                                    embeddings=[vector],
                                    documents=[chunk],
                                    ids=[f"code_{rel_path.replace('/', '_')}_{i}"],
                                    metadatas=[{"source": rel_path, "type": "code"}]
                                )

                        # Обновляем хеш в кэше
                        hash_cache[rel_path] = current_hash
                    except Exception as e:
                        print(f"[!] Ошибка индексации файла {rel_path}: {e}")

    if has_changes:
        with open(HASH_CACHE_FILE, 'w', encoding='utf-8') as f:
            json.dump(hash_cache, f, indent=2, ensure_ascii=False)
        print("[💾] База данных кода проекта успешно синхронизирована!")
    else:
        print("[🎯] Изменений в кодовой базе Go-проекта не обнаружено. Всё актуально.")


def check_vector_db_integrity():
    """Проверяет целостность векторной БД"""
    try:
        # Получаем количество записей в коллекции
        collection_count = project_collection.count()
        print(f"[🔍 Проверка БД] Количество записей в векторной БД: {collection_count}")

        # Проверяем, пустая ли коллекция
        if collection_count == 0:
            print("[⚠️ Проверка БД] Коллекция пуста! Требуется пересборка.")
            return False

        # Проверяем, есть ли записи с метаданными source
        all_docs = project_collection.get(include=["metadatas"])
        sources = set()
        for meta in all_docs.get("metadatas", []):
            if meta and "source" in meta:
                sources.add(meta["source"])

        print(f"[🔍 Проверка БД] Найдено {len(sources)} уникальных файлов в БД")
        return len(sources) > 0

    except Exception as e:
        print(f"[❌ Проверка БД] Ошибка при проверке целостности: {e}")
        return False


def rebuild_vector_db():
    """Полностью пересобирает векторную базу данных"""
    print("[🔄 Пересборка БД] Начинаю полную пересборку векторной базы...")

    try:
        # 1. Очищаем всю коллекцию
        print("[🔄 Пересборка БД] Очистка существующих данных...")
        all_ids = project_collection.get()["ids"]
        if all_ids:
            project_collection.delete(ids=all_ids)

        # 2. Сбрасываем кеш хэшей
        print("[🔄 Пересборка БД] Сброс кеша хэшей...")
        if os.path.exists(HASH_CACHE_FILE):
            os.remove(HASH_CACHE_FILE)

        # 3. Пересобираем из всех файлов Go-проекта
        print("[🔄 Пересборка БД] Индексация файлов Go-проекта...")
        if os.path.exists(GO_PROJECT_PATH):
            hash_cache = {}
            has_changes = False

            for root, dirs, files in os.walk(GO_PROJECT_PATH):
                dirs[:] = [d for d in dirs if d not in ['.git', 'vendor', 'mocks', '.venv', 'my_library_db']]

                for file in files:
                    if file.endswith('.go') or file == 'go.mod':
                        full_path = os.path.join(root, file)
                        rel_path = os.path.relpath(full_path, GO_PROJECT_PATH)

                        try:
                            with open(full_path, 'r', encoding='utf-8') as f:
                                content = f.read()

                            # Добавляем в БД
                            chunks = [content[i:i + 2000] for i in range(0, len(content), 1700)]
                            for i, chunk in enumerate(chunks):
                                vector = get_ollama_embedding(chunk)
                                if vector:
                                    project_collection.add(
                                        embeddings=[vector],
                                        documents=[chunk],
                                        ids=[f"code_{rel_path.replace('/', '_')}_{i}"],
                                        metadatas=[{"source": rel_path}]
                                    )

                            # Обновляем кеш
                            current_hash = get_file_hash(full_path)
                            hash_cache[rel_path] = current_hash
                            has_changes = True
                            print(f"[✅ Пересборка БД] Обработан файл: {rel_path}")

                        except Exception as e:
                            print(f"[❌ Пересборка БД] Ошибка при обработке {rel_path}: {e}")

            # Сохраняем новый кеш
            if has_changes:
                with open(HASH_CACHE_FILE, 'w', encoding='utf-8') as f:
                    json.dump(hash_cache, f, indent=2, ensure_ascii=False)

        # 4. Пересобираем книги (если есть)
        print("[🔄 Пересборка БД] Пересборка книг...")
        sync_books()

        print("[✅ Пересборка БД] Полная пересборка завершена успешно!")
        return True

    except Exception as e:
        print(f"[❌ Пересборка БД] Критическая ошибка: {e}")
        return False

# ========================================================
# ЗАПУСК КОНВЕЙЕРА
# Укажите путь к вашей папке с книгами (например, "/Users/имя/Downloads/Books")
TARGET_FOLDER = "./books"

def sync_books():
    # Создаем папку автоматически, если её нет, чтобы вы могли закинуть туда книги
    if not os.path.exists(TARGET_FOLDER):
        os.makedirs(TARGET_FOLDER)
        print(f"[*] Создана папка '{TARGET_FOLDER}'. Закиньте туда книги и запустите скрипт снова.")
    else:
        process_folder(TARGET_FOLDER)


def async_knowledge_crawler():
    """Фоновый поток, который непрерывно следит за изменениями кода и книг"""
    print("[🕷 Краулер] Фоновый мониторинг проекта и книг успешно запущен!")
    last_integrity_check = 0

    while True:
        try:
            # --- ЧАСТЬ А: ПРОВЕРКА КНИГ ---
            current_time = time.time()
            if current_time - last_integrity_check > 300:
                print("[🕷 Краулер] Запуск проверки целостности БД...")
                if not check_vector_db_integrity():
                    print("[🚨 Краулер] Обнаружены проблемы с БД! Выполняю пересборку...")
                    rebuild_vector_db()

                last_integrity_check = current_time

            if os.path.exists(TARGET_FOLDER):
                pdf_files = glob.glob(os.path.join(TARGET_FOLDER, "**", "*.pdf"), recursive=True)
                for pdf_path in pdf_files:
                    filename = os.path.basename(pdf_path)
                    print(f"[🕷 Краулер] Обнаружена новая книга: {filename}. Индексирую...")
                    try:
                        doc = fitz.open(pdf_path)
                        full_text = "".join([page.get_text() + "\n" for page in doc])
                        doc.close()
                        if full_text.strip():
                            chunks = [full_text[i:i + 1200] for i in range(0, len(full_text), 1000)]
                            for i, chunk in enumerate(chunks):
                                vector = get_ollama_embedding(chunk)
                                if vector:
                                    collection.add(embeddings=[vector], documents=[chunk],
                                                         ids=[f"{filename}_{i}"], metadatas=[{"source": filename}])
                        os.remove(pdf_path)
                        print(f"[🗑 Краулер] Книга обработана и удалена: {filename}")
                    except Exception as e:
                        print(f"[!] Ошибка краулера при парсинге книги {filename}: {e}")

            # --- ЧАСТЬ Б: КРАУЛЕР GO-ПРОЕКТА ---
            if os.path.exists(GO_PROJECT_PATH):
                hash_cache = {}
                if os.path.exists(HASH_CACHE_FILE) and os.stat(HASH_CACHE_FILE).st_size > 0:
                    with open(HASH_CACHE_FILE, 'r', encoding='utf-8') as f:
                        try:
                            hash_cache = json.load(f)
                        except:
                            hash_cache = {}

                has_changes = False
                for root, dirs, files in os.walk(GO_PROJECT_PATH):
                    dirs[:] = [d for d in dirs if d not in ['.git', 'vendor', 'mocks', '.venv', 'my_library_db']]
                    for file in files:
                        if file.endswith('.go') or file == 'go.mod':
                            full_path = os.path.join(root, file)
                            rel_path = os.path.relpath(full_path, GO_PROJECT_PATH)
                            current_hash = get_file_hash(full_path)
                            old_hash = hash_cache.get(rel_path, "")

                            if current_hash != old_hash:
                                print(f"[🕷 Краулер] [🔥 ИЗМЕНЕНИЕ КОДА] Файл изменен в IDE: {rel_path}")
                                has_changes = True
                                try:
                                    with open(full_path, 'r', encoding='utf-8') as f:
                                        content = f.read()
                                    try:
                                        project_collection.delete(where={"source": rel_path})
                                    except:
                                        pass

                                    chunks = [content[i:i + 2000] for i in range(0, len(content), 1700)]
                                    for i, chunk in enumerate(chunks):
                                        vector = get_ollama_embedding(chunk)
                                        if vector:
                                            project_collection.add(embeddings=[vector], documents=[chunk],
                                                                   ids=[f"code_{rel_path.replace('/', '_')}_{i}"],
                                                                   metadatas=[{"source": rel_path}])
                                    hash_cache[rel_path] = current_hash
                                except Exception as e:
                                    print(f"[!] Ошибка краулера на файле {rel_path}: {e}")

                if has_changes:
                    with open(HASH_CACHE_FILE, 'w', encoding='utf-8') as f:
                        json.dump(hash_cache, f, indent=2, ensure_ascii=False)
                    print("[💾 Краулер] Векторная БД проекта успешно обновлена в фоне!")

        except Exception as e:
            print(f"[!] Ошибка в цикле фонового краулера: {e}")

        time.sleep(3)  # Спим 3 секунды перед следующей асинхронной проверкой проекта