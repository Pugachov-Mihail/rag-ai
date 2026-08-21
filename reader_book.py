import hashlib
import json
import os
import glob
import time


from quadro.quadro import ProjectContextUpdater

# 1. Настройки конфигурации
GO_PROJECT_PATH = "/Users/pugachev.mihail9/Desktop/work_app/calculator-v2"
TARGET_FOLDER = "./books"
HASH_CACHE_FILE = "./go_project_hashes.json"

# Инициализируем Qdrant-апдейтер (без OpenAI)
updater = ProjectContextUpdater(qdrant_host="localhost", qdrant_port=6333)


def get_file_hash(file_path):
    """Сверхбыстро считает SHA-256 хеш файла"""
    hasher = hashlib.sha256()
    try:
        with open(file_path, 'rb') as f:
            hasher.update(f.read())
        return hasher.hexdigest()
    except:
        return ""


def sync_go_project():
    """Синхронизирует код проекта с Qdrant используя AST-парсер"""
    if not os.path.exists(GO_PROJECT_PATH):
        print("[!] Ошибка: Путь к проекту не найден.")
        return

    hash_cache = {}
    if os.path.exists(HASH_CACHE_FILE) and os.stat(HASH_CACHE_FILE).st_size > 0:
        with open(HASH_CACHE_FILE, 'r', encoding='utf-8') as f:
            hash_cache = json.load(f)

    has_changes = False

    for root, dirs, files in os.walk(GO_PROJECT_PATH):
        dirs[:] = [d for d in dirs if d not in ['.git', 'vendor', 'mocks', '.venv']]
        for file in files:
            if file.endswith('.go'):
                full_path = os.path.join(root, file)
                rel_path = os.path.relpath(full_path, GO_PROJECT_PATH)

                current_hash = get_file_hash(full_path)
                old_hash = hash_cache.get(rel_path, "")

                if current_hash != old_hash:
                    print(f"[🔥 ИЗМЕНЕНИЕ] Парсинг AST для файла: {rel_path}")
                    has_changes = True

                    # 1. Удаляем старые векторы этого файла из Qdrant
                    try:
                        updater.qdrant.delete(
                            collection_name="go_project_context",
                            points_selector={"filter": {"must": [{"key": "file_path", "match": {"value": full_path}}]}}
                        )
                    except Exception as e:
                        pass

                    # 2. Парсим структуру и загружаем в Qdrant
                    updater.sync_file(full_path)

                    # Обновляем кэш
                    hash_cache[rel_path] = current_hash

    if has_changes:
        with open(HASH_CACHE_FILE, 'w', encoding='utf-8') as f:
            json.dump(hash_cache, f, indent=2, ensure_ascii=False)
        print("[💾] База данных кода (Qdrant) успешно синхронизирована!")


def sync_books():
    """Синхронизирует PDF книги с Qdrant"""
    if not os.path.exists(TARGET_FOLDER):
        os.makedirs(TARGET_FOLDER)
        print(f"[*] Создана папка '{TARGET_FOLDER}'. Закиньте туда книги.")
        return

    pdf_files = glob.glob(os.path.join(TARGET_FOLDER, "**", "*.pdf"), recursive=True)
    if not pdf_files:
        return

    for pdf_path in pdf_files:
        success = updater.sync_book_to_qdrant(pdf_path)
        if success:
            try:
                os.remove(pdf_path)
                print(f"[🗑] Оригинал книги удален: {os.path.basename(pdf_path)}")
            except Exception as e:
                print(f"[!] Не удалось удалить файл {pdf_path}: {e}")


def rebuild_vector_db():
    """Полностью очищает и пересобирает Qdrant (вызывается из app.py)"""
    print("[🔄] Начинаю полную пересборку векторной базы Qdrant...")

    # Сбрасываем коллекции Qdrant
    try:
        updater.qdrant.delete_collection(collection_name="go_project_context")
        updater.qdrant.delete_collection(collection_name="books_collection")
    except Exception as e:
        print(f"[!] Ошибка при удалении коллекций: {e}")

    # Создаем заново с правильной размерностью
    updater._ensure_collections()

    # Сбрасываем кэш хэшей
    if os.path.exists(HASH_CACHE_FILE):
        os.remove(HASH_CACHE_FILE)

    # Запускаем парсинг
    sync_go_project()
    sync_books()
    print("[✅] Полная пересборка завершена! Теперь Qdrant содержит актуальные AST-деревья.")


def async_knowledge_crawler():
    """Фоновый поток, который непрерывно следит за изменениями кода и книг"""
    print("[🕷 Краулер] Фоновый мониторинг AST-зависимостей и книг запущен (Qdrant mode)!")

    # Гарантируем, что коллекции существуют при старте
    updater._ensure_collections()

    while True:
        try:
            sync_books()
            sync_go_project()
        except Exception as e:
            print(f"[!] Ошибка в цикле краулера: {e}")
        time.sleep(10)