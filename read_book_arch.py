import hashlib
import json
import os
import glob
import time

from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

from quadro.quadro import ProjectContextUpdater

GO_PROJECT_PATH = "/Users/pugachev.mihail9/Desktop/work_app/calculator-v2"
TARGET_FOLDER = "./books"
HASH_CACHE_FILE = "./go_project_hashes.json"

updater = ProjectContextUpdater(qdrant_host="localhost", qdrant_port=6333)


def get_file_hash(file_path: str) -> str:
    hasher = hashlib.sha256()
    try:
        with open(file_path, "rb") as f:
            hasher.update(f.read())
        return hasher.hexdigest()
    except Exception:
        return ""


def sync_go_project() -> None:
    """Синхронизирует AST-структуру Go-проекта в коллекцию go_project_context."""
    if not os.path.exists(GO_PROJECT_PATH):
        print("[!] Ошибка: Путь к проекту не найден.")
        return

    hash_cache = {}
    if os.path.exists(HASH_CACHE_FILE) and os.stat(HASH_CACHE_FILE).st_size > 0:
        with open(HASH_CACHE_FILE, "r", encoding="utf-8") as f:
            hash_cache = json.load(f)

    has_changes = False

    for root, dirs, files in os.walk(GO_PROJECT_PATH):
        dirs[:] = [d for d in dirs if d not in [".git", "vendor", "mocks", ".venv"]]
        for file in files:
            if not file.endswith(".go"):
                continue
            full_path = os.path.join(root, file)
            rel_path = os.path.relpath(full_path, GO_PROJECT_PATH)

            current_hash = get_file_hash(full_path)
            old_hash = hash_cache.get(rel_path, "")

            if current_hash == old_hash:
                continue

            print(f"[🔥 ИЗМЕНЕНИЕ] AST-парсинг файла: {rel_path}")
            has_changes = True

            try:
                updater.qdrant.delete(
                    collection_name="go_project_context",
                    points_selector={
                        "filter": {"must": [{"key": "file_path", "match": {"value": full_path}}]}
                    },
                )
            except Exception:
                pass

            updater.sync_file(full_path)
            hash_cache[rel_path] = current_hash

    if has_changes:
        with open(HASH_CACHE_FILE, "w", encoding="utf-8") as f:
            json.dump(hash_cache, f, indent=2, ensure_ascii=False)
        print("[💾] Qdrant: коллекция go_project_context обновлена.")


def sync_books() -> None:
    """Синхронизирует PDF-книги в коллекцию books_collection."""
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


def rebuild_vector_db() -> None:
    """Полностью очищает и пересобирает Qdrant для кода, книг и архитектурного профиля."""
    print("[🔄] Полная пересборка Qdrant...")
    try:
        updater.qdrant.delete_collection(collection_name="go_project_context")
        updater.qdrant.delete_collection(collection_name="project_arch_profile")
    except Exception as e:
        print(f"[!] Ошибка удаления коллекций: {e}")

    updater._ensure_collections()

    if os.path.exists(HASH_CACHE_FILE):
        os.remove(HASH_CACHE_FILE)

    sync_go_project()
    sync_books()
    build_project_arch_profile()
    print("[✅] Qdrant пересобран: код, книги, архитектура.")


def build_project_arch_profile() -> None:
    """Строит архитектурный профиль проекта и сохраняет его в коллекцию project_arch_profile.

    Профиль содержит по модулю:
    - module_name
    - layer (ui / transport / application / domain / infra)
    - responsibility
    - depends_on / used_by
    - public_entry_points
    - forbidden_patterns
    - change_recipe
    """
    print("[🏗] Сборка архитектурного профиля проекта...")
    try:
        updater.build_arch_profile(collection_name="project_arch_profile")
        print("[🏁] Архитектурный профиль обновлён (project_arch_profile).")
    except Exception as e:
        print(f"[!] Ошибка при обновлении архитектурного профиля: {e}")

class GoFileHandler(FileSystemEventHandler):
    def __init__(self, project_path: str, updater: ProjectContextUpdater):
        self.project_path = os.path.abspath(project_path)
        self.updater = updater

    def on_modified(self, event):
        self._handle(event)

    def on_created(self, event):
        self._handle(event)

    def _handle(self, event):
        if event.is_directory:
            return
        if not event.src_path.endswith(".go"):
            return
        if not event.src_path.startswith(self.project_path):
            return

        print(f"[🔁] Изменён Go-файл: {event.src_path}")
        # Перепарсим только этот файл
        try:
            self.updater.qdrant.delete(
                collection_name="go_project_context",
                points_selector={
                    "filter": {"must": [{"key": "file_path", "match": {"value": event.src_path}}]}
                },
            )
        except Exception:
            pass
        self.updater.sync_file(event.src_path)
        # Опционально — обновить архитектурный профиль (можно дебаунсить по времени)
        self.updater.build_arch_profile("project_arch_profile")

def async_knowledge_crawler():
    """Фоновый демон: следит за изменениями .go-файлов, обновляет Qdrant и архитектурный профиль."""
    print("[🕷] Watchdog-краулер запущен (Go-файлы + архитектура).")
    updater._ensure_collections()

    event_handler = GoFileHandler(GO_PROJECT_PATH, updater)
    observer = Observer()
    observer.schedule(event_handler, path=GO_PROJECT_PATH, recursive=True)
    observer.start()

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()
    observer.join()



if __name__ == "__main__":
    rebuild_vector_db()
