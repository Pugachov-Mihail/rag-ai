import hashlib
import json
import os
import glob
import time

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
        updater.qdrant.delete_collection(collection_name="books_collection")
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


def async_knowledge_crawler() -> None:
    """Фоновый краулер: код, книги и архитектура проекта."""
    print("[🕷] Краулер Qdrant запущен (код + книги + архитектура).")
    updater._ensure_collections()
    while True:
        try:
            sync_books()
            sync_go_project()
            build_project_arch_profile()
        except Exception as e:
            print(f"[!] Ошибка в цикле краулера: {e}")
        time.sleep(15)


if __name__ == "__main__":
    rebuild_vector_db()
