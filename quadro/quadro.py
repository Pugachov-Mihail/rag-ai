from collections import defaultdict

import requests
import uuid
import os
import fitz  # PyMuPDF
from datetime import datetime
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct, VectorParams, Distance

# Импортируем классы из вашего детерминированного парсера
from parcer_files.parser import GoAstParser, DependencyGraphBuilder


class ProjectContextUpdater:
    def __init__(self, qdrant_host="localhost", qdrant_port=6333):
        # Оставляем только локальный Qdrant
        self.qdrant = QdrantClient(qdrant_host, port=qdrant_port)

        # Настройки локальной Ollama
        self.ollama_embed_url = "http://localhost:11434/api/embeddings"
        self.embed_model = "nomic-embed-text"

        self.ast_parser = GoAstParser()
        self._ensure_collections()

    def _ensure_collections(self):
        """Создает коллекции для кода, книг и архитектурного профиля (768)"""
        # Коллекция для кода
        if not self.qdrant.collection_exists("go_project_context"):
            self.qdrant.create_collection(
                collection_name="go_project_context",
                vectors_config=VectorParams(size=768, distance=Distance.COSINE),
            )

        # Коллекция для книг
        if not self.qdrant.collection_exists("books_collection"):
            self.qdrant.create_collection(
                collection_name="books_collection",
                vectors_config=VectorParams(size=768, distance=Distance.COSINE),
            )

        # Коллекция для архитектурного профиля
        if not self.qdrant.collection_exists("project_arch_profile"):
            self.qdrant.create_collection(
                collection_name="project_arch_profile",
                vectors_config=VectorParams(size=768, distance=Distance.COSINE),
            )

    def _embed_text(self, text: str):
        """Локальная генерация векторов через Ollama"""
        try:
            response = requests.post(
                self.ollama_embed_url,
                json={"model": self.embed_model, "prompt": text},
                timeout=10
            )
            json_data = response.json()
            return json_data.get("embedding") or json_data.get("embeddings", [None])[0]
        except Exception as e:
            print(f"[!] Ошибка связи с локальной Ollama при векторизации: {e}")
            return None

    def sync_file(self, filepath: str):
        """Парсит Go файл через AST и загружает структуру в Qdrant"""
        print(f"📝 Обработка файла кода: {filepath}")

        chunks = self.ast_parser.parse_file(filepath)
        if not chunks:
            return

        builder = DependencyGraphBuilder()
        builder.build_from_chunks(chunks)

        points = []
        for chunk in chunks:
            meta = chunk['metadata']
            caller_name = meta.get('name', 'unknown')
            callees = builder.graph.get(caller_name, [])
            content = chunk['content']

            # Запрос к локальной модели
            vector = self._embed_text(content)
            if not vector:
                continue

            doc_id = str(uuid.uuid4()).replace("-", "")[:16]
            point = PointStruct(
                id=int(doc_id, 16) % (2 ** 63 - 1),
                vector=vector,
                payload={
                    "file_path": filepath,
                    "module": "unknown",
                    "component": caller_name,
                    "type": meta.get("type", "unknown"),
                    "related_modules": callees,
                    "content": content,
                    "updated_at": datetime.now().isoformat()
                }
            )
            points.append(point)

        if points:
            self.qdrant.upsert(collection_name="go_project_context", points=points)
            print(f"✅ Загружено {len(points)} узлов из AST в Qdrant")

    def sync_book_to_qdrant(self, pdf_path):
        """Обрабатывает PDF и заливает куски текста в Qdrant"""
        filename = os.path.basename(pdf_path)
        print(f"\n[*] Начинаю обработку книги: {filename}")

        try:
            doc = fitz.open(pdf_path)
            full_text = "".join([page.get_text() + "\n" for page in doc])
            doc.close()

            if not full_text.strip():
                return False

            chunks = [full_text[i:i + 1000] for i in range(0, len(full_text), 850)]
            points = []

            for i, chunk in enumerate(chunks):
                vector = self._embed_text(chunk)
                if not vector:
                    return False

                doc_id = str(uuid.uuid4()).replace("-", "")[:16]
                point = PointStruct(
                    id=int(doc_id, 16) % (2 ** 63 - 1),
                    vector=vector,
                    payload={
                        "source": filename,
                        "chunk_index": i,
                        "content": chunk,
                        "type": "book_knowledge"
                    }
                )
                points.append(point)

            if points:
                self.qdrant.upsert(collection_name="books_collection", points=points)
                print(f"[SUCCESS] {filename} добавлена в векторную память Qdrant!")
            return True

        except Exception as e:
            print(f"[!] Ошибка при обработке книги {filename}: {e}")
            return False

    def _infer_module_name(self, file_path: str) -> str:
        """Берём первые 1–2 сегмента пути как имя модуля."""
        if not file_path:
            return "unknown"
        rel = file_path.replace("\\", "/")
        parts = rel.split("/")
        if len(parts) >= 2:
            return "/".join(parts[:2])
        if parts:
            return parts[0]
        return "unknown"

    def _normalize_module_name(self, raw: str) -> str:
        return (raw or "").replace("\\", "/").strip()

    def _infer_layer(self, module_name: str, files: list) -> str:
        name = module_name.lower()
        blob = " ".join(files).lower()
        if any(x in name or x in blob for x in ["ui/", "web/", "frontend", "cmd/"]):
            return "ui"
        if any(x in name or x in blob for x in ["http", "api", "transport", "grpc"]):
            return "transport"
        if any(x in name or x in blob for x in ["usecase", "service", "app/", "application"]):
            return "application"
        if any(x in name or x in blob for x in ["domain", "core", "entity", "aggregate"]):
            return "domain"
        if any(x in name or x in blob for x in ["infra", "repository", "storage", "db", "adapter"]):
            return "infra"
        return "misc"

    def _infer_responsibility(self, module_name: str, layer: str, components: list, files: list) -> str:
        base = f"Модуль {module_name} ({layer} слой). "
        if layer == "ui":
            return base + "Отвечает за пользовательский интерфейс / команды."
        if layer == "transport":
            return base + "Отвечает за входящие/исходящие протоколы (HTTP/gRPC/API)."
        if layer == "application":
            return base + "Координирует бизнес-операции через usecase/service, не хранит доменные данные."
        if layer == "domain":
            return base + "Содержит доменную модель, правила, инварианты и бизнес-логику."
        if layer == "infra":
            return base + "Отвечает за интеграции: БД, очереди, внешние сервисы, адаптеры."
        return base + "Содержит вспомогательный/смешанный код, слой уточнить вручную."

    def _infer_public_entry_points(self, components: list, files: list) -> list:
        entries = []
        for c in components:
            lc = c.lower()
            if any(x in lc for x in ["handler", "usecase", "service", "server", "controller"]):
                entries.append(c)
        for fp in files:
            name = os.path.basename(fp).lower()
            if any(x in name for x in ["handler", "usecase", "service", "server", "controller"]):
                entries.append(fp)
        return sorted(set(entries))

    def _infer_forbidden_patterns(self, layer: str) -> list:
        if layer == "domain":
            return [
                "зависимость на инфраструктуру (db, http, внешние клиенты)",
                "использование контекста запросов напрямую",
            ]
        if layer == "application":
            return ["прямой доступ к UI слоям", "обход доменного слоя при изменении бизнес-логики"]
        if layer == "infra":
            return ["доменные инварианты в инфраструктурном коде"]
        return []

    def _infer_change_recipe(self, layer: str) -> str:
        if layer == "ui":
            return "Добавляй новые входные точки, делегируя логику в application/usecase."
        if layer == "transport":
            return "Новые эндпоинты делай как thin handlers, вызывающие application/usecase."
        if layer == "application":
            return "Новые операции оформляй как usecase/service, дергающие domain и infra."
        if layer == "domain":
            return "Меняй доменные объекты и правила, не ходи напрямую в инфраструктуру."
        if layer == "infra":
            return "Добавляй адаптеры/репозитории, соблюдая контракты, не тащи доменную логику в infra."
        return "Сначала уточни роль модуля, затем следуй общим правилам слоёв."

    def build_arch_profile(self, collection_name: str = "project_arch_profile") -> None:
        """
        Строит архитектурный профиль проекта на основе коллекции go_project_context.

        Для каждого модуля сохраняется:
        - module_name
        - layer (ui / transport / application / domain / infra / misc)
        - responsibility
        - depends_on / used_by
        - public_entry_points
        - forbidden_patterns
        - change_recipe
        """
        print(f"[🏗] Обновление архитектурного профиля ({collection_name})...")

        client = self.qdrant
        modules = defaultdict(lambda: {
            "files": [],
            "components": [],
            "depends_on_modules": set(),
        })

        # 1. Считываем все точки кода через scroll
        try:
            offset = None
            while True:
                # scroll возвращает (points, next_offset)
                points, offset = self.qdrant.scroll(
                    collection_name="go_project_context",
                    limit=256,
                    offset=offset,
                    with_payload=True,
                    with_vectors=False,
                )
                if not points:
                    break

                for p in points:
                    payload = p.payload or {}
                    file_path = payload.get("file_path", "")
                    component = payload.get("component", "")
                    related_modules = payload.get("related_modules", []) or []

                    module_name = self._infer_module_name(file_path)
                    m = modules[module_name]

                    if file_path:
                        m["files"].append(file_path)
                    if component:
                        m["components"].append(component)

                    for dep in related_modules:
                        dep_module = self._normalize_module_name(dep)
                        if dep_module and dep_module != module_name:
                            m["depends_on_modules"].add(dep_module)

                if offset is None:
                    break
        except Exception as e:
            print(f"[!] Ошибка чтения go_project_context: {e}")
            return

        if not modules:
            print("[⚠️] В go_project_context нет точек — профиль не обновлён.")
            return

        # 2. Обратные зависимости used_by
        used_by = defaultdict(set)
        for module_name, data in modules.items():
            for dep in data["depends_on_modules"]:
                used_by[dep].add(module_name)

        # 3. Формируем точки для project_arch_profile
        arch_points = []
        idx = 0

        for module_name, data in modules.items():
            files = sorted(set(data["files"]))
            components = sorted(set(data["components"]))
            depends_on_modules = sorted(data["depends_on_modules"])
            used_by_modules = sorted(used_by.get(module_name, set()))

            layer = self._infer_layer(module_name, files)
            responsibility = self._infer_responsibility(module_name, layer, components, files)
            public_entry_points = self._infer_public_entry_points(components, files)
            forbidden_patterns = self._infer_forbidden_patterns(layer)
            change_recipe = self._infer_change_recipe(layer)

            # Вектор — эмбед responsibility (чтобы можно было искать по тексту)
            vector = self._embed_text(responsibility or module_name) or [0.0] * 768

            payload = {
                "module_name": module_name,
                "layer": layer,
                "responsibility": responsibility,
                "depends_on": depends_on_modules,
                "used_by": used_by_modules,
                "public_entry_points": public_entry_points,
                "forbidden_patterns": forbidden_patterns,
                "change_recipe": change_recipe,
            }

            arch_points.append(
                PointStruct(
                    id=f"arch_{idx}",
                    vector=vector,
                    payload=payload,
                )
            )
            idx += 1

        # 4. Перезаписываем коллекцию
        try:
            try:
                client.delete_collection(collection_name=collection_name)
            except Exception:
                pass
            self._ensure_collections()  # создаст project_arch_profile, если удалили
            client.upsert(collection_name=collection_name, points=arch_points)
            print(f"[🏁] Архитектурный профиль записан: {len(arch_points)} модулей.")
        except Exception as e:
            print(f"[!] Ошибка записи project_arch_profile: {e}")