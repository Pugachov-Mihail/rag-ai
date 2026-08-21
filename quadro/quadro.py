import requests
import uuid
import os
import fitz  # PyMuPDF
from datetime import datetime
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct, VectorParams, Distance

# Импортируем классы из вашего детерминированного парсера
from parcer_files import GoAstParser, DependencyGraphBuilder
from reader_book import get_ollama_embedding


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
        """Создает коллекции для кода и книг с размерностью nomic-embed-text (768)"""
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