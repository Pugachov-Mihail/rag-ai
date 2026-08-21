import uuid
from datetime import datetime

import fitz
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct, VectorParams, Distance

# Импортируем классы из вашего parser.py
from parcer_files import GoAstParser, DependencyGraphBuilder
from reader_book import get_ollama_embedding


class ProjectContextUpdater:
    def __init__(self, qdrant_host="localhost", qdrant_port=6333, openai_api_key=None):
        self.qdrant = QdrantClient(qdrant_host, port=qdrant_port)
        self.openai = OpenAI(api_key=openai_api_key)
        self.collection_name = "go_project_context"

        # Инициализируем ваш AST парсер
        self.ast_parser = GoAstParser()

        self._ensure_collection()

    def _ensure_collection(self):
        """Создает коллекцию в Qdrant, если её еще нет."""
        if not self.qdrant.collection_exists(self.collection_name):
            self.qdrant.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(size=1536, distance=Distance.COSINE),
            )

    def _embed_text(self, text: str) -> list[float]:
        """Получаем эмбеддинги кода для семантического поиска."""
        response = self.openai.embeddings.create(
            input=text,
            model="text-embedding-3-small"
        )
        return response.data[0].embedding

    def sync_file(self, filepath: str):
        """Парсит Go файл через AST и загружает структуру в Qdrant."""
        print(f"📝 Обработка файла: {filepath}")

        # 1. Извлекаем AST-чанки (без LLM!)
        chunks = self.ast_parser.parse_file(filepath)
        if not chunks:
            print("В файле не найдено значимых функций/типов.")
            return

        # 2. Строим граф зависимостей для этого файла
        builder = DependencyGraphBuilder()
        builder.build_from_chunks(chunks)

        points = []
        for chunk in chunks:
            meta = chunk['metadata']
            caller_name = meta.get('name', 'unknown')

            # Получаем все вызываемые функции из нашего графа
            callees = builder.graph.get(caller_name, [])

            content = chunk['content']
            vector = self._embed_text(content)

            # Генерируем ID для вектора
            doc_id = str(uuid.uuid4()).replace("-", "")[:16]

            # 3. Формируем вектор с точными метаданными
            point = PointStruct(
                id=int(doc_id, 16) % (2 ** 63 - 1),
                vector=vector,
                payload={
                    "file_path": filepath,
                    "module": "unknown",  # Можно дописать парсинг package name
                    "component": caller_name,
                    "type": meta.get("type", "unknown"),  # function_declaration, и т.д.
                    "related_modules": callees,  # <-- Точные зависимости!
                    "content": content,
                    "updated_at": datetime.now().isoformat()
                }
            )
            points.append(point)

        # 4. Загружаем в базу
        if points:
            self.qdrant.upsert(
                collection_name=self.collection_name,
                points=points
            )
            print(f"✅ Загружено {len(points)} узлов из AST")

    def _ensure_collections(self):
        """Создаем коллекции и для кода, и для книг"""
        # Коллекция для кода
        if not self.qdrant.collection_exists("go_project_context"):
            self.qdrant.create_collection(
                collection_name="go_project_context",
                vectors_config=VectorParams(size=768, distance=Distance.COSINE),
                # nomic-embed-text использует размерность 768
            )

        # Коллекция для книг
        if not self.qdrant.collection_exists("books_collection"):
            self.qdrant.create_collection(
                collection_name="books_collection",
                vectors_config=VectorParams(size=768, distance=Distance.COSINE),
            )

    def sync_book_to_qdrant(self, pdf_path, os=None):
        """Обрабатывает PDF и заливает куски в Qdrant"""
        filename = os.path.basename(pdf_path)
        print(f"\n[*] Начинаю обработку книги для Qdrant: {filename}")

        try:
            doc = fitz.open(pdf_path)
            full_text = "".join([page.get_text() + "\n" for page in doc])
            doc.close()

            if not full_text.strip():
                print(f"[!] Текст не найден в {filename}. Пропускаю.")
                return False

            chunks = []
            start = 0
            # Ваша логика нарезки
            while start < len(full_text):
                end = start + 1000
                chunks.append(full_text[start:end])
                start += 1000 - 150

            points = []
            for i, chunk in enumerate(chunks):
                # Здесь вы вызываете вашу функцию get_ollama_embedding
                vector = get_ollama_embedding(chunk)
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
                self.qdrant.upsert(
                    collection_name="books_collection",
                    points=points
                )
                print(f"[SUCCESS] {filename} добавлена в векторную память Qdrant!")
            return True

        except Exception as e:
            print(f"[!] Ошибка при обработке книги {filename}: {e}")
            return False

if __name__ == "__main__":
    # Пример использования:
    updater = ProjectContextUpdater(openai_api_key="sk-ваш-ключ")

    # Для теста загрузим файл, который вы парсили
    updater.sync_file("test.go")