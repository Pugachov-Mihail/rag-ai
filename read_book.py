import chromadb
import uuid
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct


def migrate_books():
    print("[🔄] Начинаем миграцию книг из ChromaDB в Qdrant...")

    # 1. Подключаемся к базам
    chroma_client = chromadb.PersistentClient(path="./my_library_db")
    qdrant_client = QdrantClient("localhost", port=6333)

    try:
        chroma_books = chroma_client.get_collection(name="books_collection")
    except Exception as e:
        print("[!] Коллекция книг в ChromaDB не найдена. Проверьте путь.")
        return

    total_chunks = chroma_books.count()
    print(f"[*] Найдено фрагментов текста в старой базе: {total_chunks}")

    if total_chunks == 0:
        print("[!] Старая база пуста.")
        return

    # 2. Убеждаемся, что коллекция в Qdrant существует
    if not qdrant_client.collection_exists("books_collection"):
        from qdrant_client.models import VectorParams, Distance
        qdrant_client.create_collection(
            collection_name="books_collection",
            vectors_config=VectorParams(size=768, distance=Distance.COSINE),
        )

    # 3. Переносим данные пачками (батчами) по 1000 штук
    batch_size = 1000
    migrated_count = 0

    for offset in range(0, total_chunks, batch_size):
        # Достаем пачку из ChromaDB
        batch = chroma_books.get(
            include=["documents", "metadatas", "embeddings"],
            limit=batch_size,
            offset=offset
        )

        points = []
        for i in range(len(batch['ids'])):
            # Генерируем ID для Qdrant
            doc_id = str(uuid.uuid4()).replace("-", "")[:16]

            point = PointStruct(
                id=int(doc_id, 16) % (2 ** 63 - 1),
                vector=batch['embeddings'][i],  # Берем готовый вектор!
                payload={
                    "source": batch['metadatas'][i].get('source', 'unknown'),
                    "chunk_index": batch['metadatas'][i].get('chunk_index', 0),
                    "content": batch['documents'][i],
                    "type": "book_knowledge"
                }
            )
            points.append(point)

        # Отправляем пачку в Qdrant
        if points:
            qdrant_client.upsert(
                collection_name="books_collection",
                points=points
            )

        migrated_count += len(points)
        print(f"    Прогресс миграции: {migrated_count} / {total_chunks} фрагментов...")

    print("\n[✅] Миграция успешно завершена! Ваш гигабайт книг теперь в Qdrant.")


if __name__ == "__main__":
    migrate_books()