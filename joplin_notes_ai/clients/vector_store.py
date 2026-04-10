import chromadb
from chromadb.utils import embedding_functions
from loguru import logger

from joplin_notes_ai.config import Settings
from joplin_notes_ai.exceptions import VectorStoreError
from joplin_notes_ai.models import RelatedNote


class VectorStore:
    def __init__(self, settings: Settings):
        self._settings = settings
        self._emb_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=settings.embedding_model_name
        )
        self._client = chromadb.PersistentClient(path=settings.chroma_db_path)
        self._collection = self._client.get_or_create_collection(
            name="joplin_notes",
            embedding_function=self._emb_fn,
            metadata={"hnsw:space": "cosine"},
        )

    def upsert_note(self, note_id: str, title: str, content: str) -> None:
        try:
            self._collection.upsert(
                ids=[note_id],
                documents=[f"{title}\n{content}"],
                metadatas=[{"title": title, "id": note_id}],
            )
        except Exception as exc:  # noqa: BLE001 - wrapped into a typed exception
            raise VectorStoreError(f"Помилка upsert в ChromaDB: {exc}") from exc

    def find_related(self, note_id: str, content: str) -> list[RelatedNote]:
        try:
            results = self._collection.query(
                query_texts=[content],
                n_results=self._settings.similarity_top_k,
                where={"id": {"$ne": note_id}},
            )
        except Exception as exc:  # noqa: BLE001 - wrapped into a typed exception
            raise VectorStoreError(f"Помилка пошуку в ChromaDB: {exc}") from exc

        return self._filter_related_results(results, self._settings.similarity_threshold)

    @staticmethod
    def _filter_related_results(results: dict, threshold: float) -> list[RelatedNote]:
        related: list[RelatedNote] = []
        if not results.get("ids") or not results.get("distances"):
            return related

        max_distance = 1 - threshold
        logger.debug(f"Поріг similarity: {threshold} (максимальна дистанція: {max_distance:.4f})")

        distances = results["distances"][0]
        ids = results["ids"][0]
        metadatas = results.get("metadatas", [[]])[0]

        for idx, distance in enumerate(distances):
            if distance > max_distance:
                continue

            similarity = 1 - distance
            metadata = metadatas[idx] if idx < len(metadatas) else {}
            related.append(
                RelatedNote(
                    note_id=ids[idx],
                    title=metadata.get("title", ids[idx]),
                    similarity=similarity,
                )
            )

        return related


class NoOpVectorStore:
    """Vector store implementation for dry-run mode."""

    def upsert_note(self, note_id: str, title: str, content: str) -> None:
        return None

    def find_related(self, note_id: str, content: str) -> list[RelatedNote]:
        return []
