import time

import chromadb
from chromadb.utils import embedding_functions
from loguru import logger

from joplin_notes_ai.config import Settings
from joplin_notes_ai.exceptions import VectorStoreError
from joplin_notes_ai.models import RelatedCandidate, RelatedNote, WarmupResult


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

    def warmup(self) -> WarmupResult:
        if not self._settings.embedding_warmup_enabled:
            return WarmupResult(
                enabled=False,
                success=True,
                duration_ms=0.0,
                message="warmup disabled",
            )

        start_t = time.perf_counter()
        try:
            self._emb_fn([self._settings.semantic_warmup_text])
        except Exception as exc:  # noqa: BLE001 - wrapped into a typed result
            duration_ms = (time.perf_counter() - start_t) * 1000
            logger.warning(
                "embedding_warmup_failed model={} duration_ms={:.2f} reason={}",
                self._settings.embedding_model_name,
                duration_ms,
                exc,
            )
            return WarmupResult(
                enabled=True,
                success=False,
                duration_ms=duration_ms,
                degraded=True,
                message=str(exc),
            )

        duration_ms = (time.perf_counter() - start_t) * 1000
        logger.info(
            "embedding_warmup_completed model={} duration_ms={:.2f}",
            self._settings.embedding_model_name,
            duration_ms,
        )
        return WarmupResult(
            enabled=True,
            success=True,
            duration_ms=duration_ms,
            message="embedding model warmed up",
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

    def search_candidates(self, note_id: str, content: str) -> list[RelatedCandidate]:
        top_k = (
            max(self._settings.similarity_top_k, self._settings.semantic_debug_top_k)
            if self._settings.semantic_debug
            else self._settings.similarity_top_k
        )
        try:
            results = self._collection.query(
                query_texts=[content],
                n_results=top_k,
                where={"id": {"$ne": note_id}},
            )
        except Exception as exc:  # noqa: BLE001 - wrapped into a typed exception
            raise VectorStoreError(f"Помилка пошуку в ChromaDB: {exc}") from exc

        candidates = self._build_candidates(results, self._settings.similarity_threshold)
        self._log_candidates(note_id, candidates, top_k)
        return candidates

    def find_related(self, note_id: str, content: str) -> list[RelatedNote]:
        candidates = self.search_candidates(note_id, content)
        return [
            RelatedNote(
                note_id=candidate.note_id,
                title=candidate.title,
                similarity=candidate.similarity,
            )
            for candidate in candidates
            if candidate.accepted
        ]

    @staticmethod
    def _build_candidates(results: dict, threshold: float) -> list[RelatedCandidate]:
        candidates: list[RelatedCandidate] = []
        ids = results.get("ids", [[]])
        distances = results.get("distances", [[]])
        metadatas = results.get("metadatas", [[]])

        if not ids or not ids[0] or not distances or not distances[0]:
            return candidates

        max_distance = 1 - threshold
        raw_ids = ids[0]
        raw_distances = distances[0]
        raw_metadatas = metadatas[0] if metadatas else []

        for idx, distance in enumerate(raw_distances):
            metadata = raw_metadatas[idx] if idx < len(raw_metadatas) else {}
            title = metadata.get("title", raw_ids[idx]) if isinstance(metadata, dict) else raw_ids[idx]
            similarity = 1 - distance
            accepted = distance <= max_distance
            rejection_reason = None if accepted else "below_threshold"
            candidates.append(
                RelatedCandidate(
                    note_id=raw_ids[idx],
                    title=title,
                    distance=distance,
                    similarity=similarity,
                    accepted=accepted,
                    rejection_reason=rejection_reason,
                    rank=idx + 1,
                )
            )

        return candidates

    def _log_candidates(
        self,
        note_id: str,
        candidates: list[RelatedCandidate],
        top_k: int,
    ) -> None:
        accepted = [candidate for candidate in candidates if candidate.accepted]
        best_similarity = max((candidate.similarity for candidate in candidates), default=0.0)
        logger.info(
            "semantic_summary note_id={} requested_top_k={} candidates={} accepted={} "
            "best_similarity={:.4f} threshold={:.4f}",
            note_id,
            top_k,
            len(candidates),
            len(accepted),
            best_similarity,
            self._settings.similarity_threshold,
        )
        if len(candidates) < top_k:
            logger.debug(
                "semantic_summary_insufficient_candidates note_id={} requested_top_k={} actual_candidates={}",
                note_id,
                top_k,
                len(candidates),
            )

        if not self._settings.semantic_debug:
            return

        for candidate in candidates[: self._settings.semantic_log_candidates_limit]:
            logger.debug(
                "semantic_candidate note_id={} rank={} candidate_note_id={} title={!r} "
                "distance={:.4f} similarity={:.4f} accepted={} rejection_reason={}",
                note_id,
                candidate.rank,
                candidate.note_id,
                candidate.title,
                candidate.distance,
                candidate.similarity,
                candidate.accepted,
                candidate.rejection_reason or "published",
            )


class NoOpVectorStore:
    """Vector store implementation for dry-run mode."""

    def warmup(self) -> WarmupResult:
        return WarmupResult(
            enabled=False,
            success=True,
            duration_ms=0.0,
            message="no-op vector store",
        )

    def upsert_note(self, note_id: str, title: str, content: str) -> None:
        return None

    def search_candidates(self, note_id: str, content: str) -> list[RelatedCandidate]:
        return []

    def find_related(self, note_id: str, content: str) -> list[RelatedNote]:
        return []
