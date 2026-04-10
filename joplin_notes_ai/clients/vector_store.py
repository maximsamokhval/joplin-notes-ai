import time
from typing import Any

import chromadb
from chromadb.utils import embedding_functions
from loguru import logger

from joplin_notes_ai.config import Settings
from joplin_notes_ai.exceptions import VectorStoreError
from joplin_notes_ai.models import NoteDetails, RelatedCandidate, RelatedNote, WarmupResult


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
        self.upsert_note_with_metadata(note_id, title, content, metadata=None)

    def upsert_note_with_metadata(
        self,
        note_id: str,
        title: str,
        content: str,
        metadata: dict[str, Any] | None,
    ) -> None:
        computed_metadata = self._build_index_metadata(note_id, title, content, metadata)
        try:
            self._collection.upsert(
                ids=[note_id],
                documents=[f"{title}\n{content}"],
                metadatas=[computed_metadata],
            )
        except Exception as exc:  # noqa: BLE001 - wrapped into a typed exception
            raise VectorStoreError(f"Помилка upsert в ChromaDB: {exc}") from exc

    def reset_collection(self) -> None:
        try:
            self._client.delete_collection("joplin_notes")
            self._collection = self._client.get_or_create_collection(
                name="joplin_notes",
                embedding_function=self._emb_fn,
                metadata={"hnsw:space": "cosine"},
            )
        except Exception as exc:  # noqa: BLE001 - wrapped into a typed exception
            raise VectorStoreError(f"Помилка скидання колекції ChromaDB: {exc}") from exc

    @staticmethod
    def build_metadata_for_note(note: NoteDetails, machine_marker: str) -> dict[str, Any]:
        normalized_title = (note.title or "").strip().lower()
        normalized_body = (note.body or "").replace("\r\n", "\n").replace("\r", "\n")
        words = [part for part in normalized_body.split() if part]
        lines = [line for line in normalized_body.split("\n") if line.strip()]
        marker_present = bool(machine_marker and machine_marker in normalized_body)

        return {
            "notebook_id": note.parent_id or "",
            "title_normalized": normalized_title,
            "content_preview": normalized_body[:280],
            "content_length": len(normalized_body),
            "word_count": len(words),
            "line_count": len(lines),
            "has_machine_marker": marker_present,
            "source_created_time": int(note.created_time or 0),
            "source_updated_time": int(note.updated_time or 0),
            "source_user_updated_time": int(note.user_updated_time or 0),
            "source_url": (note.source_url or "")[:280],
            "is_todo": int(note.is_todo or 0),
            "indexed_at_unix": int(time.time()),
        }

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

    @staticmethod
    def _build_index_metadata(
        note_id: str,
        title: str,
        content: str,
        metadata: dict[str, Any] | None,
    ) -> dict[str, Any]:
        base: dict[str, Any] = {
            "title": title,
            "id": note_id,
            "indexed_at_unix": int(time.time()),
            "content_length": len(content or ""),
            "title_normalized": (title or "").strip().lower(),
        }
        if metadata:
            base.update(metadata)
        return base


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

    def upsert_note_with_metadata(
        self,
        note_id: str,
        title: str,
        content: str,
        metadata: dict[str, Any] | None,
    ) -> None:
        return None

    def reset_collection(self) -> None:
        return None

    def search_candidates(self, note_id: str, content: str) -> list[RelatedCandidate]:
        return []

    def find_related(self, note_id: str, content: str) -> list[RelatedNote]:
        return []
