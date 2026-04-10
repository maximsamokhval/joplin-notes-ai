import re
import time

from loguru import logger

from joplin_notes_ai.clients import JoplinClient, LlmClient, NoOpVectorStore, VectorStore
from joplin_notes_ai.config import Settings
from joplin_notes_ai.exceptions import IntegrationError
from joplin_notes_ai.models import (
    NoteDetails,
    NoteSummary,
    ProcessedNoteUpdate,
    ProcessingOutcome,
    RelatedCandidate,
    RelatedNote,
    TransformationResult,
)
from joplin_notes_ai.services.content_builder import build_final_content


class NoteProcessingService:
    def __init__(
        self,
        settings: Settings,
        joplin_client: JoplinClient,
        llm_client: LlmClient,
        vector_store: VectorStore | NoOpVectorStore,
    ):
        self._settings = settings
        self._joplin = joplin_client
        self._llm = llm_client
        self._vector_store = vector_store

    def process(
        self,
        note_summary: NoteSummary,
        notebooks_map: dict[str, str],
        dry_run: bool = False,
    ) -> ProcessingOutcome:
        started_at = time.perf_counter()
        timings_ms: dict[str, float] = {}
        source_title = note_summary.title or note_summary.id
        target_title = ""

        try:
            stage_started = time.perf_counter()
            note = self._joplin.get_note(note_summary.id)
            timings_ms["load_note"] = self._elapsed_ms(stage_started)
            if not note or not note.body.strip():
                return self._finalize_outcome(
                    note_id=note_summary.id,
                    source_title=source_title,
                    target_title=target_title,
                    status="skipped_empty",
                    message=f"Пропуск: нотатка '{source_title}' порожня.",
                    started_at=started_at,
                    timings_ms=timings_ms,
                )
            source_title = note.title

            if self._is_already_transformed(note):
                if not dry_run:
                    stage_started = time.perf_counter()
                    self._safe_add_tag(note.id, self._settings.processed_tag)
                    timings_ms["tagging"] = self._elapsed_ms(stage_started)
                return self._finalize_outcome(
                    note_id=note.id,
                    source_title=source_title,
                    target_title=target_title,
                    status="skipped_already_transformed",
                    message=f"Нотатка '{note.title}' вже була трансформована раніше.",
                    started_at=started_at,
                    timings_ms=timings_ms,
                )

            normalized_body = self._normalize_note_for_llm(note.body)
            if not normalized_body:
                return self._finalize_outcome(
                    note_id=note.id,
                    source_title=source_title,
                    target_title=target_title,
                    status="skipped_empty",
                    message=f"Пропуск: нотатка '{note.title}' порожня після нормалізації.",
                    started_at=started_at,
                    timings_ms=timings_ms,
                )

            stage_started = time.perf_counter()
            result = self._llm.transform_note(
                title=note.title,
                body=normalized_body,
                available_notebooks=list(notebooks_map.keys()),
            )
            timings_ms["llm"] = self._elapsed_ms(stage_started)
            target_title = result.new_title
        except IntegrationError as exc:
            logger.error(str(exc))
            if not dry_run:
                stage_started = time.perf_counter()
                self._safe_add_tag(note_summary.id, self._settings.failed_tag)
                timings_ms["tagging"] = self._elapsed_ms(stage_started)
            return self._finalize_outcome(
                note_id=note_summary.id,
                source_title=source_title,
                target_title=target_title,
                status="skipped_llm_failed",
                message=str(exc),
                started_at=started_at,
                timings_ms=timings_ms,
            )

        related, candidates = self._collect_related(note, result, dry_run, timings_ms)
        final_content = build_final_content(
            result=result,
            related=related,
            similarity_threshold=self._settings.similarity_threshold,
            machine_marker=self._settings.machine_marker,
        )
        update = self._build_note_update(result, final_content, notebooks_map)

        if dry_run:
            self._log_semantic_decision(note.id, candidates)
            return self._finalize_outcome(
                note_id=note.id,
                source_title=source_title,
                target_title=target_title,
                status="dry_run",
                message=f"Dry-run: згенеровано оновлення для '{note.title}' без запису.",
                started_at=started_at,
                timings_ms=timings_ms,
            )

        try:
            stage_started = time.perf_counter()
            updated = self._joplin.update_note(note.id, update)
            timings_ms["update_note"] = self._elapsed_ms(stage_started)
        except IntegrationError as exc:
            stage_started = time.perf_counter()
            self._safe_add_tag(note.id, self._settings.failed_tag)
            timings_ms["tagging"] = timings_ms.get("tagging", 0.0) + self._elapsed_ms(stage_started)
            return self._finalize_outcome(
                note_id=note.id,
                source_title=source_title,
                target_title=target_title,
                status="failed",
                message=str(exc),
                started_at=started_at,
                timings_ms=timings_ms,
            )
        if not updated:
            stage_started = time.perf_counter()
            self._safe_add_tag(note.id, self._settings.failed_tag)
            timings_ms["tagging"] = timings_ms.get("tagging", 0.0) + self._elapsed_ms(stage_started)
            return self._finalize_outcome(
                note_id=note.id,
                source_title=source_title,
                target_title=target_title,
                status="failed",
                message=f"Не вдалося оновити нотатку '{note.title}'.",
                started_at=started_at,
                timings_ms=timings_ms,
            )

        stage_started = time.perf_counter()
        self._apply_tags(note.id, result.suggested_tags)
        timings_ms["tagging"] = self._elapsed_ms(stage_started)
        self._log_semantic_decision(note.id, candidates)

        return self._finalize_outcome(
            note_id=note.id,
            source_title=source_title,
            target_title=target_title,
            status="processed",
            message=f"Нотатка '{note.title}' трансформована в '{result.new_title}'.",
            started_at=started_at,
            timings_ms=timings_ms,
        )

    def _is_already_transformed(self, note: NoteDetails) -> bool:
        return (
            self._settings.machine_marker in note.body
            or "Оригінальна чернетка (Backup)" in note.body
        )

    def _collect_related(
        self,
        note: NoteDetails,
        result: TransformationResult,
        dry_run: bool,
        timings_ms: dict[str, float],
    ) -> tuple[list[RelatedNote], list[RelatedCandidate]]:
        try:
            if not dry_run:
                stage_started = time.perf_counter()
                self._vector_store.upsert_note(note.id, result.new_title, result.content)
                timings_ms["upsert_embedding"] = self._elapsed_ms(stage_started)

            stage_started = time.perf_counter()
            candidates = self._vector_store.search_candidates(
                note.id,
                result.new_title,
                result.content,
            )
            timings_ms["semantic_query"] = self._elapsed_ms(stage_started)
            return self._select_related_notes(candidates), candidates
        except IntegrationError as exc:
            logger.warning(f"Пошук або індексація зв'язків пропущені: {exc}")
            return [], []

    @staticmethod
    def _build_note_update(
        result: TransformationResult,
        final_content: str,
        notebooks_map: dict[str, str],
    ) -> ProcessedNoteUpdate:
        parent_id = notebooks_map.get(result.target_notebook)
        if not parent_id:
            logger.warning(
                f"Блокнот '{result.target_notebook}' не знайдено. Нотатка залишиться на місці."
            )
        return ProcessedNoteUpdate(
            title=result.new_title,
            body=final_content,
            parent_id=parent_id,
        )

    def _safe_add_tag(self, note_id: str, tag_name: str) -> None:
        try:
            if not tag_name.strip():
                return
            added = self._joplin.add_tag_to_note_by_title(note_id, tag_name)
            if not added:
                logger.warning(f"Тег '{tag_name}' не було додано до нотатки {note_id}.")
        except IntegrationError as exc:
            logger.warning(f"Не вдалося додати тег '{tag_name}' для нотатки {note_id}: {exc}")

    @staticmethod
    def _select_related_notes(candidates: list[RelatedCandidate]) -> list[RelatedNote]:
        return [
            RelatedNote(
                note_id=candidate.note_id,
                title=candidate.title,
                similarity=candidate.similarity,
            )
            for candidate in candidates
            if candidate.accepted
        ]

    def _log_semantic_decision(
        self,
        note_id: str,
        candidates: list[RelatedCandidate],
    ) -> None:
        if not candidates:
            logger.debug("semantic_decision note_id={} candidates=0 published=0", note_id)
            return

        published = [candidate for candidate in candidates if candidate.accepted]
        logger.debug(
            "semantic_decision note_id={} candidates={} published={} below_threshold={}",
            note_id,
            len(candidates),
            len(published),
            len(candidates) - len(published),
        )

    def _apply_tags(self, note_id: str, suggested_tags: list[str]) -> None:
        normalized_tags = self._normalize_tags(suggested_tags)
        tags = [self._settings.processed_tag, *normalized_tags]
        for tag_name in tags:
            self._safe_add_tag(note_id, tag_name)

        try:
            actual_tags = self._joplin.get_note_tag_titles(note_id)
            missing = [tag for tag in tags if tag.lower() not in actual_tags]
            if missing:
                logger.warning(
                    f"Після оновлення нотатки {note_id} не знайдено тегів: {', '.join(missing)}"
                )
        except IntegrationError as exc:
            logger.warning(f"Не вдалося перевірити теги для нотатки {note_id}: {exc}")

    @staticmethod
    def _normalize_tags(tags: list[str]) -> list[str]:
        normalized: list[str] = []
        seen: set[str] = set()
        for raw_tag in tags:
            clean = raw_tag.strip().lstrip("#").strip()
            if not clean:
                continue
            key = clean.lower()
            if key in seen:
                continue
            seen.add(key)
            normalized.append(clean)
        return normalized

    @staticmethod
    def _normalize_note_for_llm(body: str) -> str:
        marker_re = re.compile(r"<!--\s*ai_audited_v1\s*-->")
        legacy_backup_re = re.compile(
            r"<details>\s*<summary>Оригінальна чернетка \(Backup\)</summary>.*?</details>",
            re.DOTALL,
        )
        normalized = body.replace("\r\n", "\n").replace("\r", "\n")
        normalized = marker_re.sub("", normalized)
        normalized = legacy_backup_re.sub("", normalized)
        normalized = "\n".join(line.rstrip() for line in normalized.split("\n"))
        normalized = re.sub(r"\n{3,}", "\n\n", normalized).strip()
        return normalized

    @staticmethod
    def _elapsed_ms(started_at: float) -> float:
        return (time.perf_counter() - started_at) * 1000

    def _finalize_outcome(
        self,
        note_id: str,
        source_title: str,
        target_title: str,
        status: str,
        message: str,
        started_at: float,
        timings_ms: dict[str, float],
    ) -> ProcessingOutcome:
        total_ms = self._elapsed_ms(started_at)
        logger.info(
            "processing_summary note_id={} source_title={!r} target_title={!r} status={} "
            "total_ms={:.2f} load_note_ms={:.2f} llm_ms={:.2f} upsert_embedding_ms={:.2f} "
            "semantic_query_ms={:.2f} update_note_ms={:.2f} tagging_ms={:.2f}",
            note_id,
            source_title,
            target_title,
            status,
            total_ms,
            timings_ms.get("load_note", 0.0),
            timings_ms.get("llm", 0.0),
            timings_ms.get("upsert_embedding", 0.0),
            timings_ms.get("semantic_query", 0.0),
            timings_ms.get("update_note", 0.0),
            timings_ms.get("tagging", 0.0),
        )
        return ProcessingOutcome(note_id=note_id, status=status, message=message)
