from loguru import logger

from joplin_notes_ai.clients import JoplinClient, LlmClient, VectorStore
from joplin_notes_ai.config import Settings
from joplin_notes_ai.exceptions import IntegrationError
from joplin_notes_ai.models import (
    NoteDetails,
    NoteSummary,
    ProcessedNoteUpdate,
    ProcessingOutcome,
    TransformationResult,
)
from joplin_notes_ai.services.content_builder import build_final_content


class NoteProcessingService:
    def __init__(
        self,
        settings: Settings,
        joplin_client: JoplinClient,
        llm_client: LlmClient,
        vector_store: VectorStore,
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
        note = self._joplin.get_note(note_summary.id)
        if not note or not note.body.strip():
            return ProcessingOutcome(
                note_id=note_summary.id,
                status="skipped_empty",
                message=f"Пропуск: нотатка '{note_summary.title or note_summary.id}' порожня.",
            )

        if self._is_already_transformed(note):
            if not dry_run:
                self._safe_add_tag(note.id, self._settings.processed_tag)
            return ProcessingOutcome(
                note_id=note.id,
                status="skipped_already_transformed",
                message=f"Нотатка '{note.title}' вже була трансформована раніше.",
            )

        try:
            result = self._llm.transform_note(
                title=note.title,
                body=note.body,
                available_notebooks=list(notebooks_map.keys()),
            )
        except IntegrationError as exc:
            logger.error(str(exc))
            return ProcessingOutcome(
                note_id=note.id,
                status="skipped_llm_failed",
                message=str(exc),
            )

        related = self._collect_related(note, result, dry_run)
        final_content = build_final_content(
            note=note,
            result=result,
            related=related,
            similarity_threshold=self._settings.similarity_threshold,
        )
        update = self._build_note_update(result, final_content, notebooks_map)

        if dry_run:
            return ProcessingOutcome(
                note_id=note.id,
                status="dry_run",
                message=f"Dry-run: згенеровано оновлення для '{note.title}' без запису.",
            )

        try:
            updated = self._joplin.update_note(note.id, update)
        except IntegrationError as exc:
            return ProcessingOutcome(
                note_id=note.id,
                status="failed",
                message=str(exc),
            )
        if not updated:
            return ProcessingOutcome(
                note_id=note.id,
                status="failed",
                message=f"Не вдалося оновити нотатку '{note.title}'.",
            )

        self._safe_add_tag(note.id, self._settings.processed_tag)
        for tag_name in result.suggested_tags:
            self._safe_add_tag(note.id, tag_name)

        return ProcessingOutcome(
            note_id=note.id,
            status="processed",
            message=f"Нотатка '{note.title}' трансформована в '{result.new_title}'.",
        )

    @staticmethod
    def _is_already_transformed(note: NoteDetails) -> bool:
        return "Оригінальна чернетка" in note.body

    def _collect_related(
        self,
        note: NoteDetails,
        result: TransformationResult,
        dry_run: bool,
    ):
        try:
            if not dry_run:
                self._vector_store.upsert_note(note.id, result.new_title, result.content)
            return self._vector_store.find_related(note.id, result.content)
        except IntegrationError as exc:
            logger.warning(f"Пошук або індексація зв'язків пропущені: {exc}")
            return []

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
            self._joplin.add_tag_to_note_by_title(note_id, tag_name)
        except IntegrationError as exc:
            logger.warning(f"Не вдалося додати тег '{tag_name}' для нотатки {note_id}: {exc}")
