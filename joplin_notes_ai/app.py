import time

from loguru import logger

from joplin_notes_ai.clients import JoplinClient, LlmClient, NoOpVectorStore, VectorStore
from joplin_notes_ai.config import Settings
from joplin_notes_ai.models import Notebook
from joplin_notes_ai.repositories import PromptLoader
from joplin_notes_ai.services import NoteProcessingService


class JoplinNotesAiApp:
    def __init__(self, settings: Settings):
        self._settings = settings
        self._joplin = JoplinClient(settings)
        self._llm = LlmClient(settings, PromptLoader(settings.prompt_file))

    def run(self, dry_run: bool = False, limit: int | None = None) -> None:
        logger.info("Старт автоматизованої системи трансформації знань.")
        vector_store = self._build_vector_store(dry_run)
        service = NoteProcessingService(
            settings=self._settings,
            joplin_client=self._joplin,
            llm_client=self._llm,
            vector_store=vector_store,
        )

        notebooks = self._joplin.list_notebooks()
        notebooks_map = self._to_notebook_map(notebooks)
        if not notebooks_map:
            logger.error("Не вдалося завантажити структуру блокнотів. Завершення.")
            return

        notes = self._joplin.search_unprocessed_notes(self._settings.processed_tag)
        notes_to_process = notes[:limit] if limit is not None else notes
        logger.info(f"Знайдено {len(notes_to_process)} нотаток для обробки.")

        for note_summary in notes_to_process:
            outcome = service.process(note_summary, notebooks_map, dry_run=dry_run)
            if outcome.status in {"processed", "dry_run"}:
                logger.success(outcome.message)
            elif outcome.status.startswith("skipped"):
                logger.info(outcome.message)
            else:
                logger.error(outcome.message)
            time.sleep(self._settings.pause_between_notes)

        logger.info("Сесію генерації завершено.")

    @staticmethod
    def _to_notebook_map(notebooks: list[Notebook]) -> dict[str, str]:
        return {notebook.title: notebook.id for notebook in notebooks}

    def _build_vector_store(self, dry_run: bool) -> NoOpVectorStore | VectorStore:
        if dry_run:
            logger.info("Vector store disabled in dry-run mode.")
            return NoOpVectorStore()

        try:
            vector_store = VectorStore(self._settings)
        except Exception as exc:  # noqa: BLE001 - explicit degradation path
            logger.warning(f"Не вдалося ініціалізувати vector store. Деградація до no-op режиму: {exc}")
            return NoOpVectorStore()

        warmup_result = vector_store.warmup()
        if warmup_result.degraded:
            logger.warning(
                "embedding_warmup_degraded enabled={} success={} duration_ms={:.2f} message={!r}",
                warmup_result.enabled,
                warmup_result.success,
                warmup_result.duration_ms,
                warmup_result.message,
            )
            return NoOpVectorStore()

        logger.info(
            "embedding_warmup_ready enabled={} success={} duration_ms={:.2f}",
            warmup_result.enabled,
            warmup_result.success,
            warmup_result.duration_ms,
        )
        return vector_store
