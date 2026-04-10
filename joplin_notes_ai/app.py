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
        vector_store = NoOpVectorStore() if dry_run else VectorStore(self._settings)
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
