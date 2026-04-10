import unittest
from unittest.mock import Mock, patch

from joplin_notes_ai.app import JoplinNotesAiApp
from joplin_notes_ai.clients.vector_store import NoOpVectorStore
from joplin_notes_ai.config import Settings
from joplin_notes_ai.models import NoteDetails, TagOrganizationOutcome, WarmupResult


def make_settings() -> Settings:
    return Settings.model_validate(
        {
            "JOPLIN_TOKEN": "token",
            "LLM_API_KEY": "llm-key",
            "PAUSE_BETWEEN_NOTES": 0,
        }
    )


class AppVectorStoreBootstrapTestCase(unittest.TestCase):
    def test_build_vector_store_returns_noop_in_dry_run(self) -> None:
        app = JoplinNotesAiApp(make_settings())

        vector_store = app._build_vector_store(dry_run=True)

        self.assertIsInstance(vector_store, NoOpVectorStore)

    @patch("joplin_notes_ai.app.VectorStore")
    def test_build_vector_store_returns_noop_on_degraded_warmup(self, vector_store_cls: Mock) -> None:
        app = JoplinNotesAiApp(make_settings())
        vector_store = vector_store_cls.return_value
        vector_store.warmup.return_value = WarmupResult(
            enabled=True,
            success=False,
            duration_ms=123.0,
            degraded=True,
            message="warmup failed",
        )

        result = app._build_vector_store(dry_run=False)

        self.assertIsInstance(result, NoOpVectorStore)

    @patch("joplin_notes_ai.app.VectorStore")
    def test_build_vector_store_returns_real_store_after_successful_warmup(
        self,
        vector_store_cls: Mock,
    ) -> None:
        app = JoplinNotesAiApp(make_settings())
        vector_store = vector_store_cls.return_value
        vector_store.warmup.return_value = WarmupResult(
            enabled=True,
            success=True,
            duration_ms=42.0,
            degraded=False,
            message="ok",
        )

        result = app._build_vector_store(dry_run=False)

        self.assertIs(result, vector_store)

    def test_reindex_resets_collection_and_upserts_with_metadata(self) -> None:
        app = JoplinNotesAiApp(make_settings())
        vector_store = Mock()
        vector_store.reset_collection.return_value = None
        vector_store.upsert_note_with_metadata.return_value = None
        app._joplin.list_notes_for_indexing = Mock(
            return_value=[
                NoteDetails(
                    id="n1",
                    title="Contracts",
                    body="typed contracts for reliable agents",
                    parent_id="folder-1",
                    updated_time=12345,
                ),
                NoteDetails(
                    id="n2",
                    title="",
                    body=" ",
                ),
            ]
        )

        app._reindex_vector_store(vector_store=vector_store, limit=None)

        vector_store.reset_collection.assert_called_once()
        vector_store.upsert_note_with_metadata.assert_called_once()
        call = vector_store.upsert_note_with_metadata.call_args.kwargs
        self.assertEqual(call["note_id"], "n1")
        self.assertEqual(call["title"], "Contracts")
        self.assertIn("title_normalized", call["metadata"])
        self.assertIn("content_length", call["metadata"])

    def test_reindex_stops_when_vector_store_is_noop(self) -> None:
        app = JoplinNotesAiApp(make_settings())
        app._joplin.list_notes_for_indexing = Mock(return_value=[])

        app._reindex_vector_store(vector_store=NoOpVectorStore(), limit=None)

        app._joplin.list_notes_for_indexing.assert_not_called()

    @patch("joplin_notes_ai.app.TagTaxonomyService")
    def test_run_organize_tags_invokes_taxonomy_service(self, service_cls: Mock) -> None:
        app = JoplinNotesAiApp(make_settings())
        service = service_cls.return_value
        service.organize.return_value = TagOrganizationOutcome(
            status="dry_run",
            message="Preview taxonomy",
            analyzed_tags=5,
            changed_tags=2,
        )

        app.run(dry_run=True, organize_tags=True)

        service.organize.assert_called_once_with(dry_run=True)


if __name__ == "__main__":
    unittest.main()
