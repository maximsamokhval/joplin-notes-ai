import unittest
from unittest.mock import Mock, patch

from joplin_notes_ai.app import JoplinNotesAiApp
from joplin_notes_ai.clients.vector_store import NoOpVectorStore
from joplin_notes_ai.config import Settings
from joplin_notes_ai.models import WarmupResult


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


if __name__ == "__main__":
    unittest.main()
