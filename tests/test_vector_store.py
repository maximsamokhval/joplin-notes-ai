import unittest

from joplin_notes_ai.clients.vector_store import VectorStore
from joplin_notes_ai.config import Settings
from joplin_notes_ai.models import NoteDetails


def make_settings() -> Settings:
    return Settings.model_validate(
        {
            "JOPLIN_TOKEN": "token",
            "LLM_API_KEY": "llm-key",
            "SEMANTIC_DEBUG": True,
            "SIMILARITY_THRESHOLD": 0.8,
            "SIMILARITY_TOP_K": 5,
            "SEMANTIC_DEBUG_TOP_K": 7,
        }
    )


class VectorStoreFilteringTestCase(unittest.TestCase):
    def test_build_candidates_marks_rejected_entries(self):
        results = {
            "ids": [["id-1", "id-2", "id-3"]],
            "distances": [[0.1, 0.4, 0.19]],
            "metadatas": [[{"title": "High"}, {"title": "Low"}, {"title": "Borderline"}]],
        }

        candidates = VectorStore._build_candidates(results, threshold=0.8)

        self.assertEqual(len(candidates), 3)
        self.assertTrue(candidates[0].accepted)
        self.assertFalse(candidates[1].accepted)
        self.assertEqual(candidates[1].rejection_reason, "below_threshold")
        self.assertTrue(candidates[2].accepted)

    def test_warmup_returns_success(self):
        store = VectorStore.__new__(VectorStore)
        store._settings = make_settings()
        store._emb_fn = lambda texts: [[0.1, 0.2] for _ in texts]

        result = store.warmup()

        self.assertTrue(result.enabled)
        self.assertTrue(result.success)
        self.assertFalse(result.degraded)
        self.assertGreaterEqual(result.duration_ms, 0.0)

    def test_warmup_returns_degraded_result_on_failure(self):
        store = VectorStore.__new__(VectorStore)
        store._settings = make_settings()

        def failing_embedding(_: list[str]) -> list[list[float]]:
            raise RuntimeError("model unavailable")

        store._emb_fn = failing_embedding

        result = store.warmup()

        self.assertTrue(result.enabled)
        self.assertFalse(result.success)
        self.assertTrue(result.degraded)
        self.assertIn("model unavailable", result.message)

    def test_build_metadata_for_note_contains_extended_fields(self):
        note = NoteDetails(
            id="n1",
            title="  Contract Engineering  ",
            body="Line one\nLine two",
            parent_id="folder-1",
            created_time=111,
            updated_time=222,
            user_updated_time=333,
            is_todo=0,
            source_url="https://example.com/article",
        )

        metadata = VectorStore.build_metadata_for_note(note, "<!-- ai_audited_v1 -->")

        self.assertEqual(metadata["title_normalized"], "contract engineering")
        self.assertEqual(metadata["notebook_id"], "folder-1")
        self.assertEqual(metadata["content_length"], len(note.body))
        self.assertEqual(metadata["line_count"], 2)
        self.assertEqual(metadata["source_updated_time"], 222)
        self.assertIn("indexed_at_unix", metadata)

    def test_build_semantic_text_removes_toc_and_keeps_signal(self):
        content = (
            "Короткий вступ про локальний пошук.\n\n"
            "## Зміст\n"
            "- [Що це](#що-це)\n"
            "- [Висновки](#висновки)\n\n"
            "## Що це\n"
            "QMD поєднує BM25, векторний пошук і reranking.\n\n"
            "## Ключові висновки\n"
            "- Працює локально\n"
            "- Підходить для knowledge base\n\n"
            "<!-- ai_audited_v1 -->\n"
        )

        semantic_text = VectorStore.build_semantic_text(
            "QMD: локальний пошук",
            content,
        )

        self.assertIn("QMD: локальний пошук", semantic_text)
        self.assertIn("QMD поєднує BM25, векторний пошук і reranking.", semantic_text)
        self.assertIn("Працює локально", semantic_text)
        self.assertNotIn("[Що це](#що-це)", semantic_text)
        self.assertNotIn("<!-- ai_audited_v1 -->", semantic_text)


if __name__ == "__main__":
    unittest.main()
