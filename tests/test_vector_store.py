import unittest

from joplin_notes_ai.clients.vector_store import VectorStore
from joplin_notes_ai.config import Settings


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


if __name__ == "__main__":
    unittest.main()
