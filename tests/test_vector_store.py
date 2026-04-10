import unittest

from joplin_notes_ai.clients.vector_store import VectorStore


class VectorStoreFilteringTestCase(unittest.TestCase):
    def test_filter_related_results_by_similarity_threshold(self):
        results = {
            "ids": [["id-1", "id-2"]],
            "distances": [[0.1, 0.4]],
            "metadatas": [[{"title": "High"}, {"title": "Low"}]],
        }

        related = VectorStore._filter_related_results(results, threshold=0.8)

        self.assertEqual(len(related), 1)
        self.assertEqual(related[0].note_id, "id-1")
        self.assertEqual(related[0].title, "High")
        self.assertGreaterEqual(related[0].similarity, 0.8)


if __name__ == "__main__":
    unittest.main()
