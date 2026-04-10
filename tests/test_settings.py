import unittest

from joplin_notes_ai.config import Settings


class SettingsTestCase(unittest.TestCase):
    def test_settings_defaults_and_aliases(self):
        settings = Settings(
            _env_file=None,
            JOPLIN_TOKEN="token",
            LLM_API_KEY="llm-key",
        )

        self.assertEqual(settings.joplin_token, "token")
        self.assertEqual(settings.llm_api_key, "llm-key")
        self.assertEqual(settings.similarity_threshold, 0.8)
        self.assertEqual(settings.embedding_model_name, "all-MiniLM-L6-v2")
        self.assertEqual(settings.llm_max_tokens, 4000)
        self.assertEqual(settings.similarity_top_k, 5)
        self.assertEqual(settings.pause_between_notes, 1.5)


if __name__ == "__main__":
    unittest.main()
