import unittest
from pathlib import Path
from uuid import uuid4

from joplin_notes_ai.repositories.prompt_loader import DEFAULT_SYSTEM_PROMPT, PromptLoader


class PromptLoaderTestCase(unittest.TestCase):
    def test_loads_default_prompt_when_file_missing(self):
        loader = PromptLoader("missing_prompt_file.txt")
        prompt = loader.load()
        self.assertEqual(prompt, DEFAULT_SYSTEM_PROMPT)

    def test_loads_prompt_from_file(self):
        prompt_path = Path(f"tests/.tmp_prompt_{uuid4().hex}.txt")
        try:
            prompt_path.write_text("custom prompt", encoding="utf-8")
            loader = PromptLoader(str(prompt_path))
            self.assertEqual(loader.load(), "custom prompt")
        finally:
            try:
                prompt_path.unlink(missing_ok=True)
            except PermissionError:
                pass


if __name__ == "__main__":
    unittest.main()
