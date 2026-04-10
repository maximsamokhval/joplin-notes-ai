import os

from loguru import logger

DEFAULT_SYSTEM_PROMPT = (
    "Ти — Solution Architect. Поверни JSON за заданою схемою.\n"
    "Зверни увагу на вибір `target_notebook` з наданого списку."
)


class PromptLoader:
    def __init__(self, prompt_file: str):
        self._prompt_file = prompt_file

    def load(self) -> str:
        """Read prompt from file without mutating repository files."""
        if not os.path.exists(self._prompt_file):
            logger.warning(
                f"Файл {self._prompt_file} не знайдено. Буде використано вбудований шаблон."
            )
            return DEFAULT_SYSTEM_PROMPT

        with open(self._prompt_file, encoding="utf-8") as f:
            return f.read().strip()
