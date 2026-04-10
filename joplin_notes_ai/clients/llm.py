import time

import requests
from loguru import logger
from pydantic import ValidationError

from joplin_notes_ai.config import Settings
from joplin_notes_ai.exceptions import LlmApiError, LlmResponseValidationError
from joplin_notes_ai.models import TransformationResult
from joplin_notes_ai.repositories import PromptLoader


class LlmClient:
    def __init__(self, settings: Settings, prompt_loader: PromptLoader):
        self._settings = settings
        self._prompt_loader = prompt_loader

    def transform_note(
        self,
        title: str,
        body: str,
        available_notebooks: list[str],
    ) -> TransformationResult:
        system_prompt = self._prompt_loader.load()
        notebooks_str = ", ".join([f'"{n}"' for n in available_notebooks])
        user_context = (
            f"Список ІСНУЮЧИХ блокнотів для маршрутизації: [{notebooks_str}]\n\n"
            f"Заголовок чернетки: {title}\nКонтент чернетки:\n{body}"
        )

        headers = {
            "Authorization": f"Bearer {self._settings.llm_api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self._settings.llm_model_name,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_context},
            ],
            "response_format": {"type": "json_object"},
        }

        raw: str = ""
        start_t = time.time()
        try:
            response = requests.post(
                f"{self._settings.llm_base_url}/chat/completions",
                headers=headers,
                json=payload,
                timeout=self._settings.llm_timeout,
            )
            response.raise_for_status()

            raw = response.json()["choices"][0]["message"]["content"]
            result = TransformationResult.model_validate_json(raw)
            logger.success(f"Генерацію завершено за {time.time() - start_t:.2f} сек.")
            return result
        except ValidationError as exc:
            logger.debug(f"Сира відповідь LLM: {raw}")
            raise LlmResponseValidationError(f"Невідповідність схеми відповіді LLM: {exc}") from exc
        except requests.RequestException as exc:
            raise LlmApiError(f"Помилка LLM API: {exc}") from exc
        except (KeyError, TypeError, ValueError) as exc:
            raise LlmApiError(f"Некоректна структура відповіді LLM: {exc}") from exc
