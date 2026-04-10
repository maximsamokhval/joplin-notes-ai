import json
import time
from typing import Any

import requests
from loguru import logger
from pydantic import ValidationError

from joplin_notes_ai.config import Settings
from joplin_notes_ai.exceptions import LlmApiError, LlmResponseValidationError
from joplin_notes_ai.models import TagInfo, TagTaxonomyPlan, TransformationResult
from joplin_notes_ai.repositories import PromptLoader


class LlmClient:
    _TAG_TAXONOMY_SYSTEM_PROMPT = """
Ти створюєш чисту, практичну таксономію тегів для особистої бази знань.

Мета:
- зменшити хаос у тегах;
- об'єднати дублікати і близькі за сенсом варіанти;
- залишити короткі, стабільні, корисні для пошуку канонічні теги;
- не плодити зайві майже-синоніми.

Правила:
1. Пиши ТІЛЬКИ валідний JSON.
2. Не вигадуй теги, які не допомагають навігації.
3. Об'єднуй різні мовні варіанти, однину/множину, дрібні орфографічні відмінності та майже-синоніми в один канонічний тег.
4. Якщо тег надто вузький, випадковий, шумовий або одноразовий, можна запропонувати `delete`.
5. Якщо тег уже хороший, використовуй `keep`.
6. Якщо тег треба об'єднати з іншим канонічним тегом, використовуй `merge`.
7. Канонічні теги мають бути короткими, послідовними і придатними для довгострокової системи знань.
8. Не чіпай службові теги, які явно позначені як protected.

Схема відповіді:
{
  "canonical_tags": ["ai", "architecture", "knowledge-management"],
  "assignments": [
    {
      "current_title": "artificial-intelligence",
      "canonical_title": "ai",
      "action": "merge",
      "reason": "Англомовний варіант того самого поняття"
    }
  ],
  "taxonomy_summary": "Короткий опис принципу побудови таксономії"
}
""".strip()

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
        compact_retry_instruction = (
            "ВАЖЛИВО: попередня відповідь була надто довгою або обірвалася. "
            "Поверни коротшу, щільнішу версію замітки без втрати сенсу. "
            "Не роздувай список секцій, не дублюй підпункти, не додавай зайвих прикладів. "
            "Відповідь має бути СТРОГО валідним JSON за схемою."
        )
        return self._request_model(
            system_prompt=system_prompt,
            user_context=user_context,
            response_model=TransformationResult,
            compact_retry_instruction=compact_retry_instruction,
        )

    def generate_tag_taxonomy(
        self,
        tags: list[TagInfo],
        protected_tags: set[str],
    ) -> TagTaxonomyPlan:
        serialized_tags = [
            {
                "title": tag.title,
                "note_count": tag.note_count,
                "sample_note_titles": tag.sample_note_titles,
            }
            for tag in tags
        ]
        user_context = (
            f"Protected tags: {json.dumps(sorted(protected_tags), ensure_ascii=False)}\n\n"
            "Нижче список існуючих тегів із кількістю нотаток та прикладами:\n"
            f"{json.dumps(serialized_tags, ensure_ascii=False, indent=2)}"
        )
        compact_retry_instruction = (
            "Скороти taxonomy_summary і reasons, але збережи повний список assignments. "
            "Поверни тільки валідний JSON."
        )
        return self._request_model(
            system_prompt=self._TAG_TAXONOMY_SYSTEM_PROMPT,
            user_context=user_context,
            response_model=TagTaxonomyPlan,
            compact_retry_instruction=compact_retry_instruction,
        )

    def _request_model(
        self,
        system_prompt: str,
        user_context: str,
        response_model: Any,
        compact_retry_instruction: str,
        compact_retry: bool = False,
    ) -> Any:
        headers = {
            "Authorization": f"Bearer {self._settings.llm_api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self._settings.llm_model_name,
            "messages": [
                {"role": "system", "content": system_prompt},
                {
                    "role": "user",
                    "content": self._build_user_context(
                        user_context,
                        compact_retry,
                        compact_retry_instruction,
                    ),
                },
            ],
            "response_format": {"type": "json_object"},
            "max_tokens": self._settings.llm_max_tokens,
        }

        raw: str = ""
        finish_reason = ""
        start_t = time.time()
        try:
            response = requests.post(
                f"{self._settings.llm_base_url}/chat/completions",
                headers=headers,
                json=payload,
                timeout=self._settings.llm_timeout,
            )
            response.raise_for_status()

            response_json = response.json()
            choice = response_json["choices"][0]
            finish_reason = str(choice.get("finish_reason") or "")
            raw = choice["message"]["content"]
            self._log_response_meta(response_json, finish_reason, raw, compact_retry)
            result = response_model.model_validate_json(raw)
            logger.success(f"Генерацію завершено за {time.time() - start_t:.2f} сек.")
            return result
        except ValidationError as exc:
            logger.debug(f"Сира відповідь LLM: {raw}")
            if not compact_retry and self._should_retry_invalid_json(raw, finish_reason):
                logger.warning(
                    "LLM повернула обірваний або невалідний JSON. Повтор із компактнішою інструкцією."
                )
                return self._request_model(
                    system_prompt=system_prompt,
                    user_context=user_context,
                    response_model=response_model,
                    compact_retry_instruction=compact_retry_instruction,
                    compact_retry=True,
                )
            raise LlmResponseValidationError(f"Невідповідність схеми відповіді LLM: {exc}") from exc
        except requests.RequestException as exc:
            raise LlmApiError(f"Помилка LLM API: {exc}") from exc
        except (KeyError, TypeError, ValueError) as exc:
            raise LlmApiError(f"Некоректна структура відповіді LLM: {exc}") from exc

    @staticmethod
    def _should_retry_invalid_json(raw: str, finish_reason: str) -> bool:
        if finish_reason == "length":
            return True
        stripped = raw.rstrip()
        return bool(stripped) and not stripped.endswith("}")

    @staticmethod
    def _build_user_context(
        user_context: str,
        compact_retry: bool,
        compact_retry_instruction: str,
    ) -> str:
        if not compact_retry:
            return user_context
        return f"{user_context}\n\n{compact_retry_instruction}"

    @staticmethod
    def _log_response_meta(
        response_json: dict,
        finish_reason: str,
        raw: str,
        compact_retry: bool,
    ) -> None:
        usage = response_json.get("usage") if isinstance(response_json, dict) else None
        if not isinstance(usage, dict):
            usage = {}
        logger.info(
            "llm_response_meta finish_reason={} response_chars={} prompt_tokens={} completion_tokens={} compact_retry={}",
            finish_reason or "unknown",
            len(raw),
            usage.get("prompt_tokens", "n/a"),
            usage.get("completion_tokens", "n/a"),
            compact_retry,
        )
