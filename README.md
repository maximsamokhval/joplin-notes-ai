# joplin-notes-ai

Рефакторинг застосунку у модульну архітектуру з чіткими шарами:

- `joplin_notes_ai/config.py` - конфігурація через `pydantic-settings`.
- `joplin_notes_ai/clients/` - інтеграції з Joplin API, LLM API, ChromaDB.
- `joplin_notes_ai/models/` - доменні й контрактні моделі.
- `joplin_notes_ai/services/` - orchestration і побудова фінального Markdown.
- `joplin_notes_ai/repositories/prompt_loader.py` - читання system prompt без автозапису файлів.
- `joplin_notes_ai/cli.py` - тонкий CLI-вхід.

## Запуск

Сумісний старий entrypoint:

```bash
python main_vector.py
```

Новий CLI:

```bash
python -m joplin_notes_ai
```

або через скрипт:

```bash
joplin-notes-ai
```

## Прапорці CLI

- `--dry-run` - обробляє нотатки без запису в Joplin і без upsert у ChromaDB.
- `--limit N` - обмежує кількість нотаток в одному запуску.

## Конфігурація

Обов'язкові змінні:

- `JOPLIN_TOKEN`
- `LLM_API_KEY`

Ключові опційні змінні:

- `JOPLIN_BASE_URL`
- `LLM_BASE_URL`
- `LLM_MODEL_NAME`
- `PROMPT_FILE`
- `CHROMA_DB_PATH`
- `SIMILARITY_THRESHOLD`
- `SIMILARITY_TOP_K`
- `REQUEST_TIMEOUT`
- `LLM_TIMEOUT`
- `PAUSE_BETWEEN_NOTES`
- `MAX_RETRIES`
- `RETRY_BACKOFF_SECONDS`
- `LOG_FILE`

## Тести

```bash
python -m unittest discover -s tests -v
```

## Dev tooling (uv + pre-commit)

Встановлення/оновлення dev-інструментів:

```bash
uv add --dev pre-commit ruff mypy detect-secrets nbstripout check-jsonschema bandit interrogate
```

Ініціалізація baseline для секретів:

```bash
uv run detect-secrets scan > .secrets.baseline
```

Активація git-хуків:

```bash
uv run pre-commit install
uv run pre-commit install --hook-type commit-msg
```

Ручний запуск перевірок:

```bash
uv run pre-commit run --all-files
```
