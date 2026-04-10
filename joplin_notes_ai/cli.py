import argparse
import sys

from loguru import logger
from pydantic import ValidationError

from joplin_notes_ai.app import JoplinNotesAiApp
from joplin_notes_ai.config import Settings
from joplin_notes_ai.logging_config import configure_logging


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Joplin Notes AI: трансформація і маршрутизація нотаток."
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Запустити без запису змін у Joplin і без upsert у векторне сховище.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Максимальна кількість нотаток для обробки за запуск.",
    )
    parser.add_argument(
        "--reindex-all",
        action="store_true",
        help=(
            "Очистити векторну колекцію і переіндексувати всі наявні нотатки "
            "з розширеними метаданими."
        ),
    )
    parser.add_argument(
        "--organize-tags",
        action="store_true",
        help=(
            "Проаналізувати існуючі теги, побудувати таксономію, "
            "об'єднати близькі теги і замінити їх у нотатках."
        ),
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        settings = Settings()
    except ValidationError as exc:
        required_fields = []
        for error in exc.errors():
            if error.get("type") != "missing":
                continue
            loc = error.get("loc", [])
            if not loc:
                continue
            required_fields.append(str(loc[0]))

        if required_fields:
            fields = ", ".join(sorted(set(required_fields)))
            print(
                "Ошибка конфигурации: отсутствуют обязательные переменные окружения: "
                f"{fields}\n"
                "Создайте файл .env на основе .env.example и заполните значения.",
                file=sys.stderr,
            )
        else:
            print(f"Ошибка конфигурации: {exc}", file=sys.stderr)
        return 1

    configure_logging(settings.log_file)

    try:
        app = JoplinNotesAiApp(settings)
        app.run(
            dry_run=args.dry_run,
            limit=args.limit,
            reindex_all=args.reindex_all,
            organize_tags=args.organize_tags,
        )
        return 0
    except Exception as exc:  # noqa: BLE001 - final process guard
        logger.exception(f"Критичний збій: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
