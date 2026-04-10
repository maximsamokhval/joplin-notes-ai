import os
import sys

from loguru import logger


def configure_logging(log_file: str) -> None:
    """Configure loguru outputs only during runtime bootstrap."""
    log_dir = os.path.dirname(log_file)
    if log_dir:
        os.makedirs(log_dir, exist_ok=True)

    logger.remove()
    logger.add(
        sys.stderr,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
        "<level>{level: <8}</level> | <cyan>{message}</cyan>",
    )
    logger.add(log_file, rotation="10 MB", compression="zip")
