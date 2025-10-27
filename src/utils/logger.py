"""
Logger Configuration
Centralized logging setup
"""

import sys
from pathlib import Path
from loguru import logger


def setup_logger(
    log_level: str = "INFO",
    log_file: str = "logs/trading_bot.log",
    rotation: str = "1 day",
    retention: str = "30 days"
):
    """
    Setup centralized logging

    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR)
        log_file: Path to log file
        rotation: When to rotate logs
        retention: How long to keep logs
    """
    # Remove default handler
    logger.remove()

    # Console handler with colors
    logger.add(
        sys.stderr,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan> - <level>{message}</level>",
        level=log_level,
        colorize=True
    )

    # File handler
    log_path = Path(log_file)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    logger.add(
        log_file,
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}",
        level=log_level,
        rotation=rotation,
        retention=retention,
        compression="zip"
    )

    logger.info(f"Logger initialized - Level: {log_level}")

    return logger
