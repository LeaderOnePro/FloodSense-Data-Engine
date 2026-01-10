"""
Logging configuration using loguru.

Provides a centralized logging setup for the entire project.
"""

import sys
from pathlib import Path
from typing import Optional

from loguru import logger


def get_logger(
    name: str,
    log_file: Optional[Path] = None,
    level: str = "INFO",
    rotation: str = "10 MB",
    retention: str = "7 days",
) -> "logger":
    """
    Get a configured logger instance.

    Args:
        name: Logger name (typically __name__).
        log_file: Optional path to log file.
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL).
        rotation: Log file rotation size.
        retention: Log file retention period.

    Returns:
        Configured loguru logger instance.
    """
    # Remove default handler
    logger.remove()

    # Console handler with color
    logger.add(
        sys.stderr,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
        "<level>{level: <8}</level> | "
        "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - "
        "<level>{message}</level>",
        level=level,
        colorize=True,
    )

    # File handler if specified
    if log_file:
        log_file = Path(log_file)
        log_file.parent.mkdir(parents=True, exist_ok=True)
        logger.add(
            log_file,
            format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | "
            "{name}:{function}:{line} - {message}",
            level=level,
            rotation=rotation,
            retention=retention,
            compression="zip",
        )

    return logger.bind(name=name)


def setup_project_logger(
    log_dir: Path = Path("logs"),
    level: str = "INFO",
) -> None:
    """
    Setup project-wide logging configuration.

    Args:
        log_dir: Directory for log files.
        level: Logging level.
    """
    log_dir.mkdir(parents=True, exist_ok=True)

    # Remove default handler
    logger.remove()

    # Console handler
    logger.add(
        sys.stderr,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
        "<level>{level: <8}</level> | "
        "<cyan>{name}</cyan> - <level>{message}</level>",
        level=level,
        colorize=True,
    )

    # Main log file
    logger.add(
        log_dir / "floodsense.log",
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name} - {message}",
        level=level,
        rotation="10 MB",
        retention="7 days",
        compression="zip",
    )

    # Error log file
    logger.add(
        log_dir / "errors.log",
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | "
        "{name}:{function}:{line} - {message}",
        level="ERROR",
        rotation="5 MB",
        retention="30 days",
        compression="zip",
    )
