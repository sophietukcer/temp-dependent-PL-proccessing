"""
logger_config.py
----------------
Module for configuring structured, custom logging across the PL analysis pipeline.
"""

import logging
from pathlib import Path
from typing import Optional, Union


def setup_logger(
    name: str = "pl_analysis",
    log_level: Union[int, str] = "INFO",
    log_file: Optional[Union[str, Path]] = None,
) -> logging.Logger:
    """ Configures and returns a custom logger instance with console and optional file handlers.
    ------
    OUTPUT - logging.Logger: Configured Python logger instance.
    ------
    name - str: Name of the logger instance (default: 'pl_analysis').
    log_level - Union[int, str]: Logging threshold level (e.g., 'DEBUG', 'INFO', 'WARNING').
    log_file - Optional[Union[str, Path]]: File path to write log output (default: None).
    ------
    ValueError: Raised if an invalid log level string is provided.
    """
    logger = logging.getLogger(name)

    if isinstance(log_level, str):
        numeric_level = getattr(logging, log_level.upper(), None)
        if not isinstance(numeric_level, int):
            raise ValueError(f"Invalid log level: {log_level}")
        log_level = numeric_level

    logger.setLevel(log_level)

    if logger.hasHandlers():
        logger.handlers.clear()

    formatter = logging.Formatter(
        "[%(asctime)s] [%(levelname)s] [%(name)s.%(funcName)s]: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    console_handler = logging.StreamHandler()
    console_handler.setLevel(log_level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    if log_file:
        file_path = Path(log_file)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(file_path, encoding="utf-8")
        file_handler.setLevel(log_level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger