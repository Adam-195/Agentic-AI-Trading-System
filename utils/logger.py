"""
utils/logger.py
Centralised logging setup with structured output and log rotation.
"""

import logging
import os
from logging.handlers import RotatingFileHandler

from config import config


def setup_logging(level: int = logging.INFO) -> None:
    """
    Configure root logger with:
    - Console handler (clean format)
    - Rotating file handler (keeps last 5 x 5MB logs)
    """
    os.makedirs(config.LOG_DIR, exist_ok=True)

    root = logging.getLogger()
    root.setLevel(level)

    # Avoid adding duplicate handlers on repeated calls
    if root.handlers:
        return

    fmt = logging.Formatter(
        "%(asctime)s [%(levelname)-8s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Console
    console = logging.StreamHandler()
    console.setFormatter(fmt)
    root.addHandler(console)

    # Rotating file — 5MB per file, keep 5 backups
    file_handler = RotatingFileHandler(
        filename=os.path.join(config.LOG_DIR, "agent.log"),
        maxBytes=5 * 1024 * 1024,
        backupCount=5,
        encoding="utf-8",
    )
    file_handler.setFormatter(fmt)
    root.addHandler(file_handler)

    # Silence noisy third-party loggers
    for noisy in ["httpx", "httpcore", "urllib3", "yfinance"]:
        logging.getLogger(noisy).setLevel(logging.WARNING)
