# SPDX-License-Identifier: LGPL-3.0-or-later
# Copyright (C) 2025-2026 Kris Kirby, KE4AHR

"""
pyax25_22.utils.logging.py

Centralized logging configuration for PyAX25_22.

Provides a consistent, structured logging setup across all modules:
- Hierarchical loggers (pyax25_22.core, pyax25_22.interfaces, etc.)
- Configurable log level via environment variable PYAX25_LOG_LEVEL
- Thread-safe formatter with timestamp, logger name, level, and message
- Console output with color support (if colorlog available)
- File handler optional via PYAX25_LOG_FILE

Usage:
    from pyax25_22.utils.logging import get_logger
    logger = get_logger(__name__)
    logger.info("Something happened")
"""

from __future__ import annotations

import logging
import os
import sys
from typing import Optional

# Optional color support
try:
    import colorlog
    HAS_COLORLOG = True
except ImportError:
    HAS_COLORLOG = False

# Global root logger name
ROOT_LOGGER_NAME = "pyax25_22"

# Default log level
DEFAULT_LOG_LEVEL = logging.INFO

# Environment variables
ENV_LOG_LEVEL = "PYAX25_LOG_LEVEL"
ENV_LOG_FILE = "PYAX25_LOG_FILE"


def _get_log_level() -> int:
    """
    Determine log level from environment or default.

    Supported values: DEBUG, INFO, WARNING, ERROR, CRITICAL (case-insensitive)
    """
    level_str = os.getenv(ENV_LOG_LEVEL, "").upper()
    level_map = {
        "DEBUG": logging.DEBUG,
        "INFO": logging.INFO,
        "WARNING": logging.WARNING,
        "ERROR": logging.ERROR,
        "CRITICAL": logging.CRITICAL,
    }
    return level_map.get(level_str, DEFAULT_LOG_LEVEL)


def _create_formatter(color: bool = True) -> logging.Formatter:
    """
    Create consistent log formatter.

    Args:
        color: Use colored output if colorlog available

    Returns:
        Configured formatter
    """
    if color and HAS_COLORLOG:
        return colorlog.ColoredFormatter(
            fmt="%(asctime)s %(log_color)s%(levelname)-8s%(reset)s %(name)s: %(message)s",
            datefmt="%H:%M:%S",
            log_colors={
                'DEBUG': 'cyan',
                'INFO': 'green',
                'WARNING': 'yellow',
                'ERROR': 'red',
                'CRITICAL': 'bold_red',
            },
        )
    else:
        return logging.Formatter(
            fmt="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
            datefmt="%H:%M:%S",
        )


def _configure_logging() -> None:
    """
    Configure root logger and handlers once.

    Called automatically on first get_logger() call.
    """
    root_logger = logging.getLogger(ROOT_LOGGER_NAME)
    root_logger.setLevel(_get_log_level())

    # Avoid adding handlers multiple times
    if root_logger.handlers:
        return

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(_create_formatter(color=HAS_COLORLOG))
    root_logger.addHandler(console_handler)

    # Optional file handler
    log_file = os.getenv(ENV_LOG_FILE)
    if log_file:
        try:
            file_handler = logging.FileHandler(log_file)
            file_handler.setFormatter(_create_formatter(color=False))
            root_logger.addHandler(file_handler)
            root_logger.info(f"Logging to file: {log_file}")
        except Exception as e:
            root_logger.warning(f"Could not open log file {log_file}: {e}")

    root_logger.info(f"Logging configured at level {logging.getLevelName(root_logger.level)}")


def get_logger(name: Optional[str] = None) -> logging.Logger:
    """
    Get a configured logger instance.

    Args:
        name: Logger name (use __name__ in modules)

    Returns:
        Configured logger
    """
    # Ensure root configuration
    _configure_logging()

    logger_name = f"{ROOT_LOGGER_NAME}.{name}" if name else ROOT_LOGGER_NAME
    logger = logging.getLogger(logger_name)

    # Propagate to root
    logger.propagate = True

    return logger


# Convenience loggers for common components
core_logger = get_logger("core")
interfaces_logger = get_logger("interfaces")
transport_logger = get_logger("interfaces.transport")
kiss_logger = get_logger("interfaces.kiss")
agwpe_logger = get_logger("interfaces.agwpe")


# Optional: Debug helper for development
def enable_debug_logging() -> None:
    """Enable DEBUG level logging (convenience function)."""
    logging.getLogger(ROOT_LOGGER_NAME).setLevel(logging.DEBUG)
    for handler in logging.getLogger(ROOT_LOGGER_NAME).handlers:
        handler.setLevel(logging.DEBUG)
    logger.info("Debug logging enabled globally")
