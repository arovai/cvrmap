import logging
import os
import sys
from typing import Optional


class ColoredFormatter(logging.Formatter):
    """Formatter that applies ANSI colors to log levels."""

    COLORS = {
        "DEBUG": "\033[94m",
        "INFO": "\033[92m",
        "WARNING": "\033[93m",
        "ERROR": "\033[91m",
        "CRITICAL": "\033[91m\033[1m",
    }
    RESET = "\033[0m"

    def __init__(self, fmt: str, datefmt: Optional[str] = None, use_color: bool = True):
        super().__init__(fmt=fmt, datefmt=datefmt)
        self.use_color = use_color

    def format(self, record: logging.LogRecord) -> str:
        original_levelname = record.levelname
        if self.use_color:
            color = self.COLORS.get(record.levelname, "")
            if color:
                record.levelname = f"{color}{record.levelname}{self.RESET}"
        result = super().format(record)
        record.levelname = original_levelname
        return result


def setup_logging(verbose: bool = False, log_file: Optional[str] = None, logger_name: str = "cvrmap") -> logging.Logger:
    """Configure a project logger with colored console output and optional file output."""
    level = logging.DEBUG if verbose else logging.INFO
    logger = logging.getLogger(logger_name)
    logger.setLevel(level)
    logger.handlers.clear()

    color_enabled = sys.stdout.isatty() and os.getenv("NO_COLOR") is None

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(
        ColoredFormatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
            use_color=color_enabled,
        )
    )
    logger.addHandler(console_handler)

    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(level)
        file_handler.setFormatter(
            logging.Formatter(
                "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            )
        )
        logger.addHandler(file_handler)

    logger.propagate = False
    return logger


class Logger:
    """Backward-compatible wrapper around the standardized logging setup."""

    def __init__(self, module_name: str, debug_level: int = 0, log_file: Optional[str] = None):
        verbose = debug_level >= 1
        self._logger = setup_logging(
            verbose=verbose,
            log_file=log_file,
            logger_name=f"cvrmap.{module_name}",
        )

    def debug(self, message, *args, **kwargs):
        self._logger.debug(message, *args, **kwargs)

    def info(self, message, *args, **kwargs):
        self._logger.info(message, *args, **kwargs)

    def warning(self, message, *args, **kwargs):
        self._logger.warning(message, *args, **kwargs)

    def error(self, message, *args, **kwargs):
        self._logger.error(message, *args, **kwargs)

    def critical(self, message, *args, **kwargs):
        self._logger.critical(message, *args, **kwargs)

    def exception(self, message, *args, **kwargs):
        self._logger.exception(message, *args, **kwargs)
