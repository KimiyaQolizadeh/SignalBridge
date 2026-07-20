import logging
from logging.config import dictConfig


def configure_logging() -> None:
    """Configure concise, consistently formatted application logs."""

    # Transcripts contain confidential recruiting information. Never include raw
    # transcript content, uploaded file bodies, API keys, or database credentials
    # in log messages or structured fields.
    dictConfig(
        {
            "version": 1,
            "disable_existing_loggers": False,
            "formatters": {
                "default": {
                    "format": (
                        "timestamp=%(asctime)s level=%(levelname)s "
                        "logger=%(name)s message=%(message)s"
                    )
                }
            },
            "handlers": {
                "console": {
                    "class": "logging.StreamHandler",
                    "formatter": "default",
                }
            },
            "root": {"level": "INFO", "handlers": ["console"]},
        }
    )


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)
