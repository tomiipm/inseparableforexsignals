import logging
import sys
import os
from logging.handlers import RotatingFileHandler


def setup_logger (name):
    # Utwórz katalog logs jeśli nie istnieje
    os.makedirs("logs", exist_ok=True)

    # Konfiguracja loggera
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    # Format logów
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    # Handler dla konsoli
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # Handler dla pliku
    file_handler = RotatingFileHandler(
        "logs/app.log",
        maxBytes=10485760,  # 10MB
        backupCount=5
    )
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger