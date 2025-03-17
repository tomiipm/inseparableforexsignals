# start.py
import subprocess
import os
import sys
import time
import signal
import logging
from dotenv import load_dotenv

# Załaduj zmienne środowiskowe
load_dotenv()

# Konfiguracja logowania
logging.basicConfig(
    level=getattr(logging, os.getenv('LOG_LEVEL', 'INFO')),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("logs/start.log", encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Lista procesów
processes = []


def signal_handler (sig, frame):
    """Obsługuje sygnały zakończenia."""
    logger.info("Otrzymano sygnał zakończenia. Zatrzymywanie procesów...")

    for process in processes:
        if process.poll() is None:  # Jeśli proces nadal działa
            logger.info(f"Zatrzymywanie procesu {process.args}")
            process.terminate()

    logger.info("Wszystkie procesy zatrzymane. Wyjście.")
    sys.exit(0)


def start_api ():
    """Uruchamia API."""
    try:
        logger.info("Uruchamianie API...")
        api_process = subprocess.Popen(["python", "api.py"])
        processes.append(api_process)
        logger.info("API uruchomione")
    except Exception as e:
        logger.error(f"Błąd podczas uruchamiania API: {e}")


def start_scheduler ():
    """Uruchamia scheduler."""
    try:
        # Sprawdź, czy automatyczne generowanie jest włączone
        if os.getenv('AUTO_GENERATE', 'false').lower() == 'true':
            logger.info("Uruchamianie schedulera...")
            scheduler_process = subprocess.Popen(["python", "scheduler.py"])
            processes.append(scheduler_process)
            logger.info("Scheduler uruchomiony")
        else:
            logger.info("Automatyczne generowanie jest wyłączone. Scheduler nie zostanie uruchomiony.")
    except Exception as e:
        logger.error(f"Błąd podczas uruchamiania schedulera: {e}")


def main ():
    # Zarejestruj obsługę sygnałów
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    # Utwórz katalogi, jeśli nie istnieją
    os.makedirs(os.getenv('SIGNALS_DIR', 'signals'), exist_ok=True)
    os.makedirs(os.getenv('DATA_DIR', 'data'), exist_ok=True)
    os.makedirs(os.getenv('MODELS_DIR', 'models'), exist_ok=True)
    os.makedirs('logs', exist_ok=True)

    # Uruchom API
    start_api()

    # Uruchom scheduler
    start_scheduler()

    # Generuj sygnały na początku
    logger.info("Generowanie początkowych sygnałów...")
    subprocess.run(["python", "generate_signals.py"], capture_output=True, text=True)

    logger.info("System uruchomiony. Naciśnij Ctrl+C, aby zakończyć.")

    # Czekaj na zakończenie
    while True:
        # Sprawdź, czy wszystkie procesy nadal działają
        for i, process in enumerate(processes):
            if process.poll() is not None:  # Jeśli proces zakończył działanie
                logger.warning(
                    f"Proces {process.args} zakończył działanie z kodem {process.returncode}. Ponowne uruchomienie...")

                # Uruchom ponownie proces
                if "api.py" in process.args:
                    start_api()
                elif "scheduler.py" in process.args:
                    start_scheduler()

                # Usuń zakończony proces z listy
                processes.pop(i)
                break

        time.sleep(5)  # Sprawdzaj co 5 sekund


if __name__ == "__main__":
    main()