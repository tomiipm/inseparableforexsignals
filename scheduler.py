# scheduler.py
import schedule
import time
import subprocess
import logging
import os
import json
from datetime import datetime
from dotenv import load_dotenv

# Załaduj zmienne środowiskowe
load_dotenv()

# Konfiguracja logowania
logging.basicConfig(
    level=getattr(logging, os.getenv('LOG_LEVEL', 'INFO')),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("logs/scheduler.log", encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def generate_signals ():
    """Uruchamia skrypt generowania sygnałów."""
    try:
        logger.info("Rozpoczęcie generowania sygnałów...")
        result = subprocess.run(["python", "generate_signals.py"], capture_output=True, text=True)

        if result.returncode == 0:
            logger.info("Sygnały wygenerowane pomyślnie")

            # Znajdź najnowszy plik z sygnałami
            signals_dir = os.getenv('SIGNALS_DIR', 'signals')
            signal_files = [f for f in os.listdir(signals_dir) if f.startswith('signals_') and f.endswith('.json')]

            if signal_files:
                # Sortuj pliki według daty utworzenia (od najnowszych)
                signal_files.sort(key=lambda x: os.path.getctime(os.path.join(signals_dir, x)), reverse=True)
                latest_file = os.path.join(signals_dir, signal_files[0])

                # Wyślij powiadomienia o nowych sygnałach
                if os.getenv('ENABLE_EMAIL_NOTIFICATIONS', 'false').lower() == 'true':
                    logger.info("Wysyłanie powiadomień o nowych sygnałach...")
                    subprocess.run(["python", "notifications.py", latest_file], capture_output=True, text=True)
        else:
            logger.error(f"Błąd podczas generowania sygnałów: {result.stderr}")
    except Exception as e:
        logger.error(f"Wyjątek podczas generowania sygnałów: {e}")


def update_settings ():
    """Aktualizuje ustawienia na podstawie pliku settings.json."""
    try:
        settings_file = os.path.join('data', "settings.json")

        if os.path.exists(settings_file):
            # Wczytaj dane z pliku
            with open(settings_file, 'r', encoding='utf-8') as f:
                settings = json.load(f)

            # Sprawdź, czy automatyczne generowanie jest włączone
            auto_generate = settings.get('auto_generate', {})
            if auto_generate.get('enabled', False):
                # Pobierz czas generowania
                schedule_str = auto_generate.get('schedule', '22:00 daily')
                generate_time = schedule_str.split(' ')[0]

                # Wyczyść istniejące zadania
                schedule.clear('generate_signals')

                # Zaplanuj nowe zadanie
                schedule.every().day.at(generate_time).do(generate_signals).tag('generate_signals')
                logger.info(f"Zaplanowano generowanie sygnałów na {generate_time}")
            else:
                # Wyczyść istniejące zadania
                schedule.clear('generate_signals')
                logger.info("Automatyczne generowanie sygnałów jest wyłączone")
    except Exception as e:
        logger.error(f"Błąd podczas aktualizacji ustawień: {e}")


def main ():
    # Pobierz czas generowania z zmiennych środowiskowych lub użyj domyślnego
    generate_time = os.getenv('GENERATE_TIME', '22:00')

    logger.info(f"Scheduler uruchomiony. Generowanie sygnałów zaplanowane na {generate_time}")

    # Zaplanuj generowanie sygnałów codziennie o określonej godzinie
    schedule.every().day.at(generate_time).do(generate_signals).tag('generate_signals')

    # Zaplanuj aktualizację ustawień co godzinę
    schedule.every(1).hour.do(update_settings)

    # Uruchom raz na początku
    update_settings()

    while True:
        schedule.run_pending()
        time.sleep(60)  # Sprawdzaj co minutę


if __name__ == "__main__":
    main()