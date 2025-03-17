# notifications.py
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import os
import json
import logging
from dotenv import load_dotenv

# Załaduj zmienne środowiskowe
load_dotenv()

# Konfiguracja logowania
logging.basicConfig(
    level=getattr(logging, os.getenv('LOG_LEVEL', 'INFO')),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("logs/notifications.log", encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def send_email_notification (recipient, subject, message):
    """Wysyła powiadomienie e-mail."""
    try:
        # Pobierz dane SMTP z zmiennych środowiskowych
        smtp_host = os.getenv('SMTP_HOST')
        smtp_port = int(os.getenv('SMTP_PORT', 587))
        smtp_user = os.getenv('SMTP_USER')
        smtp_password = os.getenv('SMTP_PASSWORD')

        if not all([smtp_host, smtp_port, smtp_user, smtp_password]):
            logger.error("Brak konfiguracji SMTP w zmiennych środowiskowych")
            return False

        # Utwórz wiadomość
        msg = MIMEMultipart()
        msg['From'] = smtp_user
        msg['To'] = recipient
        msg['Subject'] = subject

        # Dodaj treść wiadomości
        msg.attach(MIMEText(message, 'html'))

        # Połącz z serwerem SMTP i wyślij wiadomość
        with smtplib.SMTP(smtp_host, smtp_port) as server:
            server.starttls()
            server.login(smtp_user, smtp_password)
            server.send_message(msg)

        logger.info(f"Wysłano powiadomienie e-mail do {recipient}")
        return True
    except Exception as e:
        logger.error(f"Błąd podczas wysyłania powiadomienia e-mail: {e}")
        return False


def notify_about_new_signals (signals_file):
    """Wysyła powiadomienia o nowych sygnałach."""
    try:
        # Sprawdź, czy powiadomienia e-mail są włączone
        if os.getenv('ENABLE_EMAIL_NOTIFICATIONS', 'false').lower() != 'true':
            logger.info("Powiadomienia e-mail są wyłączone")
            return

        # Wczytaj sygnały z pliku
        with open(signals_file, 'r', encoding='utf-8') as f:
            signals = json.load(f)

        if not signals:
            logger.info("Brak sygnałów do powiadomienia")
            return

        # Utwórz treść wiadomości HTML
        html = """
        <html>
        <head>
            <style>
                body { font-family: Arial, sans-serif; }
                .signal { margin-bottom: 20px; padding: 10px; border-radius: 5px; }
                .buy { border-left: 4px solid #4CAF50; background-color: rgba(76, 175, 80, 0.1); }
                .sell { border-left: 4px solid #f44336; background-color: rgba(244, 67, 54, 0.1); }
                .neutral { border-left: 4px solid #9e9e9e; background-color: rgba(158, 158, 158, 0.1); }
                .header { display: flex; justify-content: space-between; }
                .type { padding: 3px 8px; border-radius: 3px; font-weight: bold; font-size: 12px; }
                .buy .type { background-color: rgba(76, 175, 80, 0.2); color: #4CAF50; }
                .sell .type { background-color: rgba(244, 67, 54, 0.2); color: #f44336; }
                .neutral .type { background-color: rgba(158, 158, 158, 0.2); color: #9e9e9e; }
            </style>
        </head>
        <body>
            <h2>Nowe sygnały handlowe</h2>
            <p>Wygenerowano nowe sygnały handlowe:</p>
        """

        for signal in signals:
            html += f"""
            <div class="signal {signal['type'].lower()}">
                <div class="header">
                    <h3>{signal['symbol']}</h3>
                    <span class="type">{signal['type']}</span>
                </div>
                <p>Cena wejścia: <strong>{signal['entryPrice']}</strong></p>
            """

            if signal['type'] != 'NEUTRAL':
                html += f"""
                <p>TP1: <strong>{signal['tp1']}</strong></p>
                <p>TP2: <strong>{signal['tp2']}</strong></p>
                <p>SL: <strong>{signal['sl']}</strong></p>
                """

            html += f"""
                <p>Pewność: <strong>{signal['confidence']}%</strong></p>
                <p>Czas: {signal['timestamp']}</p>
            </div>
            """

        html += """
            <p>Pozdrawiamy,<br>Zespół InseparableFX</p>
        </body>
        </html>
        """

        # Pobierz listę subskrybentów
        subscribers_file = os.path.join('data', "subscribers.json")

        if os.path.exists(subscribers_file):
            # Wczytaj dane z pliku
            with open(subscribers_file, 'r', encoding='utf-8') as f:
                subscribers = json.load(f)
        else:
            subscribers = []

        # Wyślij powiadomienia do wszystkich subskrybentów
        for subscriber in subscribers:
            send_email_notification(
                subscriber,
                "Nowe sygnały handlowe - InseparableFX",
                html
            )

        logger.info(f"Wysłano powiadomienia o nowych sygnałach do {len(subscribers)} subskrybentów")
    except Exception as e:
        logger.error(f"Błąd podczas wysyłania powiadomień o nowych sygnałach: {e}")


if __name__ == "__main__":
    # Przykładowe użycie
    import sys

    if len(sys.argv) > 1:
        signals_file = sys.argv[1]
        notify_about_new_signals(signals_file)
    else:
        logger.error("Nie podano pliku z sygnałami")