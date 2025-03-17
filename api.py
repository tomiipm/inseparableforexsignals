# api.py
from fastapi import FastAPI, HTTPException, Depends, Header
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import os
import json
import glob
from datetime import datetime, timedelta
import logging
from dotenv import load_dotenv
import subprocess

# Załaduj zmienne środowiskowe
load_dotenv()

# Konfiguracja logowania
logging.basicConfig(
    level=getattr(logging, os.getenv('LOG_LEVEL', 'INFO')),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("logs/api.log", encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Inicjalizacja aplikacji FastAPI
app = FastAPI(title="InseparableFX API", description="API do generowania sygnałów handlowych Forex")

# Dodaj obsługę CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # W produkcji zmień na konkretną domenę
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Katalogi
SIGNALS_DIR = os.getenv('SIGNALS_DIR', 'signals')
DATA_DIR = os.getenv('DATA_DIR', 'data')
MODELS_DIR = os.getenv('MODELS_DIR', 'models')


# Funkcja do weryfikacji klucza API
async def verify_api_key (x_api_key: str = Header(None)):
    api_key = os.getenv('API_KEY')
    if not api_key or x_api_key != api_key:
        raise HTTPException(status_code=401, detail="Nieprawidłowy klucz API")
    return x_api_key


# Model danych dla subskrypcji
class SubscriptionRequest(BaseModel):
    email: str


# Model danych dla ustawień
class SettingsUpdate(BaseModel):
    thresholds: dict = None
    confidence_levels: dict = None
    auto_generate: dict = None
    auto_trade: dict = None


# Endpoint do pobierania najnowszych sygnałów
@app.get("/signals")
async def get_signals ():
    """Pobiera najnowsze sygnały handlowe."""
    try:
        # Znajdź najnowszy plik z sygnałami
        signal_files = glob.glob(os.path.join(SIGNALS_DIR, "signals_*.json"))
        if not signal_files:
            return []

        # Sortuj pliki według daty utworzenia (od najnowszych)
        signal_files.sort(key=os.path.getctime, reverse=True)
        latest_file = signal_files[0]

        # Wczytaj sygnały z pliku
        with open(latest_file, 'r', encoding='utf-8') as f:
            signals = json.load(f)

        return signals
    except Exception as e:
        logger.error(f"Błąd podczas pobierania sygnałów: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Endpoint do generowania nowych sygnałów
@app.post("/generate", dependencies=[Depends(verify_api_key)])
async def generate_signals ():
    """Generuje nowe sygnały handlowe."""
    try:
        # Uruchom skrypt generowania sygnałów
        result = subprocess.run(["python", "generate_signals.py"], capture_output=True, text=True)

        if result.returncode != 0:
            logger.error(f"Błąd podczas generowania sygnałów: {result.stderr}")
            raise HTTPException(status_code=500, detail="Nie udało się wygenerować sygnałów")

        # Pobierz wygenerowane sygnały
        return await get_signals()
    except Exception as e:
        logger.error(f"Błąd podczas generowania sygnałów: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Endpoint do pobierania historii sygnałów
@app.get("/history")
async def get_signals_history (start_date: str = None, end_date: str = None, symbol: str = None):
    """Pobiera historię sygnałów handlowych."""
    try:
        # Znajdź wszystkie pliki z sygnałami
        signal_files = glob.glob(os.path.join(SIGNALS_DIR, "signals_*.json"))
        if not signal_files:
            return []

        # Sortuj pliki według daty utworzenia (od najnowszych)
        signal_files.sort(key=os.path.getctime, reverse=True)

        # Konwertuj daty, jeśli podano
        start_date_obj = None
        end_date_obj = None

        if start_date:
            start_date_obj = datetime.strptime(start_date, "%Y-%m-%d")

        if end_date:
            end_date_obj = datetime.strptime(end_date, "%Y-%m-%d")
            # Dodaj jeden dzień, aby uwzględnić cały dzień końcowy
            end_date_obj = end_date_obj + timedelta(days=1)

        history = []

        for file in signal_files:
            # Pobierz datę z pliku
            file_date = datetime.fromtimestamp(os.path.getctime(file))

            # Sprawdź, czy data mieści się w zakresie
            if start_date_obj and file_date < start_date_obj:
                continue

            if end_date_obj and file_date > end_date_obj:
                continue

            # Wczytaj sygnały z pliku
            with open(file, 'r', encoding='utf-8') as f:
                signals = json.load(f)

            # Filtruj sygnały według symbolu, jeśli podano
            if symbol:
                signals = [s for s in signals if s["symbol"] == symbol]

            if signals:
                history.append({
                    "date": file_date.isoformat(),
                    "signals": signals
                })

        return history
    except Exception as e:
        logger.error(f"Błąd podczas pobierania historii sygnałów: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Endpoint do pobierania statystyk wydajności
@app.get("/performance")
async def get_signals_performance ():
    """Pobiera statystyki skuteczności sygnałów."""
    try:
        # Sprawdź, czy istnieje plik z wydajnością
        performance_file = os.path.join(DATA_DIR, "performance.json")

        if os.path.exists(performance_file):
            # Wczytaj dane z pliku
            with open(performance_file, 'r', encoding='utf-8') as f:
                performance = json.load(f)
        else:
            # Zwróć przykładowe dane
            performance = {
                "overall": {
                    "total": 100,
                    "successful": 65,
                    "success_rate": 65.0
                },
                "by_symbol": {
                    "EURUSD": {
                        "total": 30,
                        "successful": 20,
                        "success_rate": 66.7
                    },
                    "GBPUSD": {
                        "total": 25,
                        "successful": 15,
                        "success_rate": 60.0
                    },
                    "USDJPY": {
                        "total": 25,
                        "successful": 18,
                        "success_rate": 72.0
                    },
                    "XAUUSD": {
                        "total": 20,
                        "successful": 12,
                        "success_rate": 60.0
                    }
                },
                "by_type": {
                    "BUY": {
                        "total": 50,
                        "successful": 35,
                        "success_rate": 70.0
                    },
                    "SELL": {
                        "total": 40,
                        "successful": 25,
                        "success_rate": 62.5
                    },
                    "NEUTRAL": {
                        "total": 10,
                        "successful": 5,
                        "success_rate": 50.0
                    }
                }
            }

        return performance
    except Exception as e:
        logger.error(f"Błąd podczas pobierania statystyk skuteczności: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Endpoint do pobierania ustawień
@app.get("/settings", dependencies=[Depends(verify_api_key)])
async def get_settings ():
    """Pobiera aktualne ustawienia systemu."""
    try:
        # Sprawdź, czy istnieje plik z ustawieniami
        settings_file = os.path.join(DATA_DIR, "settings.json")

        if os.path.exists(settings_file):
            # Wczytaj dane z pliku
            with open(settings_file, 'r', encoding='utf-8') as f:
                settings = json.load(f)
        else:
            # Zwróć domyślne ustawienia
            settings = {
                "thresholds": {
                    "buy": float(os.getenv('BUY_THRESHOLD', 0.2)),
                    "sell": float(os.getenv('SELL_THRESHOLD', -0.2))
                },
                "confidence_levels": {
                    "high": float(os.getenv('HIGH_CONFIDENCE', 70)),
                    "medium": float(os.getenv('MEDIUM_CONFIDENCE', 50)),
                    "low": float(os.getenv('LOW_CONFIDENCE', 30))
                },
                "symbols": ["EURUSD", "GBPUSD", "USDJPY", "XAUUSD"],
                "auto_generate": {
                    "enabled": os.getenv('AUTO_GENERATE', 'false').lower() == 'true',
                    "schedule": os.getenv('GENERATE_TIME', '22:00') + " daily"
                },
                "auto_trade": {
                    "enabled": os.getenv('AUTO_TRADE', 'false').lower() == 'true'
                }
            }

            # Zapisz domyślne ustawienia do pliku
            with open(settings_file, 'w', encoding='utf-8') as f:
                json.dump(settings, f, indent=2)

        return settings
    except Exception as e:
        logger.error(f"Błąd podczas pobierania ustawień: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Endpoint do aktualizacji ustawień
@app.post("/settings", dependencies=[Depends(verify_api_key)])
async def update_settings (settings_update: SettingsUpdate):
    """Aktualizuje ustawienia systemu."""
    try:
        # Pobierz aktualne ustawienia
        current_settings = await get_settings()

        # Aktualizuj ustawienia
        if settings_update.thresholds:
            current_settings["thresholds"].update(settings_update.thresholds)

        if settings_update.confidence_levels:
            current_settings["confidence_levels"].update(settings_update.confidence_levels)

        if settings_update.auto_generate:
            current_settings["auto_generate"].update(settings_update.auto_generate)

        if settings_update.auto_trade:
            current_settings["auto_trade"].update(settings_update.auto_trade)

        # Zapisz zaktualizowane ustawienia
        settings_file = os.path.join(DATA_DIR, "settings.json")
        with open(settings_file, 'w', encoding='utf-8') as f:
            json.dump(current_settings, f, indent=2)

        return current_settings
    except Exception as e:
        logger.error(f"Błąd podczas aktualizacji ustawień: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Endpoint do subskrypcji powiadomień
@app.post("/subscribe")
async def subscribe_to_notifications (subscription: SubscriptionRequest):
    """Zapisuje użytkownika do powiadomień e-mail."""
    try:
        email = subscription.email

        # Sprawdź, czy istnieje plik z subskrybentami
        subscribers_file = os.path.join(DATA_DIR, "subscribers.json")

        if os.path.exists(subscribers_file):
            # Wczytaj dane z pliku
            with open(subscribers_file, 'r', encoding='utf-8') as f:
                subscribers = json.load(f)
        else:
            subscribers = []

        # Sprawdź, czy e-mail już istnieje
        if email not in subscribers:
            subscribers.append(email)

            # Zapisz zaktualizowaną listę subskrybentów
            with open(subscribers_file, 'w', encoding='utf-8') as f:
                json.dump(subscribers, f, indent=2)

        return {"message": f"Zapisano {email} do powiadomień", "status": "success"}
    except Exception as e:
        logger.error(f"Błąd podczas zapisywania do powiadomień: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Uruchomienie aplikacji
if __name__ == "__main__":
    import uvicorn

    port = int(os.getenv('PORT', 8000))
    uvicorn.run("api:app", host="0.0.0.0", port=port, reload=True)