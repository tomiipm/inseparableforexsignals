import os
import pandas as pd
import requests
import argparse
import logging
from datetime import datetime

# Konfiguracja logowania
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("fetch_data.log", encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def fetch_alpha_vantage_data(symbol, api_key, output_dir):
    """Pobiera dane z Alpha Vantage API."""
    try:
        # Dla par walutowych
        if symbol.startswith("EUR") or symbol.startswith("GBP") or symbol.startswith("USD"):
            from_currency = symbol[:3]
            to_currency = symbol[3:6]
            
            url = f"https://www.alphavantage.co/query?function=FX_DAILY&from_symbol={from_currency}&to_symbol={to_currency}&outputsize=compact&apikey={api_key}"
            
            logger.info(f"Pobieranie danych dla pary walutowej {from_currency}/{to_currency}")
            response = requests.get(url)
            data = response.json()
            
            if "Error Message" in data:
                logger.error(f"Blad API: {data['Error Message']}")
                return False
            
            if "Time Series FX (Daily)" not in data:
                logger.error(f"Brak danych w odpowiedzi API: {data}")
                return False
            
            # Konwersja danych do DataFrame
            time_series = data["Time Series FX (Daily)"]
            df = pd.DataFrame.from_dict(time_series, orient='index')
            
            # Zmiana nazw kolumn
            df.columns = ['Open', 'High', 'Low', 'Close']
            
            # Konwersja typów danych
            for col in df.columns:
                df[col] = pd.to_numeric(df[col])
            
            # Resetowanie indeksu i dodanie kolumny Date
            df.reset_index(inplace=True)
            df.rename(columns={'index': 'Date'}, inplace=True)
            
            # Sortowanie po dacie (od najstarszej do najnowszej)
            df.sort_values('Date', inplace=True)
            
        # Dla zlota (XAU/USD)
        elif symbol.startswith("XAU"):
            url = f"https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol=GLD&outputsize=compact&apikey={api_key}"
            
            logger.info(f"Pobieranie danych dla zlota (XAU/USD)")
            response = requests.get(url)
            data = response.json()
            
            if "Error Message" in data:
                logger.error(f"Blad API: {data['Error Message']}")
                return False
            
            if "Time Series (Daily)" not in data:
                logger.error(f"Brak danych w odpowiedzi API: {data}")
                return False
            
            # Konwersja danych do DataFrame
            time_series = data["Time Series (Daily)"]
            df = pd.DataFrame.from_dict(time_series, orient='index')
            
            # Zmiana nazw kolumn
            df.columns = ['Open', 'High', 'Low', 'Close', 'Volume']
            
            # Usuniecie kolumny Volume
            df.drop('Volume', axis=1, inplace=True)
            
            # Konwersja typów danych
            for col in df.columns:
                df[col] = pd.to_numeric(df[col])
            
            # Resetowanie indeksu i dodanie kolumny Date
            df.reset_index(inplace=True)
            df.rename(columns={'index': 'Date'}, inplace=True)
            
            # Sortowanie po dacie (od najstarszej do najnowszej)
            df.sort_values('Date', inplace=True)
            
            # Przeskalowanie cen GLD do XAU/USD (przyblizenie)
            for col in ['Open', 'High', 'Low', 'Close']:
                df[col] = df[col] * 18.5  # Przyblizony wspólczynnik konwersji
        else:
            logger.error(f"Nieobslugiwany symbol: {symbol}")
            return False
        
        # Zapisz dane do pliku CSV
        output_file = os.path.join(output_dir, f"{symbol}.csv")
        df.to_csv(output_file, index=False)
        logger.info(f"Zapisano {len(df)} wierszy danych do {output_file}")
        
        return True
    except Exception as e:
        logger.error(f"Blad podczas pobierania danych dla {symbol}: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description='Pobieranie aktualnych danych rynkowych')
    parser.add_argument('--api_key', type=str, help='Klucz API Alpha Vantage')
    parser.add_argument('--output_dir', type=str, default='./data/current', help='Katalog wyjsciowy')
    args = parser.parse_args()
    
    # Sprawdz, czy podano klucz API
    api_key = args.api_key or os.environ.get('ALPHA_VANTAGE_API_KEY')
    if not api_key:
        logger.error("Brak klucza API Alpha Vantage. Uzyj --api_key lub ustaw zmienna srodowiskowa ALPHA_VANTAGE_API_KEY")
        return
    
    # Utwórz katalog wyjsciowy, jesli nie istnieje
    os.makedirs(args.output_dir, exist_ok=True)
    
    logger.info(f"Rozpoczecie pobierania danych")
    logger.info(f"Katalog wyjsciowy: {args.output_dir}")
    
    # Lista symboli do pobrania
    symbols = ["EURUSD_historical", "GBPUSD_historical", "XAU_historical"]
    
    success_count = 0
    
    for symbol in symbols:
        if fetch_alpha_vantage_data(symbol, api_key, args.output_dir):
            success_count += 1
    
    logger.info(f"Zakonczono pobieranie danych. Pobrano dane dla {success_count}/{len(symbols)} symboli")

if __name__ == "__main__":
    main()