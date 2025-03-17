import os
import numpy as np
import pandas as pd
import tensorflow as tf
import joblib
import json
import argparse
import logging
from datetime import datetime

# Konfiguracja logowania
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("predict.log", encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def load_model_artifacts(model_dir, symbol):
    """Laduje model i artefakty dla danego symbolu."""
    try:
        # Ladowanie modelu
        model_path = os.path.join(model_dir, f"{symbol}_model.h5")
        model = tf.keras.models.load_model(model_path)
        
        # Ladowanie skalera
        scaler_path = os.path.join(model_dir, f"{symbol}_scaler.pkl")
        scaler = joblib.load(scaler_path)
        
        # Ladowanie metadanych
        metadata_path = os.path.join(model_dir, f"{symbol}_metadata.json")
        with open(metadata_path, 'r', encoding='utf-8') as f:
            metadata = json.load(f)
        
        return model, scaler, metadata
    except Exception as e:
        logger.error(f"Blad podczas ladowania modelu dla {symbol}: {e}")
        return None, None, None

def prepare_data(data, metadata, scaler):
    """Przygotowuje dane do predykcji."""
    try:
        # Wybierz tylko kolumny uzywane podczas treningu
        feature_columns = metadata.get('feature_columns', [])
        if not feature_columns:
            logger.error("Brak informacji o kolumnach cech w metadanych")
            return None
        
        # Upewnij sie, ze wszystkie wymagane kolumny sa dostepne
        for col in feature_columns:
            if col not in data.columns:
                logger.error(f"Brak kolumny {col} w danych wejsciowych")
                return None
        
        # Wybierz tylko potrzebne kolumny
        data = data[feature_columns]
        
        # Normalizacja danych
        data_scaled = scaler.transform(data)
        
        # Pobierz dlugosc sekwencji z metadanych
        sequence_length = metadata.get('sequence_length', 10)
        
        # Przygotuj dane w formacie sekwencji
        if len(data_scaled) < sequence_length:
            logger.error(f"Za malo danych do utworzenia sekwencji (wymagane: {sequence_length}, dostepne: {len(data_scaled)})")
            return None
        
        # Uzyj ostatnich sequence_length punktów danych
        X = data_scaled[-sequence_length:].reshape(1, sequence_length, len(feature_columns))
        
        return X
    except Exception as e:
        logger.error(f"Blad podczas przygotowywania danych: {e}")
        return None

def generate_signal(prediction, current_price, symbol):
    """Generuje sygnal handlowy na podstawie predykcji."""
    try:
        predicted_price = prediction[0][0]  # Zakladamy, ze model przewiduje cene zamkniecia
        
        # Oblicz procentowa zmiane
        percent_change = ((predicted_price - current_price) / current_price) * 100
        
        # Ustaw progi dla sygnalów
        buy_threshold = 0.2  # 0.2% wzrostu
        sell_threshold = -0.2  # 0.2% spadku
        
        # Okresl typ sygnalu
        if percent_change > buy_threshold:
            signal_type = "BUY"
            confidence = min(100, 50 + percent_change * 10)  # Wyzsza zmiana = wyzsza pewnosc
        elif percent_change < sell_threshold:
            signal_type = "SELL"
            confidence = min(100, 50 + abs(percent_change) * 10)
        else:
            signal_type = "NEUTRAL"
            confidence = 50
        
        # Oblicz poziomy TP i SL
        if signal_type == "BUY":
            tp1 = current_price * (1 + abs(percent_change) * 1.5)
            tp2 = current_price * (1 + abs(percent_change) * 2.5)
            sl = current_price * (1 - abs(percent_change) * 0.8)
        elif signal_type == "SELL":
            tp1 = current_price * (1 - abs(percent_change) * 1.5)
            tp2 = current_price * (1 - abs(percent_change) * 2.5)
            sl = current_price * (1 + abs(percent_change) * 0.8)
        else:
            tp1 = current_price
            tp2 = current_price
            sl = current_price
        
        # Zaokraglij wartosci w zaleznosci od symbolu
        if symbol.startswith("XAU"):
            # Dla zlota, zaokraglij do 2 miejsc po przecinku
            tp1 = round(tp1, 2)
            tp2 = round(tp2, 2)
            sl = round(sl, 2)
        else:
            # Dla par walutowych, zaokraglij do 5 miejsc po przecinku
            tp1 = round(tp1, 5)
            tp2 = round(tp2, 5)
            sl = round(sl, 5)
        
        # Utwórz sygnal
        signal = {
            "symbol": symbol,
            "type": signal_type,
            "entryPrice": round(current_price, 5),
            "tp1": tp1,
            "tp2": tp2,
            "sl": sl,
            "confidence": round(confidence, 2),
            "timestamp": datetime.now().isoformat(),
            "predicted_price": round(predicted_price, 5),
            "percent_change": round(percent_change, 2)
        }
        
        return signal
    except Exception as e:
        logger.error(f"Blad podczas generowania sygnalu: {e}")
        return None

def predict_for_symbol(model_dir, data_file, symbol):
    """Wykonuje predykcje dla danego symbolu."""
    logger.info(f"Rozpoczecie predykcji dla symbolu: {symbol}")
    
    # Ladowanie modelu i artefaktów
    model, scaler, metadata = load_model_artifacts(model_dir, symbol)
    if model is None or scaler is None or metadata is None:
        logger.error(f"Nie mozna zaladowac modelu lub artefaktów dla {symbol}")
        return None
    
    try:
        # Wczytaj dane
        data = pd.read_csv(data_file)
        logger.info(f"Wczytano {len(data)} wierszy danych dla {symbol}")
        
        # Przygotuj dane do predykcji
        X = prepare_data(data, metadata, scaler)
        if X is None:
            logger.error(f"Nie mozna przygotowac danych do predykcji dla {symbol}")
            return None
        
        # Wykonaj predykcje
        prediction = model.predict(X)
        logger.info(f"Wykonano predykcje dla {symbol}")
        
        # Pobierz aktualna cene (ostatnia cena zamkniecia)
        current_price = data['Close'].iloc[-1]
        
        # Generuj sygnal
        signal = generate_signal(prediction, current_price, symbol)
        logger.info(f"Wygenerowano sygnal dla {symbol}: {signal}")
        
        return signal
    except Exception as e:
        logger.error(f"Blad podczas predykcji dla {symbol}: {e}")
        return None

def main():
    parser = argparse.ArgumentParser(description='Generowanie prognoz i sygnalów handlowych')
    parser.add_argument('--model_dir', type=str, default='./models', help='Katalog z modelami')
    parser.add_argument('--data_dir', type=str, default='./data/current', help='Katalog z aktualnymi danymi')
    parser.add_argument('--output_dir', type=str, default='./signals', help='Katalog wyjsciowy na sygnaly')
    args = parser.parse_args()
    
    # Utwórz katalog wyjsciowy, jesli nie istnieje
    os.makedirs(args.output_dir, exist_ok=True)
    
    logger.info(f"Rozpoczecie generowania prognoz i sygnalów")
    logger.info(f"Katalog modeli: {args.model_dir}")
    logger.info(f"Katalog danych: {args.data_dir}")
    logger.info(f"Katalog wyjsciowy: {args.output_dir}")
    
    # Lista obslugiwanych symboli
    symbols = ["EURUSD_historical", "GBPUSD_historical", "XAU_historical"]
    
    all_signals = []
    
    for symbol in symbols:
        # Sprawdz, czy model istnieje
        model_path = os.path.join(args.model_dir, f"{symbol}_model.h5")
        if not os.path.exists(model_path):
            logger.warning(f"Brak modelu dla {symbol}, pomijanie")
            continue
        
        # Sprawdz, czy dane istnieja
        data_file = os.path.join(args.data_dir, f"{symbol}.csv")
        if not os.path.exists(data_file):
            logger.warning(f"Brak danych dla {symbol}, pomijanie")
            continue
        
        # Wykonaj predykcje
        signal = predict_for_symbol(args.model_dir, data_file, symbol)
        if signal:
            all_signals.append(signal)
    
    # Zapisz wszystkie sygnaly do pliku JSON
    if all_signals:
        output_file = os.path.join(args.output_dir, f"signals_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(all_signals, f, indent=2)
        logger.info(f"Zapisano {len(all_signals)} sygnalów do pliku {output_file}")
    else:
        logger.warning("Nie wygenerowano zadnych sygnalów")
    
    logger.info("Zakonczono generowanie prognoz i sygnalów")

if __name__ == "__main__":
    main()