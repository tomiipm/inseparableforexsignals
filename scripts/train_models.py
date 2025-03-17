#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import argparse
import logging
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import talib
import warnings

# Ignoruj ostrzeżenia
warnings.filterwarnings('ignore')

# Konfiguracja loggera
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def load_data(file_path):
    """Wczytuje dane z pliku CSV"""
    try:
        # Próbuj różne separatory
        for sep in [',', ';', '\t']:
            try:
                df = pd.read_csv(file_path, sep=sep)
                if len(df.columns) > 1:  # Jeśli udało się poprawnie wczytać
                    logger.info(f"Wczytano dane z {file_path} używając separatora '{sep}'")
                    logger.info(f"Kolumny w danych: {df.columns.tolist()}")
                    logger.info(f"Liczba wierszy: {len(df)}")
                    return df
            except Exception as e:
                continue
        
        # Jeśli żaden separator nie zadziałał
        raise ValueError(f"Nie udało się wczytać danych z {file_path}")
    except Exception as e:
        logger.error(f"Błąd podczas wczytywania danych z {file_path}: {str(e)}")
        return None

def standardize_columns(df):
    """Standaryzuje nazwy kolumn i format danych"""
    # Konwersja nazw kolumn na małe litery
    df.columns = [col.lower() for col in df.columns]
    
    # Mapowanie typowych nazw kolumn
    column_mapping = {
        'date': 'date',
        'time': 'time',
        'timestamp': 'date',
        'datetime': 'date',
        'open': 'open',
        'high': 'high',
        'low': 'low',
        'close': 'close',
        'price': 'close',
        'volume': 'volume',
        'vol': 'volume',
        'vol.': 'volume',
        'adj close': 'close',
        'adj. close': 'close',
        'adjusted close': 'close'
    }
    
    # Sprawdź i zmapuj kolumny
    new_columns = {}
    for col in df.columns:
        for key, value in column_mapping.items():
            if col == key or key in col:
                new_columns[col] = value
                break
    
    # Zmień nazwy kolumn
    if new_columns:
        df = df.rename(columns=new_columns)
    
    # Sprawdź, czy mamy wymagane kolumny
    required_columns = ['open', 'high', 'low', 'close']
    missing_columns = [col for col in required_columns if col not in df.columns]
    
    if missing_columns:
        # Spróbuj znaleźć kolumny z dużej litery
        for col in list(missing_columns):
            cap_col = col.capitalize()
            if cap_col in df.columns:
                df[col] = df[cap_col]
                missing_columns.remove(col)
        
        # Spróbuj znaleźć kolumny zawierające nazwę
        for col in list(missing_columns):
            for df_col in df.columns:
                if col in df_col.lower():
                    df[col] = df[df_col]
                    missing_columns.remove(col)
                    break
    
    # Jeśli nadal brakuje kolumn, zgłoś błąd
    if missing_columns:
        missing_str = ', '.join(missing_columns)
        available_str = ', '.join(df.columns)
        raise ValueError(f"Brakujące kolumny: {missing_str}. Dostępne kolumny: {available_str}")
    
    # Konwersja daty
    if 'date' in df.columns:
        try:
            df['date'] = pd.to_datetime(df['date'])
        except:
            logger.warning("Nie udało się przekonwertować kolumny 'date' na format datetime")
    
    # Sortowanie danych według daty
    if 'date' in df.columns:
        df = df.sort_values('date')
    
    # Konwersja kolumn numerycznych
    for col in ['open', 'high', 'low', 'close', 'volume']:
        if col in df.columns:
            try:
                # Usuń znaki specjalne i zamień przecinki na kropki
                if df[col].dtype == 'object':
                    df[col] = df[col].astype(str).str.replace(',', '.').str.replace('[^\d.]', '', regex=True)
                df[col] = pd.to_numeric(df[col], errors='coerce')
            except:
                logger.warning(f"Nie udało się przekonwertować kolumny '{col}' na format numeryczny")
    
    return df

def add_technical_indicators(df):
    """Dodaje wskaźniki techniczne do danych"""
    # Upewnij się, że mamy wymagane kolumny
    required_columns = ['open', 'high', 'low', 'close']
    for col in required_columns:
        if col not in df.columns:
            raise ValueError(f"Brak kolumny {col} wymaganej do obliczenia wskaźników technicznych")
    
    # Konwersja na tablice numpy dla talib
    open_prices = df['open'].values
    high_prices = df['high'].values
    low_prices = df['low'].values
    close_prices = df['close'].values
    
    # Dodaj wskaźniki techniczne
    try:
        # Średnie kroczące
        df['sma5'] = talib.SMA(close_prices, timeperiod=5)
        df['sma10'] = talib.SMA(close_prices, timeperiod=10)
        df['sma20'] = talib.SMA(close_prices, timeperiod=20)
        df['sma50'] = talib.SMA(close_prices, timeperiod=50)
        
        # Wskaźniki momentum
        df['rsi'] = talib.RSI(close_prices, timeperiod=14)
        df['macd'], df['macd_signal'], df['macd_hist'] = talib.MACD(
            close_prices, fastperiod=12, slowperiod=26, signalperiod=9
        )
        
        # Wskaźniki zmienności
        df['atr'] = talib.ATR(high_prices, low_prices, close_prices, timeperiod=14)
        df['bollinger_upper'], df['bollinger_middle'], df['bollinger_lower'] = talib.BBANDS(
            close_prices, timeperiod=20, nbdevup=2, nbdevdn=2, matype=0
        )
        
        # Wskaźniki trendu
        df['adx'] = talib.ADX(high_prices, low_prices, close_prices, timeperiod=14)
        
        # Wskaźniki oscylacyjne
        df['stoch_k'], df['stoch_d'] = talib.STOCH(
            high_prices, low_prices, close_prices, 
            fastk_period=5, slowk_period=3, slowk_matype=0, 
            slowd_period=3, slowd_matype=0
        )
        
        # Dodatkowe wskaźniki
        df['cci'] = talib.CCI(high_prices, low_prices, close_prices, timeperiod=14)
        df['mfi'] = talib.MFI(high_prices, low_prices, close_prices, 
                             df['volume'].values if 'volume' in df.columns else np.ones_like(close_prices), 
                             timeperiod=14)
        
        # Oblicz zmiany procentowe
        df['pct_change'] = df['close'].pct_change()
        df['pct_change_1d'] = df['close'].pct_change(periods=1)
        df['pct_change_5d'] = df['close'].pct_change(periods=5)
        
        # Oblicz kierunek ruchu (target)
        df['target'] = (df['close'].shift(-1) > df['close']).astype(int)
        
        logger.info(f"Dodano wskaźniki techniczne. Kształt danych: {df.shape}")
        return df
    
    except Exception as e:
        logger.error(f"Błąd podczas dodawania wskaźników technicznych: {str(e)}")
        raise

def prepare_features(df):
    """Przygotowuje cechy i target dla modelu"""
    # Standaryzuj kolumny
    df = standardize_columns(df)
    
    # Dodaj wskaźniki techniczne
    df = add_technical_indicators(df)
    
    # Usuń wiersze z brakującymi wartościami
    df_clean = df.dropna()
    
    # Sprawdź, czy mamy wystarczająco danych
    if len(df_clean) < 10:
        raise ValueError(f"Za mało danych po usunięciu wartości NaN: {len(df_clean)} wierszy")
    
    # Wybierz cechy
    feature_columns = [
        'sma5', 'sma10', 'sma20', 'sma50', 
        'rsi', 'macd', 'macd_signal', 'macd_hist',
        'atr', 'bollinger_upper', 'bollinger_middle', 'bollinger_lower',
        'adx', 'stoch_k', 'stoch_d', 'cci', 'mfi',
        'pct_change', 'pct_change_1d', 'pct_change_5d'
    ]
    
    # Sprawdź, czy wszystkie kolumny istnieją
    available_features = [col for col in feature_columns if col in df_clean.columns]
    
    if len(available_features) < 5:
        raise ValueError(f"Za mało dostępnych cech: {len(available_features)}. Minimum 5 wymagane.")
    
    # Wybierz cechy i target
    X = df_clean[available_features]
    y = df_clean['target']
    
    logger.info(f"Przygotowano cechy. X shape: {X.shape}, y shape: {y.shape}")
    return X, y

def train_models_for_symbol(symbol, data_path, output_dir):
    """Trenuje modele dla danego symbolu"""
    try:
        logger.info(f"Trenowanie modeli dla {symbol}")
        
        # Wczytaj dane
        df = load_data(data_path)
        if df is None or len(df) < 30:
            logger.warning(f"Za mało danych dla {symbol} (minimum 30 wierszy). Pomijanie.")
            return False
        
        # Przygotuj cechy i target
        X, y = prepare_features(df)
        
        # Sprawdź liczbę próbek po przetworzeniu
        if len(X) < 10:
            logger.warning(f"Za mało próbek po przetworzeniu dla {symbol} (minimum 10). Pomijanie.")
            return False
        
        # Dostosuj wielkość zbioru testowego dla małych zbiorów danych
        test_size = min(0.2, max(0.1, 10/len(X)))  # Co najmniej 10 próbek w zbiorze testowym, ale nie mniej niż 10%
        
        # Podział na zbiór treningowy i testowy
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
        
        # Skalowanie cech
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Trenowanie modeli
        models = {
            'random_forest': RandomForestClassifier(n_estimators=100, random_state=42),
            'gradient_boosting': GradientBoostingClassifier(n_estimators=100, random_state=42)
        }
        
        results = {}
        
        for name, model in models.items():
            # Trenowanie modelu
            model.fit(X_train_scaled, y_train)
            
            # Predykcja
            y_pred = model.predict(X_test_scaled)
            
            # Ocena modelu
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred, zero_division=0)
            recall = recall_score(y_test, y_pred, zero_division=0)
            f1 = f1_score(y_test, y_pred, zero_division=0)
            
            results[name] = {
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1': f1
            }
            
            logger.info(f"Model {name} dla {symbol}: Accuracy={accuracy:.4f}, Precision={precision:.4f}, Recall={recall:.4f}, F1={f1:.4f}")
            
            # Zapisz model
            model_dir = os.path.join(output_dir, symbol)
            os.makedirs(model_dir, exist_ok=True)
            
            model_path = os.path.join(model_dir, f"{name}.joblib")
            joblib.dump(model, model_path)
            
            # Zapisz scaler
            scaler_path = os.path.join(model_dir, "scaler.joblib")
            joblib.dump(scaler, scaler_path)
            
            # Zapisz informacje o cechach
            features_path = os.path.join(model_dir, "features.joblib")
            joblib.dump(X.columns.tolist(), features_path)
        
        # Zapisz wyniki
        results_path = os.path.join(model_dir, "results.joblib")
        joblib.dump(results, results_path)
        
        logger.info(f"Zapisano modele i wyniki dla {symbol} w {model_dir}")
        return True
    
    except Exception as e:
        logger.error(f"Błąd podczas trenowania modeli dla {symbol}: {str(e)}")
        return False

def main():
    parser = argparse.ArgumentParser(description="Trenuje modele ML dla sygnałów forex")
    parser.add_argument("--data_dir", type=str, required=True, help="Katalog z danymi historycznymi")
    parser.add_argument("--output_dir", type=str, required=True, help="Katalog wyjściowy dla modeli")
    parser.add_argument("--symbols", type=str, help="Lista symboli do trenowania (oddzielona przecinkami)")
    
    args = parser.parse_args()
    
    # Utwórz katalog wyjściowy, jeśli nie istnieje
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Znajdź pliki danych
    data_files = {}
    
    if args.symbols:
        symbols = args.symbols.split(',')
        for symbol in symbols:
            symbol = symbol.strip()
            # Szukaj pliku dla danego symbolu
            for file in os.listdir(args.data_dir):
                if symbol.lower() in file.lower() and file.endswith(('.csv', '.CSV')):
                    data_files[symbol] = os.path.join(args.data_dir, file)
                    break
    else:
        # Automatycznie wykryj symbole na podstawie nazw plików
        for file in os.listdir(args.data_dir):
            if file.endswith(('.csv', '.CSV')):
                # Wyodrębnij symbol z nazwy pliku
                symbol = file.split('_')[0].split('.')[0]
                data_files[symbol] = os.path.join(args.data_dir, file)
    
    logger.info(f"Znaleziono {len(data_files)} symboli do trenowania: {', '.join(data_files.keys())}")
    
    # Trenuj modele dla każdego symbolu
    successful_symbols = 0
    
    for symbol, data_path in data_files.items():
        logger.info(f"Rozpoczynanie trenowania dla {symbol} z pliku {data_path}")
        if train_models_for_symbol(symbol, data_path, args.output_dir):
            successful_symbols += 1
    
    logger.info(f"Zakończono trenowanie. Pomyślnie wytrenowano modele dla {successful_symbols}/{len(data_files)} symboli.")

if __name__ == "__main__":
    main()