#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import argparse
import pandas as pd
import numpy as np
from datetime import datetime
import glob
import logging
import traceback
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import joblib
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

# Konfiguracja logowania
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("data_processing.log", encoding='utf-8'),  # Dodano encoding='utf-8'
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

def detect_csv_format(file_path):
    """Wykrywa format pliku CSV (separator, nagłówki)."""
    try:
        # Próba różnych separatorów
        for sep in [',', ';', '\t', '|']:
            try:
                # Sprawdź z nagłówkami
                df = pd.read_csv(file_path, sep=sep)
                if len(df.columns) > 1:
                    return {'sep': sep, 'header': 'infer'}
                
                # Sprawdź bez nagłówków
                df = pd.read_csv(file_path, sep=sep, header=None)
                if len(df.columns) > 1:
                    return {'sep': sep, 'header': None}
            except:
                continue
        
        raise ValueError(f"Nie można wykryć formatu pliku: {file_path}")
    except Exception as e:
        logger.error(f"Błąd podczas wykrywania formatu CSV: {str(e)}")
        return None

def load_and_clean_data(file_path, expected_columns=None):
    """Wczytuje i czyści dane z pliku CSV."""
    try:
        # Wykryj format pliku
        format_info = detect_csv_format(file_path)
        if not format_info:
            raise ValueError(f"Nie można wykryć formatu pliku: {file_path}")
        
        # Wczytaj dane
        df = pd.read_csv(file_path, sep=format_info['sep'], header=format_info['header'])
        
        # Jeśli nie ma nagłówków, nadaj im nazwy
        if format_info['header'] is None:
            if expected_columns and len(df.columns) == len(expected_columns):
                df.columns = expected_columns
            else:
                default_columns = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']
                if len(df.columns) <= len(default_columns):
                    df.columns = default_columns[:len(df.columns)]
                else:
                    df.columns = default_columns + [f'Column_{i+1}' for i in range(len(df.columns) - len(default_columns))]
        
        # Sprawdź i przekształć kolumnę daty
        date_col = df.columns[0]  # Zakładamy, że pierwsza kolumna to data
        try:
            df[date_col] = pd.to_datetime(df[date_col])
        except:
            # Jeśli konwersja się nie powiedzie, spróbuj różnych formatów
            for date_format in ['%Y-%m-%d', '%d-%m-%Y', '%m/%d/%Y', '%d/%m/%Y', '%Y/%m/%d']:
                try:
                    df[date_col] = pd.to_datetime(df[date_col], format=date_format)
                    break
                except:
                    continue
        
        # Konwersja kolumn numerycznych
        numeric_cols = df.columns[1:]  # Zakładamy, że wszystkie kolumny poza datą są numeryczne
        for col in numeric_cols:
            try:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            except:
                logger.warning(f"Nie można przekonwertować kolumny {col} na typ numeryczny")
        
        # Obsługa brakujących wartości
        df = df.dropna()
        
        # Sortowanie po dacie
        df = df.sort_values(by=date_col)
        
        return df
    except Exception as e:
        logger.error(f"Błąd podczas wczytywania pliku {file_path}: {str(e)}")
        logger.error(traceback.format_exc())
        return None

def prepare_data_for_training(df, sequence_length=60, target_column='Close', test_size=0.2):
    """Przygotowuje dane do treningu modelu LSTM."""
    try:
        # Wybierz kolumny do treningu (OHLC)
        feature_columns = ['Open', 'High', 'Low', 'Close']
        if all(col in df.columns for col in feature_columns):
            data = df[feature_columns].values
        else:
            # Jeśli nie ma standardowych kolumn, użyj wszystkich kolumn numerycznych
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            data = df[numeric_cols].values
            feature_columns = numeric_cols
            
            # Upewnij się, że target_column jest w numeric_cols
            if target_column not in numeric_cols and len(numeric_cols) > 0:
                target_column = numeric_cols[-1]  # Użyj ostatniej kolumny jako target
        
        # Normalizacja danych
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(data)
        
        # ZMIANA: Dostosuj długość sekwencji do ilości dostępnych danych
        # Użyj maksymalnie 1/3 dostępnych danych jako długość sekwencji
        adjusted_sequence_length = min(sequence_length, len(scaled_data) // 3)
        if adjusted_sequence_length < 2:
            adjusted_sequence_length = 2  # Minimalna długość sekwencji
        
        logger.info(f"Dostosowano długość sekwencji z {sequence_length} do {adjusted_sequence_length}")
        
        # Przygotowanie sekwencji X i wartości docelowych y
        X, y = [], []
        for i in range(len(scaled_data) - adjusted_sequence_length):
            X.append(scaled_data[i:i+adjusted_sequence_length])
            # Użyj indeksu kolumny docelowej
            target_idx = feature_columns.index(target_column) if target_column in feature_columns else -1
            y.append(scaled_data[i+adjusted_sequence_length, target_idx])
        
        X, y = np.array(X), np.array(y)
        
        # Sprawdź, czy mamy wystarczająco dużo próbek
        if len(X) < 2:
            raise ValueError(f"Za mało próbek do podziału na zbiory treningowy i testowy: {len(X)}")
        
        # Podział na zbiory treningowy i testowy
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, shuffle=False)
        
        return {
            'X_train': X_train,
            'X_test': X_test,
            'y_train': y_train,
            'y_test': y_test,
            'scaler': scaler,
            'feature_columns': feature_columns,
            'sequence_length': adjusted_sequence_length
        }
    except Exception as e:
        logger.error(f"Błąd podczas przygotowywania danych do treningu: {str(e)}")
        logger.error(traceback.format_exc())
        return None

def build_and_train_model(data_dict, epochs=50, batch_size=32):
    """Buduje i trenuje model LSTM."""
    try:
        X_train = data_dict['X_train']
        y_train = data_dict['y_train']
        X_test = data_dict['X_test']
        y_test = data_dict['y_test']
        
        # Dostosuj batch_size do ilości danych
        adjusted_batch_size = min(batch_size, len(X_train) // 2)
        if adjusted_batch_size < 1:
            adjusted_batch_size = 1
        
        # Budowa modelu
        model = Sequential()
        model.add(LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])))
        model.add(Dropout(0.2))
        model.add(LSTM(units=50, return_sequences=False))
        model.add(Dropout(0.2))
        model.add(Dense(units=1))
        
        # Kompilacja modelu
        model.compile(optimizer='adam', loss='mean_squared_error')
        
        # Trening modelu
        history = model.fit(
            X_train, y_train,
            epochs=epochs,
            batch_size=adjusted_batch_size,
            validation_data=(X_test, y_test),
            verbose=1
        )
        
        return model, history
    except Exception as e:
        logger.error(f"Błąd podczas budowy i treningu modelu: {str(e)}")
        logger.error(traceback.format_exc())
        return None, None

def save_model_and_artifacts(model, data_dict, output_dir, symbol):
    """Zapisuje model i artefakty do katalogu wyjściowego."""
    try:
        # Utwórz katalog wyjściowy, jeśli nie istnieje
        os.makedirs(output_dir, exist_ok=True)
        
        # Zapisz model
        model_path = os.path.join(output_dir, f"{symbol}_model.h5")
        model.save(model_path)
        
        # Zapisz scaler
        scaler_path = os.path.join(output_dir, f"{symbol}_scaler.pkl")
        joblib.dump(data_dict['scaler'], scaler_path)
        
        # Zapisz informacje o kolumnach
        columns_path = os.path.join(output_dir, f"{symbol}_columns.txt")
        with open(columns_path, 'w') as f:
            f.write(','.join(data_dict['feature_columns']))
        
        # Zapisz informacje o długości sekwencji
        sequence_path = os.path.join(output_dir, f"{symbol}_sequence_length.txt")
        with open(sequence_path, 'w') as f:
            f.write(str(data_dict['sequence_length']))
        
        logger.info(f"Model i artefakty zapisane w katalogu: {output_dir}")
        return True
    except Exception as e:
        logger.error(f"Błąd podczas zapisywania modelu i artefaktów: {str(e)}")
        logger.error(traceback.format_exc())
        return False

def process_all_files(data_dir, output_dir, sequence_length=60, epochs=50, batch_size=32):
    """Przetwarza wszystkie pliki CSV w katalogu danych."""
    try:
        # Znajdź wszystkie pliki CSV w katalogu
        csv_files = glob.glob(os.path.join(data_dir, "*.csv"))
        
        if not csv_files:
            logger.warning(f"Nie znaleziono plików CSV w katalogu: {data_dir}")
            return False
        
        logger.info(f"Znaleziono {len(csv_files)} plików CSV do przetworzenia")
        
        # Przetwórz każdy plik
        for file_path in csv_files:
            try:
                # Wyodrębnij symbol z nazwy pliku
                symbol = os.path.splitext(os.path.basename(file_path))[0]
                logger.info(f"Przetwarzanie pliku: {file_path} (symbol: {symbol})")
                
                # Wczytaj i wyczyść dane
                df = load_and_clean_data(file_path)
                if df is None or df.empty:
                    logger.warning(f"Pominięto plik {file_path} - brak danych lub błąd wczytywania")
                    continue
                
                logger.info(f"Wczytano {len(df)} wierszy danych dla symbolu {symbol}")
                
                # Przygotuj dane do treningu
                data_dict = prepare_data_for_training(df, sequence_length=sequence_length)
                if data_dict is None:
                    logger.warning(f"Pominięto plik {file_path} - błąd przygotowania danych")
                    continue
                
                # Zbuduj i wytrenuj model
                model, history = build_and_train_model(data_dict, epochs=epochs, batch_size=batch_size)
                if model is None:
                    logger.warning(f"Pominięto plik {file_path} - błąd treningu modelu")
                    continue
                
                # Zapisz model i artefakty
                save_model_and_artifacts(model, data_dict, output_dir, symbol)
                
                logger.info(f"Zakończono przetwarzanie pliku: {file_path}")
            except Exception as e:
                logger.error(f"Błąd podczas przetwarzania pliku {file_path}: {str(e)}")
                logger.error(traceback.format_exc())
                continue
        
        return True
    except Exception as e:
        logger.error(f"Błąd podczas przetwarzania plików: {str(e)}")
        logger.error(traceback.format_exc())
        return False

def main():
    parser = argparse.ArgumentParser(description='Skrypt do przetwarzania danych i treningu modeli LSTM')
    parser.add_argument('--data_dir', type=str, default='./data', help='Katalog z danymi wejściowymi')
    parser.add_argument('--output_dir', type=str, default='./models', help='Katalog wyjściowy dla modeli')
    parser.add_argument('--sequence_length', type=int, default=60, help='Długość sekwencji dla LSTM')
    parser.add_argument('--epochs', type=int, default=50, help='Liczba epok treningu')
    parser.add_argument('--batch_size', type=int, default=32, help='Rozmiar batcha')
    
    args = parser.parse_args()
    
    logger.info("Rozpoczęcie przetwarzania danych i treningu modeli")
    logger.info(f"Katalog danych: {args.data_dir}")
    logger.info(f"Katalog wyjściowy: {args.output_dir}")
    
    success = process_all_files(
        args.data_dir,
        args.output_dir,
        sequence_length=args.sequence_length,
        epochs=args.epochs,
        batch_size=args.batch_size
    )
    
    if success:
        logger.info("Przetwarzanie zakończone sukcesem")
    else:
        logger.error("Przetwarzanie zakończone z błędami")

if __name__ == "__main__":
    main()