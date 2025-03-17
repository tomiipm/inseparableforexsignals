# -*- coding: utf-8 -*-
import os
import json
import pandas as pd
import numpy as np
from datetime import datetime
import argparse
import logging

# Konfiguracja logowania
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("signal_generator.log", encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def generate_mock_signals (output_dir):
    """Generuje przykładowe sygnały handlowe."""
    try:
        # Lista obsługiwanych par walutowych
        symbols = ["EURUSD", "GBPUSD", "USDJPY", "XAUUSD"]

        all_signals = []

        for symbol in symbols:
            # Losowo wybierz typ sygnału
            signal_type = np.random.choice(["BUY", "SELL", "NEUTRAL"], p=[0.4, 0.4, 0.2])

            # Generuj losową cenę wejściową
            if symbol == "XAUUSD":
                entry_price = round(np.random.uniform(1800, 2200), 2)
            else:
                entry_price = round(np.random.uniform(0.8, 1.5), 5)

            # Oblicz losową zmianę procentową
            percent_change = round(np.random.uniform(-1.5, 1.5), 2)

            # Oblicz poziomy TP i SL
            if signal_type == "BUY":
                tp1 = round(entry_price * (1 + abs(percent_change) * 0.01 * 1.5), 5)
                tp2 = round(entry_price * (1 + abs(percent_change) * 0.01 * 2.5), 5)
                sl = round(entry_price * (1 - abs(percent_change) * 0.01 * 0.8), 5)
            elif signal_type == "SELL":
                tp1 = round(entry_price * (1 - abs(percent_change) * 0.01 * 1.5), 5)
                tp2 = round(entry_price * (1 - abs(percent_change) * 0.01 * 2.5), 5)
                sl = round(entry_price * (1 + abs(percent_change) * 0.01 * 0.8), 5)
            else:
                tp1 = entry_price
                tp2 = entry_price
                sl = entry_price

            # Zaokrąglij wartości w zależności od symbolu
            if symbol == "XAUUSD":
                # Dla złota, zaokrąglij do 2 miejsc po przecinku
                tp1 = round(tp1, 2)
                tp2 = round(tp2, 2)
                sl = round(sl, 2)

            # Oblicz poziom pewności
            confidence = round(min(100, 50 + abs(percent_change) * 10), 2)

            # Utwórz sygnał
            signal = {
                "symbol": symbol,
                "type": signal_type,
                "entryPrice": entry_price,
                "tp1": tp1,
                "tp2": tp2,
                "sl": sl,
                "confidence": confidence,
                "timestamp": datetime.now().isoformat(),
                "percent_change": percent_change
            }

            all_signals.append(signal)

        # Zapisz wszystkie sygnały do pliku JSON
        os.makedirs(output_dir, exist_ok=True)
        output_file = os.path.join(output_dir, f"signals_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(all_signals, f, indent=2)

        logger.info(f"Wygenerowano {len(all_signals)} sygnałów i zapisano do {output_file}")
        return all_signals
    except Exception as e:
        logger.error(f"Błąd podczas generowania sygnałów: {e}")
        return []


def main ():
    parser = argparse.ArgumentParser(description='Generator sygnałów handlowych')
    parser.add_argument('--output_dir', type=str, default='./signals', help='Katalog wyjściowy na sygnały')
    args = parser.parse_args()

    logger.info(f"Rozpoczęcie generowania sygnałów")
    signals = generate_mock_signals(args.output_dir)
    logger.info(f"Zakończono generowanie sygnałów")


if __name__ == "__main__":
    main()