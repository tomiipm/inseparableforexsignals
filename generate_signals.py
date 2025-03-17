import os
import time
import random
import datetime
import json

def generate_random_signal(signal_id):
    """Generates a random signal data point."""
    timestamp = datetime.datetime.utcnow().isoformat()
    value = random.uniform(0, 100)
    return {"signal_id": signal_id, "timestamp": timestamp, "value": value}

def save_signal_to_file(signal_data, signals_dir):
    """Saves signal data to a JSON file."""
    timestamp = datetime.datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    filename = f"signal_{timestamp}.json"
    filepath = os.path.join(signals_dir, filename)
    with open(filepath, "w") as f:
        json.dump(signal_data, f)
    return filepath

def generate_and_save_signals(num_signals, signals_dir):
    """Generates and saves a specified number of signals."""
    if not os.path.exists(signals_dir):
        os.makedirs(signals_dir)

    for i in range(num_signals):
        signal_id = f"signal_{i+1}"
        signal_data = generate_random_signal(signal_id)
        filepath = save_signal_to_file(signal_data, signals_dir)
        print(f"Saved signal to {filepath}")
        time.sleep(random.uniform(0.1, 1))  # Simulate varying signal generation intervals

def get_latest_signal_file(signals_dir):
    """Retrieves the path to the most recently created signal file."""
    signal_files = [f for f in os.listdir(signals_dir) if f.startswith("signal_") and f.endswith(".json")]

    if not signal_files:
        return None

    # Sortuj pliki według daty utworzenia (od najnowszych)
    signal_files.sort(key=lambda x: os.path.getctime(os.path.join(signals_dir, x)), reverse=True)
    latest_file = os.path.join(signals_dir, signal_files[0])

    return latest_file

if __name__ == "__main__":
    num_signals_to_generate = 5
    signals_directory = "signals"

    generate_and_save_signals(num_signals_to_generate, signals_directory)

    latest_signal = get_latest_signal_file(signals_directory)
    if latest_signal:
        print(f"The latest signal file is: {latest_signal}")
    else:
        print("No signal files found.")

