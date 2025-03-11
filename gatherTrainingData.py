import os
import json
import time
from yahooquery import Ticker

# List of countries to track stocks from
COUNTRIES = ["US", "DE", "FR", "GB", "PL", "IT"]

# Define the features needed for stock data
FEATURES = ["open", "high", "low", "close", "volume"]

# Ensure directories exist
def ensure_directories():
    for country in COUNTRIES:
        os.makedirs(f"data/{country}", exist_ok=True)

# Load stock symbols from JSON files
def load_symbols():
    all_symbols = {}
    for country in COUNTRIES:
        try:
            with open(f"symbols/{country}_symbols.json", "r") as file:
                all_symbols[country] = json.load(file)  # Load JSON list
        except FileNotFoundError:
            print(f"⚠️ Warning: Missing symbols file for {country}")
            all_symbols[country] = []
    return all_symbols

# Load existing stock data to append new entries
def load_existing_data(file_path):
    if os.path.exists(file_path):
        try:
            with open(file_path, "r") as json_file:
                return json.load(json_file)
        except json.JSONDecodeError:
            print(f"⚠️ Warning: Corrupt JSON file: {file_path}, resetting...")
            return []
    return []

# Fetch historical stock data and save to JSON
def fetch_and_save_stock_data(country, symbol):
    try:
        stock = Ticker(symbol)
        
        # Get historical data - max available data
        historical_data = stock.history(period="max")  # Fetch maximum historical data

        if historical_data.empty:
            print(f"❌ No data for {symbol}")
            return

        # Extract relevant features (open, high, low, close, volume)
        historical_data = historical_data[FEATURES]
        
        # Prepare the data as a list of dictionaries for each day
        historical_data = historical_data.reset_index().to_dict(orient="records")

        # Add timestamp of when the data was fetched
        for entry in historical_data:
            entry["timestamp"] = time.time()

        # Define save path
        save_path = f"data/{country}/{symbol}_data.json"

        # Load existing data to append
        existing_data = load_existing_data(save_path)
        
        # Append new historical data
        existing_data.extend(historical_data)

        # Save to JSON file
        with open(save_path, "w") as json_file:
            json.dump(existing_data, json_file, indent=4)

        print(f"✅ Updated: {save_path}")

    except Exception as e:
        print(f"❌ Error fetching {symbol}: {e}")

# Process all stocks
def process_stocks():
    symbols_dict = load_symbols()
    ensure_directories()

    for country, symbols in symbols_dict.items():
        for symbol in symbols:
            fetch_and_save_stock_data(country, symbol)

# Run script once to gather data
if __name__ == "__main__":
    print("🚀 Fetching historical stock data...")
    process_stocks()
    print("✅ Data collection complete!")
