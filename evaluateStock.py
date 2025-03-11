import subprocess
import json
import time
from yahooquery import Ticker

# Function to fetch current stock data using Yahoo Finance
def get_current_stock_data(stock_name):
    try:
        stock = Ticker(stock_name)
        # Fetch the most recent data available
        data = stock.history(period="1d")
        
        # Extract relevant features (you can add more as needed)
        current_data = data[['open', 'high', 'low', 'close', 'volume']].iloc[-1].to_dict()
        current_data['timestamp'] = time.time()  # Add timestamp for tracking

        print(f"✅ Current data for {stock_name}: {current_data}")
        return current_data
    
    except Exception as e:
        print(f"❌ Error fetching current data for {stock_name}: {e}")
        return None


# Function to run the C++ model with stock name as argument
def run_cpp_model(stock_name):
    try:
        # Run the C++ model program with stock_name as an argument
        result = subprocess.run(['./stock_model', stock_name], capture_output=True, text=True)

        # Check if the C++ program ran successfully
        if result.returncode != 0:
            print(f"❌ C++ program error: {result.stderr}")
            return None
        
        # Parse the output from the C++ model (expects 'buy', 'sell', or 'keep')
        decision = result.stdout.strip().lower()
        
        if decision not in ['buy', 'sell', 'keep']:
            print(f"❌ Invalid decision returned from C++ model: {decision}")
            return None
        
        print(f"✅ Model decision for {stock_name}: {decision}")
        return decision
    
    except Exception as e:
        print(f"❌ Error running C++ model for {stock_name}: {e}")
        return None


# Main function to download current stock data and make a decision using the C++ model
def evaluate_stock(stock_name):
    # Step 1: Fetch current stock data
    stock_data = get_current_stock_data(stock_name)
    
    if stock_data is None:
        return "Error: Unable to fetch stock data"
    
    # Step 2: Run the C++ model to get decision
    decision = run_cpp_model(stock_name)
    
    if decision is None:
        return "Error: Unable to run model"
    
    # Step 3: Return the decision from the C++ model (buy, sell, or keep)
    return decision

# Example usage
if __name__ == "__main__":
    stock_name = "AAPL"  # Example stock name (e.g., Apple Inc.)
    decision = evaluate_stock(stock_name)
    print(f"Decision for {stock_name}: {decision}")

