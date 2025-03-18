import subprocess
import json
import time
import yfinance as yf
from randomForest import predict, loadOrTrain
from gatherTrainingData import computeFeatures, WINDOW_SIZE
import pandas as pd
import os

PAST = 7

def getStockData(symbol) -> dict:
    stock = yf.Ticker(symbol)
    hist = stock.history(period="max")

    if len(hist) < WINDOW_SIZE + PAST:
        return None

    hist = hist.tail(WINDOW_SIZE + PAST)
    hist = hist.head(WINDOW_SIZE)

    hist = hist.reset_index()[["Date", "Open", "High", "Low", "Close", "Volume"]]
    hist.columns = ["date", "open", "high", "low", "close", "volume"]
    hist["date"] = pd.to_datetime(hist["date"])

    features = computeFeatures(hist)
    return features

def convertToMatrix():
    with open("data/temp.json", "r") as f, open("data/input.csv", "w") as fMatrix:
        data = json.load(f)

        ordered_keys = [
            "day", "month", "weekday", 
            "close", "volume", "open", "high", "low", 
            "sma5", "ema5", "macd", "signal", 
            "bollingerUpper", "bollingerLower", "rsi14", 
            "stochasticOscillator14k", "stochasticOscillator14d", 
            "roc14", "atr14", "std14", "volumeAverage5"
        ]
        for key in ordered_keys:
            value = data.get(key, [])
            if isinstance(value, list):
                for v in value:
                    fMatrix.write(f"{v},")
            else:
                fMatrix.write(f"{value},")
        fMatrix.write("\n")

def evaluateStock(stock):
    data = getStockData(stock)
    with open("data/temp.json", "w") as f:
        json.dump(data, f)
    convertToMatrix()
    prediction = predict(loadOrTrain())
    return prediction

def main():
    stock = input("Enter stock symbol: ")
    prediction = evaluateStock(stock)
    print("Prediction: ", prediction)

if __name__ == "__main__":
    main()
