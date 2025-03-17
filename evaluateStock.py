import subprocess
import json
import time
import yfinance as yf
from randomForest import predict, loadOrTrain
from gatherTrainingData import computeFeatures
import pandas as pd

def getStockData(symbol) -> dict:
    stock = yf.Ticker(symbol)
    hist = stock.history(period="max")
    hist = hist.reset_index()[["Date", "Open", "High", "Low", "Close", "Volume"]]
    hist.columns = ["date", "open", "high", "low", "close", "volume"]
    hist["date"] = pd.to_datetime(hist["date"])

    # take only the last 205 days and cut off the first 5
    hist = hist.tail(230)
    hist = hist.head(200)

    return computeFeatures(hist)


def convertToMatrix(data: dict):
    dataList = []
    # data is a dictionary with floats and lists of floats
    # we want to convert it to a single list of floats
    for k, v in data.items():
        if isinstance(v, list):
            for i in v:
                dataList.append(i)
        else:
            dataList.append(v)

    with open("data/input.csv", "w") as f:
        for v in dataList:
            f.write(str(v))
            f.write(",")

def evaluateStock(stock):
    data = getStockData(stock)
    convertToMatrix(data)
    prediction = predict(loadOrTrain())
    return prediction

def main():
    stock = input("Enter stock symbol: ")
    prediction = evaluateStock(stock)
    print("Prediction: ", prediction)

if __name__ == "__main__":
    main()
