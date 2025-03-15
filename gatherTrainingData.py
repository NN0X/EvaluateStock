import yfinance as yf
import json
import os
import pandas as pd
from datetime import datetime
import numpy as np
import tqdm

PERCENTAGE_INC_TO_BUY = 5 # in percent
PERCENTAGE_DEC_TO_SELL = 2 # in percent

DATASET_SIZE = 100000

COUNTRIES = ["BE", "CH", "DE", "DK", "ES", "FI", "FR", "IT", "NL", "NO", "PL", "PT", "SE", "UK", "US"]

EX_SUFFIXES = {
    "BE": [".BR"],  # Belgium - Euronext Brussels
    "CH": [".SW"],  # Switzerland - SIX Swiss Exchange
    "DE": [".DE", ".F", ".XETRA"],  # Germany - Deutsche Börse, Frankfurt
    "DK": [".CO"],  # Denmark - Copenhagen
    "ES": [".MC"],  # Spain - Madrid
    "FI": [".HE"],  # Finland - Helsinki
    "FR": [".PA"],  # France - Paris
    "IT": [".MI"],  # Italy - Milan
    "NL": [".AS"],  # Netherlands - Euronext Amsterdam
    "NO": [".OL"],  # Norway - Oslo
    "PL": [".WA"],  # Poland - Warsaw
    "PT": [".LS"],  # Portugal - Lisbon
    "SE": [".ST"],  # Sweden - Stockholm
    "UK": [".L"],  # United Kingdom - London Stock Exchange
    "US": [""],  # US stocks typically have no suffix
}

def loadSymbols(country):
    symbolsFile = f"symbols/{country}_symbols.json"
    if not os.path.exists(symbolsFile):
        return

    with open(symbolsFile, "r") as f:
        symbols = json.load(f)

    return symbols

def fetchAndSaveData(country):
    symbols = loadSymbols(country)
    if not symbols:
        return

    dataDir = "data/raw"
    os.makedirs(dataDir, exist_ok=True)

    fetchedSymbols = os.listdir(dataDir)
    symbols = [symbol for symbol in symbols if f"{country}_{symbol}.json" not in fetchedSymbols]

    print(f"Fetching data for {country}...")
    print(f"Symbols: {len(symbols)}")

    q = tqdm.tqdm(total=len(symbols))
    for symbol in symbols:
        try:
            stock = yf.Ticker(symbol)
            hist = stock.history(period="max")

            if hist.empty:
                for suffix in EX_SUFFIXES[country]:
                    stock = yf.Ticker(symbol + suffix)
                    hist = stock.history(period="max")
                    if not hist.empty:
                        break
                else:
                    continue

            hist = hist.reset_index()[["Date", "Open", "High", "Low", "Close", "Volume"]]
            hist.columns = ["date", "open", "high", "low", "close", "volume"]

            outputFile = f"{dataDir}/{country}_{symbol}.json"
            hist.to_json(outputFile, orient="records", date_format="iso")
        except Exception as e:
            pass
        finally:
            q.update(1)

def loadStockData(country, symbol):
    dataFile = f"data/raw/{country}_{symbol}.json"
    if not os.path.exists(dataFile):
        return

    data = pd.read_json(dataFile)
    data["date"] = pd.to_datetime(data["date"])

    return data

def computeSMA(data, period) -> list:
    sma = data["close"].rolling(window=period).mean()
    sma = sma.bfill()

    return sma.tolist()

def computeEMA(data, period) -> list:
    ema = data["close"].ewm(span=period, adjust=False).mean()
    ema = ema.bfill()

    return ema.tolist()

def computeMACD(data, shortPeriod, longPeriod, signalPeriod) -> list:
    shortEMA = computeEMA(data, shortPeriod)
    longEMA = computeEMA(data, longPeriod)

    macd = [shortEMA[i] - longEMA[i] for i in range(len(shortEMA))]
    signal = pd.Series(macd).ewm(span=signalPeriod, adjust=False).mean()

    return macd, signal.tolist()

def computeBollinger(data, period) -> list:
    sma = computeSMA(data, period)
    std = computeStandardDeviation(data, period)

    upper = [sma[i] + 2 * std[i] for i in range(len(sma))]
    lower = [sma[i] - 2 * std[i] for i in range(len(sma))]

    return upper, lower

def computeRSI(data, period) -> list:
    delta = data["close"].diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)

    avgGain = gain.rolling(window=period).mean()
    avgLoss = loss.rolling(window=period).mean()

    rs = avgGain / avgLoss
    rsi = 100 - (100 / (1 + rs))

    return rsi.tolist()

def computeStochasticOscillator(data, period) -> list:
    low = data["low"].rolling(window=period).min()
    high = data["high"].rolling(window=period).max()

    k = 100 * (data["close"] - low) / (high - low)
    d = k.rolling(window=3).mean()

    return k.tolist(), d.tolist()

def computeROC(data, period) -> list:
    roc = data["close"].pct_change(period)
    roc = roc.bfill()

    return roc.tolist()

def computeATR(data, period) -> list:
    tr = pd.DataFrame()
    tr["h-l"] = data["high"] - data["low"]
    tr["h-cp"] = (data["high"] - data["close"].shift(1)).abs()
    tr["l-cp"] = (data["low"] - data["close"].shift(1)).abs()

    tr = tr.max(axis=1)
    atr = tr.rolling(window=period).mean()

    return atr.tolist()

def computeStandardDeviation(data, period) -> list:
    std = data["close"].rolling(window=period).std()
    std = std.bfill()

    return std.tolist()

def computeVolumeAverage(data, period) -> list:
    volume = data["volume"].rolling(window=period).mean()
    volume = volume.bfill()

    return volume.tolist()

def computeFeatures(data) -> dict:
    features = {}

    features["day"] = data["date"].dt.day.tolist()
    features["month"] = data["date"].dt.month.tolist()
    features["weekday"] = data["date"].dt.weekday.tolist()

    features["close"] = data["close"].tolist()
    features["volume"] = data["volume"].tolist()
    features["open"] = data["open"].tolist()
    features["high"] = data["high"].tolist()
    features["low"] = data["low"].tolist()

    features["sma5"] = computeSMA(data, 5)
    features["sma10"] = computeSMA(data, 10)
    features["sma20"] = computeSMA(data, 20)
    #features["sma100"] = computeSMA(data, 100)
    #features["sma200"] = computeSMA(data, 200)

    features["ema5"] = computeEMA(data, 5)
    features["ema10"] = computeEMA(data, 10)
    features["ema20"] = computeEMA(data, 20)
    #features["ema50"] = computeEMA(data, 50)
    #features["ema100"] = computeEMA(data, 100)
    #features["ema200"] = computeEMA(data, 200)

    macd, signal = computeMACD(data, 12, 26, 9)
    features["macd"] = macd
    features["signal"] = signal

    features["bollingerUpper"], features["bollingerLower"] = computeBollinger(data, 20)

    features["rsi14"] = computeRSI(data, 14)
    #features["rsi28"] = computeRSI(data, 28)

    k14, d14 = computeStochasticOscillator(data, 14)
    features["stochasticOscillator14k"] = k14
    features["stochasticOscillator14d"] = d14
    #k28, d28 = computeStochasticOscillator(data, 28)
    #features["stochasticOscillator28k"] = k28
    #features["stochasticOscillator28d"] = d28
    #k50, d50 = computeStochasticOscillator(data, 50)
    #features["stochasticOscillator50k"] = k50
    #features["stochasticOscillator50d"] = d50

    features["roc14"] = computeROC(data, 14)
    #features["roc28"] = computeROC(data, 28)
    #features["roc50"] = computeROC(data, 50)

    features["atr14"] = computeATR(data, 14)
    #features["atr28"] = computeATR(data, 28)
    #features["atr50"] = computeATR(data, 50)

    features["std14"] = computeStandardDeviation(data, 14)
    #features["std28"] = computeStandardDeviation(data, 28)
    #features["std50"] = computeStandardDeviation(data, 50)

    features["volumeAverage5"] = computeVolumeAverage(data, 5)
    features["volumeAverage10"] = computeVolumeAverage(data, 10)
    features["volumeAverage20"] = computeVolumeAverage(data, 20)
    #features["volumeAverage50"] = computeVolumeAverage(data, 50)
    #features["volumeAverage100"] = computeVolumeAverage(data, 100)
    #features["volumeAverage200"] = computeVolumeAverage(data, 200)

    return features

def evaluateFuture(data) -> int:
    """ 0: sell, 1: hold, 2: buy """

    decDecToSell = PERCENTAGE_DEC_TO_SELL / 100
    decIncToBuy = PERCENTAGE_INC_TO_BUY / 100

    futureData = data.iloc[-7:]
    futureClose = futureData["close"].tolist()
    lastClose = data.iloc[-1]["close"]

    smallestFutureClose = min(futureClose)
    biggestFutureClose = max(futureClose)

    if smallestFutureClose < lastClose * (1 - decDecToSell):
        return 0
    elif biggestFutureClose > lastClose * (1 + decIncToBuy):
        return 2
    else:
        return 1

def printFeatures(features):
    for key, value in features.items():
        print(f"{key}: {value[-5:]}")
        input()

def createTrainingCase(data):
    dataNoFuture = data.iloc[:-7]
    features = computeFeatures(dataNoFuture)
    label = evaluateFuture(data)

    return features, label

def isSimilar(a, b):
    return abs(a - b) / ((a + b) / 2) < 0.05

def generateTrainingCases(n):
    symbols = []
    for country in COUNTRIES:
        countrySymbols = loadSymbols(country)
        for symbol in countrySymbols:
            symbols.append((country, symbol))

    labelsCount = [0, 0, 0] # sell, hold, buy

    print(f"Generating {n} training cases...")
    with open("data/training.json", "w") as f:
        i = 0
        q = tqdm.tqdm(total=n)
        while i < n:
            # get random symbol
            symbol = symbols[np.random.randint(0, len(symbols))]
            data = loadStockData(symbol[0], symbol[1])
            # get random period of data that is at least 200 days long and is continuous
            if data is None or len(data) < 207:
                continue
            dataStartIndex = 0
            dataEndIndex = len(data) - 207
            randomStartIndex = np.random.randint(dataStartIndex, dataEndIndex)
            data = data.iloc[randomStartIndex:randomStartIndex+207]

            features, label = createTrainingCase(data)
            if label == 0 and labelsCount[0] >= n / 3 and not isSimilar(labelsCount[1], labelsCount[0]) and not isSimilar(labelsCount[2], labelsCount[0]):
                continue
            if label == 1 and labelsCount[1] >= n / 3 and not isSimilar(labelsCount[0], labelsCount[1]) and not isSimilar(labelsCount[2], labelsCount[1]):
                continue
            if label == 2 and labelsCount[2] >= n / 3 and not isSimilar(labelsCount[0], labelsCount[2]) and not isSimilar(labelsCount[1], labelsCount[2]):
                continue

            labelsCount[label] += 1
            trainingCase = {
                "features": features,
                "label": label
            }
            f.write(json.dumps(trainingCase))
            f.write("\n")
            i += 1
            q.update(1)
        q.close()

def convertTrainingDataToMatrix():
    with open("data/training.json", "r") as f:
        with open("data/training_matrix.csv", "w") as fMatrix:
            with open("data/training_labels.csv", "w") as fLabels:
                q = tqdm.tqdm(total=DATASET_SIZE)
                for line in f:
                    data = json.loads(line)
                    for key, value in data["features"].items():
                        for i in range(len(value)):
                            if value[i] is None:
                                value[i] = 0
                            fMatrix.write(str(value[i]))
                            fMatrix.write(",")
                    fMatrix.write("\n")
                    fLabels.write(str(data["label"]) + "\n")
                    q.update(1)
                q.close()

def main():
    #for country in COUNTRIES:
    #    fetchAndSaveData(country)
    #print("Data fetching completed.")

    print("Generating training cases...")
    generateTrainingCases(DATASET_SIZE)
    print("Training cases generated.")

    print("Converting training data to matrix...")
    convertTrainingDataToMatrix()
    print("Training data converted to matrix.")

    print("All done.")


if __name__ == "__main__":
    main()
