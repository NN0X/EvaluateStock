import yfinance as yf
import json
import os
import pandas as pd
from datetime import datetime
import numpy as np
import tqdm
from multiprocessing import Process, Value, Lock, Queue, Manager
import multiprocessing
from queue import Full
import time
import psutil

DATASET_SIZE = 1000000
WINDOW_SIZE = 30
FUTURE_WINDOW_SIZE = 14

COUNTRIES = ["BE", "CH", "DE", "DK", "ES", "FI", "FR", "IT", "NL", "NO", "PL", "PT", "SE", "UK", "US"]

EX_SUFFIXES = {
    "BE": [".BR"], "CH": [".SW"], "DE": [".DE", ".F", ".XETRA"],
    "DK": [".CO"], "ES": [".MC"], "FI": [".HE"], "FR": [".PA"],
    "IT": [".MI"], "NL": [".AS"], "NO": [".OL"], "PL": [".WA"],
    "PT": [".LS"], "SE": [".ST"], "UK": [".L"], "US": [""],
}

QUEUE_LIMIT = int(psutil.virtual_memory().total / (1024 ** 3) * psutil.cpu_count() * 100000)

def loadSymbols(country):
    symbolsFile = f"symbols/{country}_symbols.json"
    if not os.path.exists(symbolsFile):
        return
    with open(symbolsFile, "r") as f:
        return json.load(f)

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
    roc = data["close"].pct_change(period, fill_method=None)
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

    features["day"] = data["date"].dt.day.tolist()[-1]
    features["month"] = data["date"].dt.month.tolist()[-1]
    features["weekday"] = data["date"].dt.weekday.tolist()[-1]

    features["close"] = data["close"].tolist()
    features["volume"] = data["volume"].tolist()
    features["open"] = data["open"].tolist()
    features["high"] = data["high"].tolist()
    features["low"] = data["low"].tolist()

    features["sma5"] = computeSMA(data, 5)
    features["sma10"] = computeSMA(data, 10)
    features["sma15"] = computeSMA(data, 15)
    features["sma20"] = computeSMA(data, 20)
    features["sma25"] = computeSMA(data, 25)
    features["sma30"] = computeSMA(data, 30)
    #features["sma40"] = computeSMA(data, 40)
    #features["sma100"] = computeSMA(data, 100)
    #features["sma200"] = computeSMA(data, 200)

    features["ema5"] = computeEMA(data, 5)
    features["ema10"] = computeEMA(data, 10)
    features["ema15"] = computeEMA(data, 15)
    features["ema20"] = computeEMA(data, 20)
    features["ema25"] = computeEMA(data, 25)
    features["ema30"] = computeEMA(data, 30)
    #features["ema40"] = computeEMA(data, 40)
    #features["ema50"] = computeEMA(data, 50)
    #features["ema100"] = computeEMA(data, 100)
    #features["ema200"] = computeEMA(data, 200)

    macd, signal = computeMACD(data, 12, 26, 9)
    features["macd"] = macd
    features["signal"] = signal

    features["bollingerUpper"], features["bollingerLower"] = computeBollinger(data, 20)

    features["rsi14"] = computeRSI(data, 14)
    features["rsi28"] = computeRSI(data, 28)

    k14, d14 = computeStochasticOscillator(data, 14)
    features["stochasticOscillator14k"] = k14
    features["stochasticOscillator14d"] = d14
    k28, d28 = computeStochasticOscillator(data, 28)
    features["stochasticOscillator28k"] = k28
    features["stochasticOscillator28d"] = d28
    #k50, d50 = computeStochasticOscillator(data, 50)
    #features["stochasticOscillator50k"] = k50
    #features["stochasticOscillator50d"] = d50

    features["roc14"] = computeROC(data, 14)
    features["roc28"] = computeROC(data, 28)
    #features["roc50"] = computeROC(data, 50)

    features["atr14"] = computeATR(data, 14)
    features["atr28"] = computeATR(data, 28)
    #features["atr50"] = computeATR(data, 50)

    features["std14"] = computeStandardDeviation(data, 14)
    features["std28"] = computeStandardDeviation(data, 28)
    #features["std50"] = computeStandardDeviation(data, 50)

    features["volumeAverage5"] = computeVolumeAverage(data, 5)
    features["volumeAverage10"] = computeVolumeAverage(data, 10)
    features["volumeAverage15"] = computeVolumeAverage(data, 15)
    features["volumeAverage20"] = computeVolumeAverage(data, 20)
    features["volumeAverage25"] = computeVolumeAverage(data, 25)
    features["volumeAverage30"] = computeVolumeAverage(data, 30)
    #features["volumeAverage40"] = computeVolumeAverage(data, 40)
    #features["volumeAverage50"] = computeVolumeAverage(data, 50)
    #features["volumeAverage100"] = computeVolumeAverage(data, 100)
    #features["volumeAverage200"] = computeVolumeAverage(data, 200)

    return features

def evaluateFutureRegression(data, currentClose) -> float:
    lastFutureClose = data["close"].iloc[-1]
    growth = ((lastFutureClose - currentClose) / currentClose) * 100
    return float(growth)

def createTrainingCase(data):
    dataNoFuture = data.iloc[:-FUTURE_WINDOW_SIZE]
    dataFuture = data.iloc[-FUTURE_WINDOW_SIZE:]
    currentClose = dataNoFuture.iloc[-1]["close"]
    features = computeFeatures(dataNoFuture)
    label = evaluateFutureRegression(dataFuture, currentClose)
    return features, label

def generateTrainingCasesWorker(symbols, globalI, symbolsUsed, lock, queue, n):
    window = WINDOW_SIZE + FUTURE_WINDOW_SIZE
    while True:
        if len(symbols) == 0:
            return
        with lock:
            if globalI.value >= n:
                return
            else:
                currentI = globalI.value
        with lock:
            try:
                symbol = symbols[np.random.randint(0, len(symbols))]
            except Exception:
                return
        index = 0
        if symbol in symbolsUsed.keys():
            index = symbolsUsed[symbol]
        data = loadStockData(symbol[0], symbol[1])
        if data is None or len(data) < index*window + window:
            with lock:
                try:
                    symbols.remove(symbol)
                except ValueError:
                    pass
            continue
        dataSegment = data.iloc[index*window:index*window+window]
        with lock:
            symbolsUsed[symbol] = index + 1
        features, label = createTrainingCase(dataSegment)
        if features is None:
            continue
        with lock:
            currentI = globalI.value
            if currentI >= n:
                return
            globalI.value += 1
            while True:
                try:
                    queue.put((features, label), timeout=0.1)
                    break
                except Full:
                    if globalI.value >= n:
                        return

def generateTrainingCasesWriter(queue, n):
    with open("data/training_regression.json", "w") as f:
        count = 0
        while True:
            case = queue.get()
            if case is None:  # stop signal
                break
            features, label = case
            trainingCase = {"features": features, "label": float(label)}
            f.write(json.dumps(trainingCase) + "\n")
            count += 1
            if count >= n:
                break
    print(f"Writer finished. Wrote {count} cases.")

def generateTrainingCases(n):
    symbols = []
    for country in COUNTRIES:
        countrySymbols = loadSymbols(country)
        if countrySymbols:
            symbols.extend((country, sym) for sym in countrySymbols)
    symbolsManaged = multiprocessing.Manager().list()
    symbolsManaged.extend(symbols)
    globalI = Value('i', 0)
    symbolsUsed = multiprocessing.Manager().dict()
    lock = Lock()
    queue = Queue(maxsize=5000)  # reduced queue size
    pbar = tqdm.tqdm(total=n, desc="Generating training cases")
    writerProcess = Process(target=generateTrainingCasesWriter, args=(queue, n))
    writerProcess.start()
    numWorkers = os.cpu_count()
    workers = []
    for _ in range(numWorkers):
        p = Process(target=generateTrainingCasesWorker,
                   args=(symbolsManaged, globalI, symbolsUsed, lock, queue, n))
        p.start()
        workers.append(p)
    try:
        while any(w.is_alive() for w in workers):
            current_i = globalI.value
            pbar.n = current_i
            pbar.refresh()
            time.sleep(0.1)
            with lock:
                if len(symbolsManaged) == 0:
                    break
    except KeyboardInterrupt:
        pass
    for w in workers:
        w.join()
    # send stop signal to writer
    queue.put(None)
    writerProcess.join()
    pbar.n = globalI.value
    pbar.close()
    print(f"Generated {globalI.value} training cases (regression).")

def convertTrainingDataToMatrix():
    with open("data/training_regression.json", "r") as f:
        with open("data/training_matrix.csv", "w") as fMatrix:
            with open("data/training_labels.csv", "w") as fLabels:
                q = tqdm.tqdm(total=DATASET_SIZE)
                for line in f:
                    data = json.loads(line)
                    for key, value in data["features"].items():
                        if isinstance(value, list):
                            for i in range(len(value)):
                                if value[i] is None:
                                    value[i] = 0
                                fMatrix.write(str(value[i]))
                                fMatrix.write(",")
                        else:
                            if value is None:
                                value = 0
                            fMatrix.write(str(value))
                            fMatrix.write(",")
                    fMatrix.write("\n")
                    fLabels.write(str(data["label"]) + "\n")
                    q.update(1)
                q.close()

def main():
    print("Generating regression training cases...")
    generateTrainingCases(DATASET_SIZE)
    print("Cases generated.")
    print("Converting regression data to matrix...")
    convertTrainingDataToMatrix()
    print("Conversion complete.")
    print("All done.")

if __name__ == "__main__":
    main()
