import yfinance as yf
import json
import os
import pandas as pd
from datetime import datetime
import numpy as np
import tqdm
from multiprocessing import Process, Array, Value, Lock, Queue
from queue import Full
import time
import psutil

PERCENTAGE_INC_TO_BUY = 5 # in percent
PERCENTAGE_DEC_TO_SELL = 2 # in percent

DATASET_SIZE = 1000000

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

QUEUE_LIMIT = int(psutil.virtual_memory().total / (1024 ** 3) * psutil.cpu_count() * 100000)

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

def evaluateFuture(data, currentClose) -> int:
    """ 0: sell, 1: hold, 2: buy """

    decDecToSell = PERCENTAGE_DEC_TO_SELL / 100
    decIncToBuy = PERCENTAGE_INC_TO_BUY / 100

    futureClose = data["close"].tolist()

    smallestFutureClose = min(futureClose)
    biggestFutureClose = max(futureClose)

    if smallestFutureClose < currentClose * (1 - decDecToSell):
        return 0
    elif biggestFutureClose > currentClose * (1 + decIncToBuy):
        return 2
    else:
        return 1

def printFeatures(features):
    for key, value in features.items():
        print(f"{key}: {value[-5:]}")
        input()

def createTrainingCase(data):
    dataNoFuture = data.iloc[:-7]
    dataFuture = data.iloc[-7:]
    currentClose = dataNoFuture.iloc[-1]["close"]
    features = computeFeatures(dataNoFuture)
    label = evaluateFuture(dataFuture, currentClose)

    return features, label

def isSimilar(a, b):
    return abs(a - b) / ((a + b) / 2) < 0.05

def generateTrainingCasesWorker(symbols, labelsCount, globalI, lock, queue, n):
    while True:
        with lock:
            if globalI.value >= n:
                return

        symbol = symbols[np.random.randint(0, len(symbols))]
        data = loadStockData(symbol[0], symbol[1])
        if data is None or len(data) < 207:
            continue
        dataEndIndex = len(data) - 207
        if dataEndIndex <= 0:
            continue
        randomStartIndex = np.random.randint(0, dataEndIndex)
        dataSegment = data.iloc[randomStartIndex:randomStartIndex+207]
        features, label = createTrainingCase(dataSegment)
        if features is None:
            continue

        with lock:
            currentI = globalI.value
            if currentI >= n:
                return

            lc = labelsCount[:]
            allow = True
            if label == 0:
                if (lc[0] >= n/3 and
                    not isSimilar(lc[1], lc[0]) and
                    not isSimilar(lc[2], lc[0])):
                    allow = False
            elif label == 1:
                if (lc[1] >= n/3 and
                    not isSimilar(lc[0], lc[1]) and
                    not isSimilar(lc[2], lc[1])):
                    allow = False
            elif label == 2:
                if (lc[2] >= n/3 and
                    not isSimilar(lc[0], lc[2]) and
                    not isSimilar(lc[1], lc[2])):
                    allow = False

            if allow:
                labelsCount[label] += 1
                globalI.value += 1
                while True:
                    try:
                        queue.put((features, label), timeout=0.1)
                        break
                    except Full:
                        if globalI.value >= n:
                            return

def generateTrainingCasesWriter(queue, n):
    with open("data/training.json", "w") as f:
        count = 0
        while count < n:
            case = queue.get()
            if case is None:
                break
            features, label = case
            trainingCase = {
                "features": features,
                "label": int(label)
            }
            f.write(json.dumps(trainingCase) + "\n")
            count += 1

def generateTrainingCases(n):
    symbols = []
    for country in COUNTRIES:
        countrySymbols = loadSymbols(country)
        symbols.extend((country, sym) for sym in countrySymbols)

    labelsCount = Array('i', [0, 0, 0])
    globalI = Value('i', 0)
    lock = Lock()
    queue = Queue(maxsize=80000)

    pbar = tqdm.tqdm(total=n, desc="Generating training cases")

    writerProcess = Process(target=generateTrainingCasesWriter, args=(queue, n))
    writerProcess.start()

    numWorkers = os.cpu_count()
    workers = []
    for _ in range(numWorkers):
        p = Process(target=generateTrainingCasesWorker,
                   args=(symbols, labelsCount, globalI, lock, queue, n))
        p.start()
        workers.append(p)

    try:
        while any(w.is_alive() for w in workers):
            current_i = globalI.value
            pbar.n = current_i
            pbar.refresh()
            time.sleep(0.1)
    except KeyboardInterrupt:
        pass

    for w in workers:
        w.join()

    writerProcess.join()

    pbar.n = globalI.value
    pbar.close()

    print(f"Generated {globalI.value} training cases with label distribution: {list(labelsCount)}")

def convertTrainingDataToMatrix():
    with open("data/training.json", "r") as f:
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

def shuffleTrainingData():
    matrixOffsets = []
    with open("data/training_matrix.csv", "rb") as f:
        for _ in tqdm.tqdm(range(DATASET_SIZE), desc="Reading matrix offsets"):
            matrixOffsets.append(f.tell())
            f.readline()

    labelsOffsets = []
    with open("data/training_labels.csv", "rb") as f:
        for _ in tqdm.tqdm(range(DATASET_SIZE), desc="Reading labels offsets"):
            labelsOffsets.append(f.tell())
            f.readline()

    assert len(matrixOffsets) == DATASET_SIZE and len(labelsOffsets) == DATASET_SIZE, "Dataset size mismatch"

    lineIndices = np.arange(DATASET_SIZE)
    np.random.shuffle(lineIndices)

    with open("data/training_matrix.csv", "rb") as fMatrixIn, \
         open("data/training_labels.csv", "rb") as fLabelsIn, \
         open("data/training_matrix_shuffled.csv.tmp", "wb") as fMatrixOut, \
         open("data/training_labels_shuffled.csv.tmp", "wb") as fLabelsOut:

        for idx in tqdm.tqdm(lineIndices, desc="Shuffling data"):
            fMatrixIn.seek(matrixOffsets[idx])
            matrixLine = fMatrixIn.readline()
            fMatrixOut.write(matrixLine)

            fLabelsIn.seek(labelsOffsets[idx])
            labelLine = fLabelsIn.readline()
            fLabelsOut.write(labelLine)

    os.replace("data/training_matrix_shuffled.csv.tmp", "data/training_matrix.csv")
    os.replace("data/training_labels_shuffled.csv.tmp", "data/training_labels.csv")

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

    #print("Shuffling training data...")
    #shuffleTrainingData()
    #print("Training data shuffled.")

    print("All done.")


if __name__ == "__main__":
    main()
