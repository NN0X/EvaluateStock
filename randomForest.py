import os
import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight
import numpy as np
import math

MODEL_FILE_PATH = "models/"
TRAIN_DATA = "data/training_matrix.csv"
TRAIN_LABELS = "data/training_labels.csv"
INPUT_FILE = "data/input.csv"
SIZE_LIMIT = 20
CHUNK_SIZE = 100000

N_ESTIMATORS = 500
MAX_DEPTH = 20
MIN_SAMPLES_SPLIT = 10
MIN_SAMPLES_LEAF = 5
MAX_FEATURES = "sqrt"
RANDOM_STATE = 42

TEST_SIZE = 0.2

def trainBatch():
    print("Batch training initialized...")

    scaler = StandardScaler()
    totalRows = 0
    print("Computing scaler statistics...")
    for chunk in pd.read_csv(TRAIN_DATA, chunksize=CHUNK_SIZE, header=None):
        if np.any(np.isnan(chunk)):
            print("Found NaN values in a chunk; replacing with 0s...")
            chunk.fillna(0, inplace=True)
        if np.any(np.isinf(chunk)):
            print("Found infinite values in a chunk; replacing with 0s...")
            chunk.replace([np.inf, -np.inf], 0, inplace=True)
        chunk = chunk.astype(np.float32)
        scaler.partial_fit(chunk)
        totalRows += chunk.shape[0]
    print(f"Scaler computed from {totalRows} rows.")

    labels = pd.read_csv(TRAIN_LABELS, header=None).values.ravel()
    labels = labels.astype(np.float32)
    if len(labels) != totalRows:
        print("Number of labels does not match number of rows in training data!")

    print("Computing class weights...")
    classWeights = compute_class_weight("balanced", classes=np.unique(labels), y=labels)
    classWeightDict = {i: classWeights[i] for i in range(len(classWeights))}
    print(f"Class weights: {classWeightDict}")

    trainRows = int(totalRows * (1 - TEST_SIZE))
    print(f"Total rows: {totalRows}, Training rows: {trainRows}, Testing rows: {totalRows - trainRows}")

    rf = RandomForestClassifier(n_estimators=0,
                                warm_start=True,
                                class_weight=classWeightDict,
                                max_depth=MAX_DEPTH,
                                min_samples_split=MIN_SAMPLES_SPLIT,
                                min_samples_leaf=MIN_SAMPLES_LEAF,
                                max_features=MAX_FEATURES,
                                random_state=RANDOM_STATE,
                                n_jobs=-1)

    numTrainChunks = math.ceil(trainRows / CHUNK_SIZE)
    treesPerChunk = N_ESTIMATORS // numTrainChunks
    if treesPerChunk < 1:
        treesPerChunk = 1

    treesAdded = 0
    currentIndex = 0

    xTestAll = []
    yTestAll = []

    print("Starting batch training...")
    for chunk in pd.read_csv(TRAIN_DATA, chunksize=CHUNK_SIZE, header=None):
        chunkRows = chunk.shape[0]
        chunkLabels = labels[currentIndex: currentIndex + chunkRows]


        if np.any(np.isnan(chunk)):
            print("Found NaN values in a chunk; replacing with 0s...")
            chunk.fillna(0, inplace=True)
        if np.any(np.isinf(chunk)):
            print("Found infinite values in a chunk; replacing with 0s...")
            chunk.replace([np.inf, -np.inf], 0, inplace=True)

        chunk = chunk.astype(np.float32)
        chunkScaled = scaler.transform(chunk)

        if currentIndex < trainRows:
            if currentIndex + chunkRows <= trainRows:
                xTrainChunk = chunkScaled
                yTrainChunk = chunkLabels
            else:
                splitPoint = trainRows - currentIndex
                xTrainChunk = chunkScaled[:splitPoint]
                yTrainChunk = chunkLabels[:splitPoint]
                xTestChunk = chunkScaled[splitPoint:]
                yTestChunk = chunkLabels[splitPoint:]
                xTestAll.append(xTestChunk)
                yTestAll.append(yTestChunk)
        else:
            xTestAll.append(chunkScaled)
            yTestAll.append(chunkLabels)
            currentIndex += chunkRows
            continue

        if treesAdded + treesPerChunk > N_ESTIMATORS:
            treesToAdd = N_ESTIMATORS - treesAdded
        else:
            treesToAdd = treesPerChunk

        rf.n_estimators += treesToAdd

        rf.fit(xTrainChunk, yTrainChunk)
        treesAdded += treesToAdd
        print(f"Processed rows {currentIndex} to {currentIndex + chunkRows}: added {treesToAdd} trees (Total trees: {rf.n_estimators}).")
        currentIndex += chunkRows

    if xTestAll:
        xTestAll = np.vstack(xTestAll)
        yTestAll = np.concatenate(yTestAll)
        print("Evaluating the model...")
        yPred = rf.predict(xTestAll)
        acc = accuracy_score(yTestAll, yPred)
        print(f"Model Accuracy: {acc:.2f}")
    else:
        print("No test data collected for evaluation.")

    modelName = f"rd-model-f32-{totalRows}-{N_ESTIMATORS}-{MAX_DEPTH}-{MIN_SAMPLES_SPLIT}-{MIN_SAMPLES_LEAF}-{MAX_FEATURES}-{RANDOM_STATE}.model"
    modelPath = MODEL_FILE_PATH + modelName
    joblib.dump(rf, modelPath)
    print(f"Model trained and saved successfully as '{modelName}'!")

    return rf

def loadOrTrain():
    if os.path.exists(MODEL_FILE_PATH):
        models = os.listdir(MODEL_FILE_PATH)
        for model in models:
            if model.endswith(".model"):
                print(f"Found model '{model}' in '{MODEL_FILE_PATH}'")
        if input("Do you want to load a model? (Y/n): ").lower() == "y":
            modelName = input("Enter the model name: ")
            modelPath = f"{MODEL_FILE_PATH}{modelName}"
            if os.path.exists(modelPath):
                print(f"Loading model '{modelName}'...")
                rf = joblib.load(modelPath)
                print("Model loaded successfully!")
                return rf
            else:
                print(f"Model '{modelName}' not found! Training a new one...")
    else:
        os.makedirs(MODEL_FILE_PATH)
        print("Model not found!")

    print("Training a new model...")

    print("Loading training data...")


    if os.path.getsize(TRAIN_DATA) > SIZE_LIMIT * 1e9:
        print(f"Training data exceeds {SIZE_LIMIT}GB! Initializing batch training...")
        return trainBatch()
    else:
        x = pd.read_csv(TRAIN_DATA, header=None)
        sizeData = x.shape[0]

        print("Preprocessing the data...")
        if np.any(np.isnan(x)):
            print("Found NaN values in the data! Replacing with 0s...")
            x.fillna(0, inplace=True)
        if np.any(np.isinf(x)):
            print("Found infinite values in the data! Replacing with 0s...")
            x.replace([np.inf, -np.inf], 0, inplace=True)
        print("Data preprocessed successfully!")

        print(f"Training data shape: {x.shape}")
        print("Scaling the data...")
        x = StandardScaler().fit_transform(x.astype(np.float32))
        print(f"Scaled data successfully!")

        print("Loading training labels...")
        y = pd.read_csv(TRAIN_LABELS, header=None).values.ravel()
        print(f"Training labels shape: {y.shape}")

        print("Splitting data into training and testing sets...")
        xTrain, xTest, yTrain, yTest = train_test_split(x, y, test_size=TEST_SIZE, random_state=RANDOM_STATE)
        print(f"Successfully split data into training and testing sets!")

        print("Training the model...")
        rf = RandomForestClassifier(n_estimators=N_ESTIMATORS,
                                    class_weight="balanced",
                                    max_depth=MAX_DEPTH,
                                    min_samples_split=MIN_SAMPLES_SPLIT,
                                    min_samples_leaf=MIN_SAMPLES_LEAF,
                                    max_features=MAX_FEATURES,
                                    random_state=RANDOM_STATE,
                                    n_jobs=-1)
        rf.fit(xTrain, yTrain)
        print("Model trained successfully!")

        print("Evaluating the model...")
        yPred = rf.predict(xTest)
        print(f"Model Accuracy: {accuracy_score(yTest, yPred):.2f}")

        print("Saving the model...")

        name = f"rd-model-f32-{sizeData}-{nEstimators}-{maxDepth}-{minSamplesSplit}-{minSamplesLeaf}-{maxFeatures}-{randomState}.model"
        joblib.dump(rf, MODEL_FILE)
        print("Model trained and saved successfully!")

    return rf

def predict(rf):
    if not os.path.exists(INPUT_FILE):
        print(f"Input file '{INPUT_FILE}' not found! Skipping prediction.")
        return

    inputData = pd.read_csv(INPUT_FILE, header=None)
    predictions = rf.predict(inputData)

    print("Predictions:")
    print(predictions)


if __name__ == "__main__":
    rf = loadOrTrain()
    predict(rf)
