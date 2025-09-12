import os
import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report
from sklearn.model_selection import cross_val_score, TimeSeriesSplit
from sklearn.datasets import make_classification
import numpy as np
import math

MODEL_FILE_PATH = "models/"
TRAIN_DATA = "data/training_matrix.csv"
TRAIN_LABELS = "data/training_labels.csv"
INPUT_FILE = "data/input.csv"
SIZE_LIMIT = 20
CHUNK_SIZE = 80000

N_ESTIMATORS = 10000
MAX_DEPTH = 10
MIN_SAMPLES_SPLIT = 10
MIN_SAMPLES_LEAF = 5
MAX_FEATURES = "sqrt"
RANDOM_STATE = 42

TEST_SIZE = 0.2

def trainBatch():
    print("Batch training initialized...")

    totalRows = 0
    print("Preprocessing dataset...")
    for chunk in pd.read_csv(TRAIN_DATA, chunksize=CHUNK_SIZE, header=None):
        if np.any(np.isnan(chunk)):
            print("Found NaN values in a chunk; replacing with 0s...")
            chunk.fillna(0, inplace=True)
        if np.any(np.isinf(chunk)):
            print("Found infinite values in a chunk; replacing with 0s...")
            chunk.replace([np.inf, -np.inf], 0, inplace=True)
        totalRows += chunk.shape[0]
    print("Dataset preprocessed successfully!")

    labels = pd.read_csv(TRAIN_LABELS, header=None).values.ravel()
    labels = labels.astype(np.float32)
    if len(labels) != totalRows:
        print("Number of labels does not match number of rows in training data!")

    print("Computing class weights...")
    classWeights = compute_class_weight("balanced", classes=np.unique(labels), y=labels)
    classWeightDict = {i: classWeights[i] for i in range(len(classWeights))}
    print(f"Class weights:")
    for key, value in classWeightDict.items():
        print(f"Class {key}: {value:.2f}")

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
                                bootstrap=True,
                                n_jobs=-1)

    numTrainChunks = math.ceil(trainRows / CHUNK_SIZE)
    treesPerChunk = N_ESTIMATORS // numTrainChunks
    if treesPerChunk < 1:
        treesPerChunk = 1

    treesAdded = 0
    currentIndex = 0

    xTestAll = []
    yTestAll = []

    chunkAccuracies = []

    print("Starting batch training...")
    for chunk in pd.read_csv(TRAIN_DATA, chunksize=CHUNK_SIZE, header=None):
        print(f"Processing rows {currentIndex} to {currentIndex + chunk.shape[0]} of {totalRows}...")
        chunkRows = chunk.shape[0]
        chunkLabels = labels[currentIndex: currentIndex + chunkRows]


        if np.any(np.isnan(chunk)):
            print("Found NaN values in a chunk; replacing with 0s...")
            chunk.fillna(0, inplace=True)
        if np.any(np.isinf(chunk)):
            print("Found infinite values in a chunk; replacing with 0s...")
            chunk.replace([np.inf, -np.inf], 0, inplace=True)

        chunk = chunk.astype(np.float32)

        if currentIndex < trainRows:
            if currentIndex + chunkRows <= trainRows:
                xTrainChunk = chunk
                yTrainChunk = chunkLabels
            else:
                splitPoint = trainRows - currentIndex
                xTrainChunk = chunk[:splitPoint]
                yTrainChunk = chunkLabels[:splitPoint]
                xTestChunk = chunk[splitPoint:]
                yTestChunk = chunkLabels[splitPoint:]
                xTestAll.append(xTestChunk)
                yTestAll.append(yTestChunk)
        else:
            xTestAll.append(chunk)
            yTestAll.append(chunkLabels)
            currentIndex += chunkRows
            targetRows = currentIndex + chunkRows
            if targetRows > totalRows:
                targetRows = totalRows
            print(f"Processed test rows {currentIndex} to {targetRows}.")
            continue

        if treesAdded + treesPerChunk > N_ESTIMATORS:
            treesToAdd = N_ESTIMATORS - treesAdded
        else:
            treesToAdd = treesPerChunk

        rf.n_estimators += treesToAdd

        rf.fit(xTrainChunk, yTrainChunk)

        yPred = rf.predict(xTrainChunk)
        acc = accuracy_score(yTrainChunk, yPred)
        chunkAccuracies.append(acc)

        treesAdded += treesToAdd
        targetRows = currentIndex + chunkRows
        if targetRows > trainRows:
            targetRows = trainRows
        print(f"Processed training rows {currentIndex} to {targetRows}: added {treesToAdd} trees (Total trees: {rf.n_estimators}).")
        print(f"Chunk Accuracy: {acc:.2f}")
        currentIndex += chunkRows

    print("Batch training completed!")
    print("Training Accuracy:")
    print(f"Mean: {np.mean(chunkAccuracies):.2f}")
    print(f"Standard Deviation: {np.std(chunkAccuracies):.2f}")

    print("Preprocessing test data...")
    numFeatures = xTestAll[0].shape[1]
    if xTestAll:
        xTestAll = np.vstack(xTestAll)
        yTestAll = np.concatenate(yTestAll)
        print(f"Test data shape: {xTestAll.shape}")
        print("Evaluating the model...")
        yPred = rf.predict(xTestAll)
        acc = accuracy_score(yTestAll, yPred)
        kf = TimeSeriesSplit(n_splits=5)
        cvAcc = cross_val_score(rf, xTestAll, yTestAll, cv=kf, scoring="accuracy")
        print(f"Model Accuracy: {acc:.2f}")
        print("Classification Report:")
        print(classification_report(yTestAll, yPred))
        print(f"Cross Validation Accuracy: {np.mean(cvAcc):.2f} (+/- {np.std(cvAcc) * 2:.2f})")
    else:
        print("No test data collected for evaluation.")

    modelName = f"rf-model-f32-{totalRows}-{numFeatures}-{N_ESTIMATORS}-{MAX_DEPTH}-{MIN_SAMPLES_SPLIT}-{MIN_SAMPLES_LEAF}-{MAX_FEATURES}-{RANDOM_STATE}.model"
    modelPath = MODEL_FILE_PATH + modelName
    joblib.dump(rf, modelPath)
    print(f"Model trained and saved successfully as '{modelName}'!")

    return rf

def loadOrTrain(skipPrompt=False):
    if os.path.exists(MODEL_FILE_PATH) and not skipPrompt:
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
        try:
            os.makedirs(MODEL_FILE_PATH)
        except FileExistsError:
            pass
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
        x = x.astype(np.float32)
        if np.any(np.isnan(x)):
            print("Found NaN values in the data! Replacing with 0s...")
            x.fillna(0, inplace=True)
        if np.any(np.isinf(x)):
            print("Found infinite values in the data! Replacing with 0s...")
            x.replace([np.inf, -np.inf], 0, inplace=True)
        print("Data preprocessed successfully!")

        print(f"Training data shape: {x.shape}")

        print("Loading training labels...")
        y = pd.read_csv(TRAIN_LABELS, header=None).values.ravel()
        print(f"Training labels shape: {y.shape}")

        print("Splitting data into training and testing sets...")
        xTrain, xTest, yTrain, yTest = train_test_split(x, y, test_size=TEST_SIZE, shuffle=False)
        print(f"Successfully split data into training and testing sets!")

        print("Computing class weights...")
        classWeights = compute_class_weight("balanced", classes=np.unique(yTrain), y=yTrain)
        classWeightDict = {i: classWeights[i] for i in range(len(classWeights))}
        print(f"Class weights:")
        for key, value in classWeightDict.items():
            print(f"Class {key}: {value:.2f}")

        print("Training the model...")
        rf = RandomForestClassifier(n_estimators=N_ESTIMATORS,
                                    class_weight=classWeightDict,
                                    max_depth=MAX_DEPTH,
                                    min_samples_split=MIN_SAMPLES_SPLIT,
                                    min_samples_leaf=MIN_SAMPLES_LEAF,
                                    max_features=MAX_FEATURES,
                                    random_state=RANDOM_STATE,
                                    bootstrap=True,
                                    n_jobs=-1)
        rf.fit(xTrain, yTrain)
        print("Model trained successfully!")

        trainPred = rf.predict(xTrain)
        print(f"Training Accuracy: {accuracy_score(yTrain, trainPred):.2f}")

        print("Evaluating the model...")
        yPred = rf.predict(xTest)
        kf = TimeSeriesSplit(n_splits=5)
        cvAcc = cross_val_score(rf, xTest, yTest, cv=kf, scoring="accuracy")
        print(f"Model Accuracy: {accuracy_score(yTest, yPred):.2f}")
        print("Classification Report:")
        print(classification_report(yTest, yPred))
        print("k-Fold Cross Validation:")
        print(f"Cross Validation Accuracy: {np.mean(cvAcc):.2f} (+/- {np.std(cvAcc) * 2:.2f})")

        numFeatures = x.shape[1]
        print("Saving the model...")

        name = f"rf-model-f32-{sizeData}-{numFeatures}-{N_ESTIMATORS}-{MAX_DEPTH}-{MIN_SAMPLES_SPLIT}-{MIN_SAMPLES_LEAF}-{MAX_FEATURES}-{RANDOM_STATE}.model"
        joblib.dump(rf, MODEL_FILE_PATH + name)
        print("Model trained and saved successfully!")

    return rf


def predict(rf):
    input = pd.read_csv(INPUT_FILE, header=None)
    input = input.astype(np.float32)
    print("Preprocessing the input data...")
    if np.any(np.isnan(input)):
        print("Found NaN values in the input data! Replacing with 0s...")
        input.fillna(0, inplace=True)
    if np.any(np.isinf(input)):
        print("Found infinite values in the input data! Replacing with 0s...")
        input.replace([np.inf, -np.inf], 0, inplace=True)
    print("Data preprocessed successfully!")

    print("Predicting...")
    pred = rf.predict(input)
    print("Predictions:")
    print(pred)

    return pred

if __name__ == "__main__":
    rf = loadOrTrain(skipPrompt=True)
    predict(rf)
