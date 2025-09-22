import os
import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error

MODEL_FILE_PATH = "models/"
TRAIN_DATA = "data/training_matrix.csv"
TRAIN_LABELS = "data/training_labels.csv"
INPUT_FILE = "data/input.csv"

TEST_SIZE = 0.2
RANDOM_STATE = 42

def trainLinearRegression(X, y):
    model = LinearRegression()
    model.fit(X, y)
    return model

def saveModel(model, X_train, seed=RANDOM_STATE, fit_intercept=True, normalize=False):
    n_samples, n_features = X_train.shape

    fi_flag = 1 if fit_intercept else 0
    norm_flag = 1 if normalize else 0

    filename = (
        f"linreg-model-f32-{n_samples}-{n_features}-"
        f"{fi_flag}-{norm_flag}-{seed}.model"
    )

    os.makedirs(MODEL_FILE_PATH, exist_ok=True)
    filepath = os.path.join(MODEL_FILE_PATH, filename)

    joblib.dump(model, filepath)
    print(f"Model saved to {filepath}")
    return filepath

def loadModel(filename):
    return joblib.load(filename)

def predict(model, input_file=None, X=None):
    if X is None:
        if input_file is None:
            input_file = INPUT_FILE
        if not os.path.exists(input_file):
            raise FileNotFoundError(f"Input file '{input_file}' not found!")
        X = pd.read_csv(input_file, header=None).astype(np.float32)

    if np.any(np.isnan(X)):
        print("Found NaN values in input! Replacing with 0s...")
        X.fillna(0, inplace=True)
    if np.any(np.isinf(X)):
        print("Found infinite values in input! Replacing with 0s...")
        X.replace([np.inf, -np.inf], 0, inplace=True)

    print("Predicting...")
    preds = model.predict(X)
    return preds

def evaluate(model, X, y):
    y_pred = model.predict(X)

    r2 = r2_score(y, y_pred)
    mse = mean_squared_error(y, y_pred)

    actual_sign = np.sign(y)
    pred_sign = np.sign(y_pred)
    directional_acc = np.mean(actual_sign == pred_sign)

    print(f"R² Score: {r2:.4f}")
    print(f"Mean Squared Error: {mse:.4f}")
    print(f"Directional Accuracy: {directional_acc:.2%}")

    return r2, mse, directional_acc

def loadOrTrain(skipPrompt=False):
    if os.path.exists(MODEL_FILE_PATH) and not skipPrompt:
        models = [m for m in os.listdir(MODEL_FILE_PATH) if m.endswith(".model")]
        for model_file in models:
            print(f"Found model '{model_file}' in '{MODEL_FILE_PATH}'")
        if input("Do you want to load a model? (Y/n): ").lower() == "y":
            modelName = input("Enter the model name: ")
            modelPath = os.path.join(MODEL_FILE_PATH, modelName)
            if os.path.exists(modelPath):
                print(f"Loading model '{modelName}'...")
                model = loadModel(modelPath)
                print("Model loaded successfully!")
                return model
            else:
                print(f"Model '{modelName}' not found! Training a new one...")
    else:
        os.makedirs(MODEL_FILE_PATH, exist_ok=True)
        print("Model not found!")

    print("Training a new Linear Regression model...")

    X = pd.read_csv(TRAIN_DATA, header=None).astype(np.float32)
    y = pd.read_csv(TRAIN_LABELS, header=None).values.ravel().astype(np.float32)

    if np.any(np.isnan(X)):
        print("Found NaN values in training data! Replacing with 0s...")
        X.fillna(0, inplace=True)
    if np.any(np.isinf(X)):
        print("Found infinite values in training data! Replacing with 0s...")
        X.replace([np.inf, -np.inf], 0, inplace=True)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, shuffle=False
    )

    model = trainLinearRegression(X_train, y_train)

    print("Evaluating model...")
    evaluate(model, X_test, y_test)

    modelName = saveModel(model, X_train, seed=RANDOM_STATE, fit_intercept=True, normalize=False)
    print(f"Model trained and saved as '{modelName}'")

    return model

if __name__ == "__main__":
    model = loadOrTrain(skipPrompt=True)
