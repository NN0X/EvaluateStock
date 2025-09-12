import numpy as np
from sklearn.linear_model import LinearRegression

def train_linear_regression(X, y):
    model = LinearRegression()
    model.fit(X, y)
    return model

def predict(model, X):
    return model.predict(X)

def evaluate(model, X, y):
    r2_score = model.score(X, y)
    return r2_score

if __name__ == "__main__":
    pass
