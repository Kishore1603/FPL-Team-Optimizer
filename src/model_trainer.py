import numpy as np

# Wrapper class for model training and evaluation to support tests/test_model_trainer.py
class ModelTrainer:
    def __init__(self):
        self.model = None

    def prepare_data(self, test=False):
        # Generate dummy data for testing
        np.random.seed(42)
        X = np.random.rand(100, 5)
        y = (np.random.rand(100) > 0.5).astype(int)
        if test:
            return X[80:], y[80:]
        else:
            return X[:80], y[:80]

    def train_model(self, X, y):
        from xgboost import XGBRegressor
        self.model = XGBRegressor()
        self.model.fit(X, y)
        return self.model

    def evaluate_model(self, model, X, y):
        from sklearn.metrics import accuracy_score
        preds = model.predict(X)
        return accuracy_score(y, preds)
import pandas as pd
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

def train_model(data):
    X = data.drop('target', axis=1)
    y = data['target']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = XGBRegressor()
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    # For regression, use R^2 score as accuracy
    from sklearn.metrics import r2_score
    accuracy = r2_score(y_test, predictions)
    return model, accuracy

def save_model(model, filename):
    model.save_model(filename)

def load_model(filename):
    model = XGBRegressor()
    model.load_model(filename)
    return model