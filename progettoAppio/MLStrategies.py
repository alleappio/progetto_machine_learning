import pandas as pd
import numpy as np

from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn import metrics

class GeneralStrategy:
    def train(self):
        self.reg = self.reg.fit(self.X_train,self.y_train)
    
    def get_metrics(self, y_val, y_pred):
        calculated_metrics={
            "MAE": metrics.mean_absolute_error(y_val, y_pred),
            "MSE": metrics.mean_squared_error(y_val, y_pred),
            "RMSE": np.sqrt(metrics.mean_squared_error(y_val, y_pred)),
        }
        return calculated_metrics
    
    def get_scores(self):
        self.y_pred_train = self.reg.predict(self.X_train)
        self.y_pred_test = self.reg.predict(self.X_test)
        scores={
            "train": self.get_metrics(self.y_pred_train, self.y_train),
            "test": self.get_metrics(self.y_pred_test, self.y_test)
        }
        return scores

class LR(GeneralStrategy):
    def __init__(self, X_train, y_train, X_test, y_test):
        self.X_train=X_train
        self.y_train=y_train
        self.X_test=X_test
        self.y_test=y_test
        self.reg = LinearRegression()
    

class KNNR(GeneralStrategy):
    def __init__(self, X_train, y_train, X_test, y_test, neighbors_number=5, weights = 'uniform'):
        self.X_train=X_train
        self.y_train=y_train
        self.X_test=X_test
        self.y_test=y_test
        self.reg = KNeighborsRegressor(
            n_neighbors = neighbors_number,
            weights = weights
        )
    

class DT(GeneralStrategy):
    def __init__(self, X_train, y_train, X_test, y_test):
        self.X_train=X_train
        self.y_train=y_train
        self.X_test=X_test
        self.y_test=y_test
        self.reg = DecisionTreeRegressor()
    

class SVM(GeneralStrategy):
    def __init__(self, X_train, y_train, X_test, y_test):
        self.X_train=X_train
        self.y_train=y_train
        self.X_test=X_test
        self.y_test=y_test
        self.reg = SVR()
    
