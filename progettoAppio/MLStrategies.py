import pandas as pd
import numpy as np

from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVR

from sklearn import metrics
from matplotlib import pyplot as plt

class GeneralStrategy:
    def train(self):
        self.reg = self.reg.fit(self.X_train, self.y_train)
    
    def cv_train(self):
        grid_search = GridSearchCV(estimator = self.reg, param_grid = self.param_grid, cv = 5, scoring = 'neg_mean_squared_error', n_jobs = -1)
        grid_search = grid_search.fit(self.X_train, self.y_train)
        self.reg = grid_search.best_estimator_

    def get_metrics(self, y_test, y_pred):
        calculated_metrics={
            "MAE": metrics.mean_absolute_error(y_test, y_pred),
            "MSE": metrics.mean_squared_error(y_test, y_pred),
            "RMSE": np.sqrt(metrics.mean_squared_error(y_test, y_pred)),
        }
        return calculated_metrics
    
    def get_scores(self, X_test, y_test):
        self.y_test = y_test
        self.y_pred_train = self.reg.predict(self.X_train)
        self.y_pred_test = self.reg.predict(X_test)
        scores={
            "train": self.get_metrics(self.y_pred_train, self.y_train),
            "test": self.get_metrics(self.y_pred_test, y_test)
        }
        return scores
    
    def get_predictions(self):
        return (self.y_pred_test, self.y_test)
    
    def get_model_name(self):
        return self.model_name
    
    def get_model(self):
        return self.reg
    
    def get_best_cv(self):
        return self.best_cv

    def set_model(self, model):
        self.reg = model

    def set_X_y_train(self,X,y):
        self.X_train = X
        self.y_train = y

    def set_selected_features(self, selected_features):
        self.selected_features = selected_features
        self.X_train = self.X_train[selected_features]
    
    def set_param_grid(self, grid):
        self.param_grid = grid


class LR(GeneralStrategy):
    def __init__(self):
        self.reg = LinearRegression()
        self.model_name = "Linear regression"
    
class KNNR(GeneralStrategy):
    def __init__(self):
        self.reg = KNeighborsRegressor()
        self.model_name = "K Nearest Neighbors"

class DT(GeneralStrategy):
    def __init__(self):
        self.reg = DecisionTreeRegressor()
        self.model_name = "Decision tree"

class RF(GeneralStrategy):
    def __init__(self):
        self.reg = RandomForestRegressor()
        self.model_name = "Random forest"

class SVM(GeneralStrategy):
    def __init__(self):
        self.reg = SVR()
        self.model_name = "Support vector machine"
    
