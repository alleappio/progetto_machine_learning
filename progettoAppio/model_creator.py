import pandas as pd
import numpy as np

from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVR
from sklearn.pipeline import Pipeline

from sklearn.feature_selection import RFE
from sklearn.feature_selection import SequentialFeatureSelector

from sklearn import metrics
from matplotlib import pyplot as plt

from correlation_feature_selection import CorrelationFeatureSelection

class ModelCreator:
    def __init__(self, name):
        self.pipe_list = []
        self.name = name
    
    def set_model_estimator(self, estimator):
        self.estimator = estimator

    def set_pipe_recursive_feature_selection(self, scoring_rule='r2', n_features='auto'):
        self.pipe_list.append(
            ("SFE", SequentialFeatureSelector(estimator = self.estimator, n_features_to_select=n_features, scoring=scoring_rule, n_jobs=-1))
        )

    def set_pipe_corr_feature_selection(self, threshold=0.4):
        self.pipe_list.append(
            ("CFS", CorrelationFeatureSelection(threshold))
        )
    
    def set_pipe_estimator(self):
        self.pipe_list.append(
            ("estimator", self.estimator)
        )
    
    def do_pipe_grid_search(self, hyper_parameters, scoring_rule, n_cv, X_train, y_train):
        self.grid_search = GridSearchCV(self.pipe, param_grid=hyper_parameters, scoring=scoring_rule, cv=n_cv, n_jobs=-1, refit=True)
        self.grid_search.fit(X_train,y_train)
        self.pipe = self.grid_search.best_estimator_

    def assemble_pipe(self):
        self.pipe = Pipeline(self.pipe_list)
        return self.pipe

    def get_pipe(self):
        return self.pipe
