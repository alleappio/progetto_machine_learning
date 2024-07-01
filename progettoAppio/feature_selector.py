import pandas as pd
import numpy as np

from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR

from sklearn.feature_selection import RFE
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
# from mlxtend.feature_selection import ExhaustiveFeatureSelector as EFS

class FeatureSelectorFilter:
    def __init__(self, dataset, target):
        self.dataset = dataset
        self.target = target
        self.calc_correlations()

    def calc_correlations(self):
        self.correlations = self.dataset.corr()[self.target]
        
    def select_from_threshold(self, threshold):
        corr_dict = self.correlations.to_dict()
        winners = []
        for i in corr_dict:
            if np.abs(corr_dict[i])>=threshold:
                winners.append(i)
        return winners
    
            
class FeatureSelectorWrapper:
    def __init__(self, X_train, y_train, strategy, n_features=20):
        self.X_train = X_train
        self.y_train = y_train
        self.reg = strategy
 
        self.rfe = RFE(estimator = self.reg, n_features_to_select=n_features, step=1)
    
    def calc_rfe(self):
        # self.pipe.fit(self.X_train, self.y_train)
        self.rfe.fit(self.X_train, self.y_train)

    def get_new_X(self):
        return self.rfe.transform(self.X_train)
