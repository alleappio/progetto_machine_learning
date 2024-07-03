import pandas as pd
import numpy as np

from sklearn.feature_selection import RFE
from sklearn.feature_selection import SequentialFeatureSelector
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
        self.feature_selector = SequentialFeatureSelector(estimator = self.reg, n_features_to_select=n_features, scoring='neg_mean_absolute_error', n_jobs=-1)
    
    def cut_dataset(self, records_number):
        self.X_train = self.X_train.sample(n=records_number)
        self.y_train = self.y_train[self.X_train.index] 

    def calc_rfe(self):
        # self.pipe.fit(self.X_train, self.y_train)
        self.feature_selector.fit(self.X_train, self.y_train)

    def get_new_X(self):
        return self.feature_selector.transform(self.X_train)

    def get_selected_features(self):
        return self.feature_selector.get_feature_names_out()
