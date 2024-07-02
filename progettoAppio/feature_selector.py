import pandas as pd
import numpy as np

from sklearn.feature_selection import RFE
from sklearn.feature_selection import SequentialFeatureSelector
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
        self.feature_selector = SequentialFeatureSelector(estimator = self.reg, n_features_to_select=n_features, scoring='neg_mean_absolute_error')
    
    def cut_dataset(self, step=10, drop_num=1):
        i=0
        if drop_num>step:
            print("Invalid parameters in cut database")
            exit()
        dataset_indexes=list(self.X_train.index.values)
        print(f"before cut: {self.X_train.shape}")
        while i<self.X_train.shape[0]:
            for j in range(drop_num):
                self.X_train = self.X_train.drop(labels=dataset_indexes[i+j], axis=0)
                self.y_train = self.y_train.drop(labels=dataset_indexes[i+j], axis=0)
            i+=step
        print(f"after cut: {self.X_train.shape}")

    def calc_rfe(self):
        # self.pipe.fit(self.X_train, self.y_train)
        self.feature_selector.fit(self.X_train, self.y_train)

    def get_new_X(self):
        return self.feature_selector.transform(self.X_train)

    def get_selected_features(self):
        return self.feature_selector.get_feature_names_out()
