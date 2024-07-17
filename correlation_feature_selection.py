import pandas as pd
import numpy as np

class CorrelationFeatureSelection(object):
    def __init__(self, threshold):
        self.threshold = threshold
    
    def fit(self, X, y):
        self.target = y.name
        dataset = X
        dataset[y.name] = y
        self.correlations = dataset.corr()[self.target]
        self.columns = self.select_from_threshold(self.threshold)
        return self

    def transform(self, X):
        return X[self.columns]
    
    def select_from_threshold(self, threshold):
        corr_dict = self.correlations.to_dict()
        winners = []
        for i in corr_dict:
            if np.abs(corr_dict[i])>=threshold and i!=self.target:
                winners.append(i)

        return winners

    def __str__(self):
        return f"CorrelationFeatureSelection(threshold={self.threshold})"
    
    def __repr__(self):
        return f"CorrelationFeatureSelection(threshold={self.threshold})"
