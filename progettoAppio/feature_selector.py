import pandas as pd
import numpy as np
class FeatureSelector:
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
            if np.abs(corr_dict[i])>threshold:
                winners.append(i)
        return winners
            
