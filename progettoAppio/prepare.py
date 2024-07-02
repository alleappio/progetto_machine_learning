import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

class PrepareData:
    def __init__(self, data, target = 'critical_temp'):
        self.data = data
        self.target = target
        self.train_data, self.test_data = train_test_split(self.data)
        self.train_data = self.train_data.reset_index(drop=True)
        self.test_data=self.test_data.reset_index(drop=True)
        
    def get_train_data(self):
        y_train = self.train_data[self.target]
        X_train = self.train_data.drop(self.target, axis=1)
        return X_train, y_train

    def get_test_data(self):
        y_test = self.test_data[self.target]
        X_test = self.test_data.drop(self.target, axis=1)
        return X_test, y_test

