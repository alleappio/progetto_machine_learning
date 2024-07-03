import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

class PrepareData:
    def __init__(self, data, target = 'critical_temp', normalization=True):
        self.data = data
        self.target = target
        self.train_data, temp_data = train_test_split(self.data, test_size=0.4)
        self.val_data, self.test_data = train_test_split(temp_data, test_size=0.5)
        
        self.train_data = self.train_data.reset_index(drop=True)
        self.val_data = self.val_data.reset_index(drop=True)
        self.test_data=self.test_data.reset_index(drop=True)
        
        self.split_X_y()
        if normalization:
            self.normalization()

    def split_X_y(self):
        self.y_train = self.train_data[self.target]
        self.X_train = self.train_data.drop(self.target, axis=1)
        self.y_val = self.val_data[self.target]
        self.X_val = self.val_data.drop(self.target, axis=1)
        self.y_test = self.test_data[self.target]
        self.X_test = self.test_data.drop(self.target, axis=1)

    def normalization(self):
        mms = MinMaxScaler()
        
        mms.fit(self.X_train)
        self.X_train[self.X_train.columns] = mms.transform(self.X_train[self.X_train.columns])
        
        mms.fit(self.X_val)
        self.X_val[self.X_val.columns] = mms.transform(self.X_val[self.X_val.columns])
        
        mms.fit(self.X_test)
        self.X_test[self.X_test.columns] = mms.transform(self.X_test[self.X_test.columns])
    
    def get_train_data(self):
        return self.X_train, self.y_train

    def get_val_data(self):
        return self.X_val, self.y_val
    
    def get_test_data(self):
        return self.X_test, self.y_test

