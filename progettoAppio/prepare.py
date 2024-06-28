import pandas as pd
import numpy as np

class PrepareData:
    def __init__(self, data):
        self.data = data
        self.shuffle()
        self.divide()

    def shuffle(self):
        self.data = self.data.sample(n = len(self.data))
        self.data = self.data.reset_index(drop = True)
    
    def divide(self):
        self.train_data, self.validate_data, self.test_data = np.split(self.data, [int(.6*len(self.data)), int(.8*len(self.data))])

    def get_train_data(self):
        return self.train_data

    def get_validate_data(self):
        return self.validate_data

    def get_test_data(self):
        return self.test_data

