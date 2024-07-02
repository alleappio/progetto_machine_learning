import pandas as pd
import numpy as pd

from matplotlib import pyplot as plt

class Plotter:
    def __init__(self):
        self.prediction_lists={}

    def add_prediction(self, title, y_pred, y_test):
        self.prediction_lists[title] = [y_pred, y_test]
        print(self.prediction_lists)

    def show(self):
        print(self.prediction_lists)
        for i in self.prediction_lists:
            plt.plot(self.prediction_lists[i][0], self.prediction_lists[i][1], 'bo', label=i)
        plt.show()
