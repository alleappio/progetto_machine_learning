import pandas as pd
import numpy as pd

from matplotlib import pyplot as plt

class Plotter:
    def __init__(self):
        self.prediction_lists={}

    def add_prediction(self, title, y_pred, y_test):
        self.prediction_lists[title] = [y_pred, y_test]

    def show(self):
        figure, axis = plt.subplots(2, 2)
        x, y = 0, 0
        for i in self.prediction_lists:
            plt.figure(figure)
            axis[x,y].plot(self.prediction_lists[i][0], self.prediction_lists[i][1], 'bo')
            axis[x,y].set_title(i)
            if(x==1):
                y+=1
                x=0
            else:
                x+=1
        plt.show()
        

    def save_plot(self, save_dir, title):
        figure, axis = plt.subplots(2, 2)
        x, y = 0, 0
        for i in self.prediction_lists:
            plt.figure(figure)
            axis[x,y].plot(self.prediction_lists[i][0], self.prediction_lists[i][1], 'bo')
            axis[x,y].set_title(i)
            if(x==1):
                y+=1
                x=0
            else:
                x+=1
        plt.gcf().set_size_inches(30, 17)
        plt.savefig(f"{save_dir}/{title}", dpi = 65)

    def save_single_plot(self, y_pred, y_test, save_dir, title):
        plt.plot(y_pred, y_test, 'bo', markersize=5)
        plt.title(title)
        plt.savefig(f"{save_dir}/{title}")
