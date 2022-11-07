import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


class Initial:

    def __init__(self):
        df = pd.read_csv("NewPoints.txt", header=None, delimiter="\t")

        self.__x = df[0]
        self.__y = df[1]
        #self.__analytic = lambda x: (x + 0.8)*(x + 0.3)*(x - 0.8) + 3*np.exp(-(x**2 / (2 * 0.04**2)))

    def plotInitialRithm(self, x, y):

        plt.plot(x, y, 'o', markersize=4, label='data')
        plt.grid()

    def plotAnalytics(self, x):

        y = self.__analytic(x)
        self.plotInitialRithm(x, y)

    def getPointsX(self):
        return self.__x

    def getPointsY(self):
        return self.__y
