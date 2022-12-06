
import numpy as np
import AlgorithmApp
import time
from scipy.fft import fft, fftfreq
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from scipy.optimize import leastsq


SIGMA = "sigma"
A = "A"
MEAN = "mean"


class Approximation:

    def __init__(self, pointsX, pointsY, parameters, model):

        self.__parameters = parameters

        self.__x = pointsX
        self.__y = pointsY
        self.__plotFunc = lambda x, y: plt.plot(x, y)
        self.__modelFunc = model

    def calculate(self, mode:int):

        x0 = self.unpackInit(mode)
        #x0 = np.array([3, 0, 0.04])
        print(x0)

        if len(x0) < 3 * mode:
            print("Little init parametrs")

        algorithm = AlgorithmApp.AlgNewton(mode, x0)
        #algorithm = AlgorithmApp.Newton_Conjugate_Gradient(mode, x0)
        #algorithm = AlgorithmApp.CurveFit(mode, x0)
        algorithm.setPoints(self.__x, self.__y)
        algorithm.setModel(self.__modelFunc)
        res = algorithm.process()
        resY = res.x
        #resY = res
        print(resY)

        self.__x = np.linspace(-1, 1)

        self.__y = self.__modelFunc(resY, self.__x)
        #mean = res.x[0]
        #amplitude = res.x[1]
        #deviation = res.x[2]

        #self.__y = np.array( [ amplitude*np.exp(-((x - mean)**2 / (2 * deviation**2))) for x in self.__x ] )

    def setPlotFunc(self, plotfunc):
        self.__plotFunc = plotfunc

    def unpackInit(self, mode : int):
        x0 = []
        for i in range(mode):# depends on parameters size or peaks
            x0.append(self.__parameters["A" + str(i)])
            x0.append(self.__parameters["mean" + str(i)])
            x0.append(self.__parameters["sigma" + str(i)])

        return np.array(x0)

    def plot(self):
        self.__plotFunc(self.__x, self.__y)
