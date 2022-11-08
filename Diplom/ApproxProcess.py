
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

    def __init__(self, pointsX, pointsY, parameters):

        self.__parameters = parameters

        self.__x = pointsX
        self.__y = pointsY
        self.__plotFunc = lambda x, y: plt.plot(x, y)

    def calculate(self, mode:int):

        def model(x, u):
            return x[0] * np.exp(-((u - x[1]) ** 2 / (2 * x[2] ** 2))) + x[3] * np.exp(-((u - x[4]) ** 2 / (2 * x[5] ** 2)))

        x0 = self.unpackInit(mode)
        #x0 = np.array([3, 0, 0.04])
        print(x0)

        if len(x0) < 3 * mode:
            print("Little init parametrs")

        #algorithm = AlgorithmApp.AlgNewton(mode, x0)
        algorithm = AlgorithmApp.Newton_Conjugate_Gradient(mode, x0)
        algorithm.setPoints(self.__x, self.__y)
        algorithm.setModel(model)
        res = algorithm.process()

        print(res.x)

        self.__x = np.linspace(-1, 1)

        self.__y = model(res.x, self.__x)
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
