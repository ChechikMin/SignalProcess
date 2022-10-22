
import numpy as np
import time
from scipy.fft import fft, fftfreq
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from scipy.optimize import leastsq
from scipy.optimize import least_squares


class Approximation:

    def __init__(self, pointsX, pointsY, parameters):

        self.__parameters = parameters

        self.__x = pointsX
        self.__y = pointsY
        self.__plotFunc = lambda x, y: plt.plot(x, y)

    def calculate(self):

        def model(x, u):
            return x[0] * np.exp(-((u - x[1]) ** 2 / (2 * x[2] ** 2)))

        def fun(x, u, y):
            return model(x, u) - y

        def jac(x, u, y):

            J = np.empty((u.size, x.size))

            expa = x[0] * np.exp( - (u - x[1])**2/(2*x[2]**2) )

            J[:, 0] = expa / x[0]

            J[:, 1] = - expa * (u - x[1]) / x[2]**2

            J[:, 2] = expa * (u - x[1])**2 / x[2]**3

            return J

        x0 = np.array([0.1, 0, 0.1])
        res = least_squares(fun, x0, jac=jac, bounds=(-1, 50), args=(self.__x, self.__y), verbose=1)
        print(res.x)

        self.__x = np.linspace(-1, 1)

        self.__y = model(res.x, self.__x)
        #mean = res.x[0]
        #amplitude = res.x[1]
        #deviation = res.x[2]

        #self.__y = np.array( [ amplitude*np.exp(-((x - mean)**2 / (2 * deviation**2))) for x in self.__x ] )

    def setPlotFunc(self, plotfunc):
        self.__plotFunc = plotfunc

    def plot(self):
        self.__plotFunc(self.__x, self.__y)
