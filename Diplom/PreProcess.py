
import numpy as np
import math as mt
import statistics as st
from numpy import diff

class InitialProcess:
    def __init__(self, x, y):
        self.__x = x
        self.__y = y

        self.__A = []
        self.__MEAN = []
        self.__SIGMA = []

        self.__mode = "Null"
        self.__parameters = {}

        self.__indexesOfAmplitude = []
        self.__indexesOfMin = []

    def setMode(self, mode : str):
        self.__mode = mode

    def setX(self, x):
        self.__x = x

    def setY(self, y):
        self.__y = y

    def calcInit(self):

        if self.__mode == "OneModal":
            self.oneModal()
        elif self.__mode == "BiModal":
            self.biModal()
        elif self.__mode == "ThreeModal":
            self.threeModal()
        elif self.__mode == "FourModal":
            pass
        elif self.__mode == "FiveModal":
            pass


    def oneModal(self):
        self.__parameters.clear()
        std = np.std(np.array(self.__y))
        #mean = np.mean(np.array(self.__y))
        mean = st.median(np.array(self.__y))
        self.__parameters["A0"] = 1 / (np.sqrt(2 * mt.pi) * std)
        self.__parameters["mean0"] = mean
        self.__parameters["sigma0"] = std

    def biModal(self):
        self.__parameters.clear()
        std = np.std(np.array(self.__y))
        # mean = np.mean(np.array(self.__y))
        mean = st.median(np.array(self.__y))
        self.__parameters["A0"] = 3
        self.__parameters["mean0"] = 0
        self.__parameters["sigma0"] = 0.04
        self.__parameters["A1"] = 1
        self.__parameters["mean1"] = 0.25
        self.__parameters["sigma1"] = 0.04

    def threeModal(self):
        #написать сюда алгоритм определения начальных параметров трёх гауссов
        self.__parameters.clear()

        #для начала написать алгоритм определения максимумов
        self.fullA3Modal()
        self.fullMean3Modal()
        self.fullSigma3Modal()

    def fullA3Modal(self):

        #dx = np.pi / 10
        #x = np.arange(0, 2 * np.pi, np.pi / 10)

        # we calculate the derivative, with np.gradient
        #plt.plot(x, np.gradient(np.sin(x), dx), '-*', label='approx')

        dx = self.__x[1] - self.__x[0]
        dy = diff(self.__y) / dx
        sign = lambda x: mt.copysign(1, x)

        index = 0
        k = 0
        self.__indexesOfMin.append(k)

        while k < len(dy):
            if sign(dy[k]) == -1:
                self.__parameters["A" + str(index)] = self.__y[k]
                index += 1
                self.__indexesOfAmplitude.append(k)
                while sign(dy[k]) == -1:
                    k += 1
                    if k >= len(dy):
                        return
                self.__indexesOfMin.append(k)
            k += 1

        #self.__parameters["A0"] = 2
        #self.__parameters["A1"] = 1
        #self.__parameters["A2"] = 0.04

    def fullMean3Modal(self):

        self.__parameters["mean0"] = self.__x[self.__indexesOfAmplitude[0]]
        self.__parameters["mean1"] = self.__x[self.__indexesOfAmplitude[1]]
        self.__parameters["mean2"] = self.__x[self.__indexesOfAmplitude[2]]

        # self.__parameters["mean0"] = -0.7
        # self.__parameters["mean1"] = -0.1
        # self.__parameters["mean2"] = 0.4

    def fullSigma3Modal(self):

        self.__parameters["sigma0"] = np.var( np.array( self.__y[ self.__indexesOfMin[0] :  self.__indexesOfMin[1] ] ) )

        #self.__parameters["sigma0"] = 0.24
        # self.__parameters["sigma0"] = 0.14
        # self.__parameters["sigma0"] = 0.14
        self.__parameters["sigma1"] = np.var( np.array( self.__y[ self.__indexesOfMin[1] :  self.__indexesOfMin[2] ] ) )
        self.__parameters["sigma2"] = np.var( np.array( self.__y[ self.__indexesOfMin[2] :  len(self.__y) ] ) )


    def getParameters(self):
        i = 0
        if not self.__parameters:
            print("Empty param")

        for a, mean, sigma in zip(self.__A, self.__MEAN, self.__SIGMA):
            self.__parameters["A" + str(i)] = a
            self.__parameters["MEAN" + str(i)] = mean
            self.__parameters["SIGMA" + str(i)] = sigma
            i += 1

        return self.__parameters