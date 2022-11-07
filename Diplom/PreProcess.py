
import numpy as np
import math as mt
import statistics as st

class InitialProcess:
    def __init__(self, x, y):
        self.__x = x
        self.__y = y

        self.__A = []
        self.__MEAN = []
        self.__SIGMA = []

        self.__mode = "Null"
        self.__parameters = {}

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
        self.__parameters["A1"] = -1
        self.__parameters["mean1"] = -0.25
        self.__parameters["sigma1"] = 0.04

    def threeModal(self):
        self.__parameters.clear()
        std = np.std(np.array(self.__y))
        # mean = np.mean(np.array(self.__y))
        mean = st.median(np.array(self.__y))
        self.__parameters["A0"] = 3
        self.__parameters["mean0"] = 0
        self.__parameters["sigma0"] = 0.04
        self.__parameters["A1"] = -1
        self.__parameters["mean1"] = -0.25
        self.__parameters["sigma1"] = 0.04
        self.__parameters["A2"] = -0.25
        self.__parameters["mean2"] = 0.25
        self.__parameters["sigma2"] = 0.04

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