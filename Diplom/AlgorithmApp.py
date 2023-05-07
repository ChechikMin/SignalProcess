

import numpy as np
from scipy.optimize import least_squares
from scipy.optimize import minimize
from scipy.optimize import curve_fit
from astropy import modeling

class Algorithm:

    def __init__(self, mode: int, x0):
        self.mode = mode
        self.model = lambda x: 0
        self.x0 = x0
        self._x = []
        self._y = []

    def setPoints(self, x, y):
        self._x = x
        self._y = y

    def setModel(self, model):
        self.model = model

    def process(self):
        pass

    def fun(self,x, u, y):
        return self.model(x, u) - y

    def jac3(self, x, u, y):  # For three gauss

        J = np.empty((u.size, x.size))

        expa = np.zeros((self.mode, len(u)))

        for i in range(self.mode):
            expa[i] = x[i * 3] * np.exp(- (u - x[3 * i + 1]) ** 2 / (2 * x[3 * i + 2] ** 2))

        J[:, 0] = expa[0] / x[0]

        J[:, 1] = - expa[0] * (u - x[1]) / x[2] ** 2

        J[:, 2] = expa[0] * (u - x[1]) ** 2 / x[2] ** 3

        J[:, 3] = expa[1] / x[3]

        J[:, 4] = - 2 * expa[1] * (u - x[4]) / x[5] ** 2

        J[:, 5] = expa[1] * (u - x[4]) ** 2 / x[5] ** 3

        J[:, 6] = expa[2] / x[6]

        J[:, 7] = -expa[2] * (u - x[7]) / x[8] ** 2

        J[:, 8] = expa[2] * (u - x[7]) ** 2 / x[8] ** 3

        return J

    def jac2(self, x, u, y):  # For one gauss

        J = np.empty((u.size, x.size))

        expa = 0

        for i in range(self.mode):
            expa += x[i * 3] * np.exp(- (u - x[3 * i + 1]) ** 2 / (2 * x[3 * i + 2] ** 2))

        J[:, 0] = expa / x[0] + expa / x[3]

        J[:, 1] = - expa * (u - x[1]) / x[2] ** 2 - expa * (u - x[4]) / x[5] ** 2

        J[:, 2] = expa * (u - x[1]) ** 2 / x[2] ** 3 + expa * (u - x[4]) ** 2 / x[5] ** 3

        return J

    def jac1(self, x, u, y):  # For two gauss

        J = np.empty((u.size, x.size))

        expa = x[0] * np.exp(- (u - x[1]) ** 2 / (2 * x[2] ** 2))

        J[:, 0] = expa / x[0]

        J[:, 1] = - expa * (u - x[1]) / x[2] ** 2

        J[:, 2] = expa * (u - x[1]) ** 2 / x[2] ** 3

        return J


class AlgNewton(Algorithm):

    def __init__(self, mode : int, x0):
        super().__init__(mode, x0)

    def process(self):
        return least_squares\
                (
            super().fun, self.x0, jac=super().jac3,
            bounds = (-5, 5), args=(self._x, self._y), verbose=1,
                loss='huber', xtol = 1e-10
            )


class Newton_Conjugate_Gradient(Algorithm):
    def __init__(self, mode : int, x0):
        super().__init__(mode, x0)

    def model(self, x, u):
        return x[0] * np.exp(-((u - x[1]) ** 2 / (2 * x[2] ** 2))) + \
               x[3] * np.exp(-((u - x[4]) ** 2 / (2 * x[5] ** 2))) + \
               x[6] * np.exp(-((u - x[7]) ** 2 / (2 * x[8] ** 2)))

    def process(self):
        res = []
        for x,y in zip(self._x, self._y):
            res = minimize(self.model, self.x0, method='nelder-mead',

                       args= ( y ), jac=super().jac3, hess=self.Hessian,

                            options={'xtol': 1e-8, 'disp': True})
            if (self.model(res.x, x) - y)**2 < 1e-10:
                break
        return res



    def Hessian(self, x, y):

        expa = np.zeros((self.mode, len(x)))

        for i in range(self.mode):
            expa[i] = x[i * 3] * np.exp(- (y - x[3 * i + 1]) ** 2 / (2 * x[3 * i + 2] ** 2))

        x = np.asarray(x)

        diagonalAbove = np.zeros_like(x)
        diagonalAbove[0] = (expa[0] * (y - x[1]) ** 2 / x[2] ** 2) / x[0]
        diagonalAbove[1] = -expa[0] * ( 2 * (x[1] - y)/(2*x[2]**3) + (x[1] - y)/(2*x[2]**5))
        diagonalAbove[2] = 0
        diagonalAbove[3] = (expa[1] * (y - x[4]) ** 2 / x[5] ** 2) / x[3]
        diagonalAbove[4] = -expa[1] * (2 * (x[4] - y) / (2 * x[5] ** 3) + (x[4] - y) / (2 * x[5] ** 5))
        diagonalAbove[5] = 0
        diagonalAbove[6] = (expa[2] * (y - x[7]) ** 2 / x[8] ** 2) / x[6]
        diagonalAbove[7] = -expa[2] * (2 * (x[7] - y) / (2 * x[8] ** 3) + (x[7] - y) / (2 * x[8] ** 5))
        diagonalAbove[8] = 0

        #H = np.diag(-400 * x[:-1], 1) - np.diag(400 * x[:-1], -1)

        diagonalMain = np.zeros_like(x)

        diagonalMain[0] = 0
        diagonalMain[1] = expa[0] * ( 1 / x[2]**2 + (y - x[1]) / x[2]**4)
        diagonalMain[2] = expa[0] * ( 3 * (y -x[1])**2 / x[2]**4 + (y - x[1])**4 / x[2]**6 )
        diagonalMain[3] = 0
        diagonalMain[4] = expa[1] *( 1 / x[5]**2 + (y - x[4]) / x[5]**4)
        diagonalMain[5] = expa[1] * ( 3 * (y -x[5])**2 / x[4]**4 + (y - x[4])**4 / x[5]**6 )
        diagonalMain[6] = 0
        diagonalMain[7] = expa[2] *( 1 / x[8]**2 + (y - x[7]) / x[8]**4)
        diagonalMain[8] = expa[2] * ( 3 * (y -x[7])**2 / x[8]**4 + (y - x[7])**4 / x[8]**6 )

        H = np.diag(diagonalAbove, 1) + np.diag(diagonalMain) + np.diag(diagonalAbove, -1)

        return H


class CurveFit(Algorithm):
    def __init__(self, mode : int, x0):
        super().__init__(mode, x0)

    def getGauss(self, number):
        interval1 = 960
        interval2 = 1132
        if number == 1:
            return self._x[:interval1],self._y[:interval1]
        if number == 2:
            return self._x[interval1:interval2], self._y[interval1:interval2]
        if number == 3:
            return self._x[interval2:600 * number], self._y[interval2:600 * number]

    def process(self):

        def func(x, a, x0, sigma):
            return a * np.exp(-(x - x0) ** 2 / (2 * sigma ** 2))

        x, y = self.getGauss(1)

        # fitter = modeling.fitting.LevMarLSQFitter()
        # model = modeling.models.Gaussian1D()  # depending on the data you need to give some initial values
        # fitted_model = fitter(model, x, y)

        [a1, b1, c1], res1 = curve_fit(func, x, y)
        x, y = self.getGauss(2)
        [a2, b2, c2], res1 = curve_fit(func, x, y)
        x, y = self.getGauss(3)
        [a3, b3, c3], res1 = curve_fit(func, x, y, method="lm")

        return [a1,b1,c1, a2,b2,c2, a3,b3,c3]
