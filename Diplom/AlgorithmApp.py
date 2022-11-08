

import numpy as np
from scipy.optimize import least_squares
from scipy.optimize import minimize

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

    def jac3(self, x, u):  # For one gauss

        J = np.empty((u.size, x.size))

        expa = 0

        for i in range(self.mode):
            expa = x[i * 3] * np.exp(- (u - x[3 * i + 1]) ** 2 / (2 * x[3 * i + 2] ** 2))

        J[:, 0] = expa / x[0] + expa / x[3] + expa / x[6]

        J[:, 1] = - expa * (u - x[1]) / x[2] ** 2 - expa * (u - x[4]) / x[5] ** 2 - expa * (u - x[7]) / x[8] ** 2

        J[:, 2] = expa * (u - x[1]) ** 2 / x[2] ** 3 + expa * (u - x[4]) ** 2 / x[5] ** 3 + expa * (u - x[7]) / x[
            8] ** 3

        return J

    def jac2(self, x, u):  # For one gauss

        J = np.empty((u.size, x.size))

        expa = 0

        for i in range(self.mode):
            expa = x[i * 3] * np.exp(- (u - x[3 * i + 1]) ** 2 / (2 * x[3 * i + 2] ** 2))

        J[:, 0] = expa / x[0] + expa / x[3]

        J[:, 1] = - expa * (u - x[1]) / x[2] ** 2 - expa * (u - x[4]) / x[5] ** 2

        J[:, 2] = expa * (u - x[1]) ** 2 / x[2] ** 3 + expa * (u - x[4]) ** 2 / x[5] ** 3

        return J

    def jac1(self, x, u):  # For two gauss

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
            bounds = (-1, 50), args=(self._x, self._y), verbose=1
            )


class Newton_Conjugate_Gradient(Algorithm):
    def __init__(self, mode : int, x0):
        super().__init__(mode, x0)

    def process(self):
        res = []
        for x,y in zip(self._x, self._y):
            res = minimize(super().fun, self.x0, method='nelder-mead',

                       args=(x, y), jac=super().jac3, hess=self.Hessian,

                            options={'xtol': 1e-8, 'disp': True})
            print(res)

        return res

    def Hessian(self):
        pass#write hessian


