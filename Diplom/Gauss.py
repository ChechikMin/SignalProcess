

import PlotInit
import ApproxProcess
import PreProcess
# Press the green button in the gutter to run the script.

def derModel(x,u):
    return u * x[0] * ApproxProcess.np.exp(-((u - x[1]) ** 2 / (2 * x[2] ** 2))) + \
                     u * x[3] * ApproxProcess.np.exp(-((u - x[4]) ** 2 / (2 * x[5] ** 2))) + \
                    u * x[6] * ApproxProcess.np.exp(-((u - x[7]) ** 2 / (2 * x[8] ** 2)))


def model(x, u):
    return x[0] * ApproxProcess.np.exp(-((u - x[1]) ** 2 / (2 * x[2] ** 2))) + \
           x[3] * ApproxProcess.np.exp(-((u - x[4]) ** 2 / (2 * x[5] ** 2))) + \
           x[6] * ApproxProcess.np.exp(-((u - x[7]) ** 2 / (2 * x[8] ** 2)))

modes = {"ThreeModal":  3, "BiModal": 2, "OneModal": 1}

modeler = "ThreeModal"
hint = modes.get(modeler)

init = PlotInit.Initial()
x = init.getPointsX()
y = init.getPointsY()
initValues = PreProcess.InitialProcess(x, y)
initValues.setMode(modeler)
initValues.calcInit()
init.plotInitialRithm(x, y)
#init.plotAnalytics(x)

#parametrs = { ApproxProcess.SIGMA : 0.04, ApproxProcess.A : 3, ApproxProcess.MEAN:0 }
parameters = initValues.getParameters()

model = ApproxProcess.Approximation(x, y, parameters, derModel)

model.calculate(hint)
model.setPlotFunc(init.plotInitialRithm)

model.plot()
PlotInit.plt.legend(('Rithm', 'Fitting'),
                   loc='upper right', shadow=True)
PlotInit.plt.xlabel("x")
PlotInit.plt.ylabel("Signal")
PlotInit.plt.grid()
PlotInit.plt.show()

#
# def rosen(x):
#     """The Rosenbrock function"""
#
#     return x[0] * np.exp(-((x - x[1]) ** 2 / (2 * x[2] ** 2)))
#
#
# x0 = np.array([1.3, 0.7, 0.8, 1.9, 1.2])
#
# res = minimize(rosen, x0, method='nelder-mead',
#
#                options={'xatol': 1e-8, 'disp': True})