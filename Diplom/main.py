

import PlotInit
import ApproxProcess
import PreProcess


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

parametrsShouldBe = { "A0" : 2, "A1" : 1, "A2": 0.04, 'mean0': -0.7, 'mean1': -0.1, 'mean2': 0.4 }
parameters = initValues.getParameters()
print(parametrsShouldBe)
print(parameters)


PlotInit.plt.show()