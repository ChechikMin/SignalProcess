

import PlotInit
import ApproxProcess
# Press the green button in the gutter to run the script.



init = PlotInit.Initial()
x = init.getPointsX()
y = init.getPointsY()
init.plotInitialRithm(x, y)
init.plotAnalytics(x)

model = ApproxProcess.Approximation(x, y, { "sigma": 0.04, "a": 3, "mean":0 })

model.calculate()
#model.setPlotFunc(init.plotInitialRithm)

model.plot()
PlotInit.plt.legend(('File', 'Formula', 'Test'),
                   loc='upper right', shadow=True)
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