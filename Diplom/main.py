# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

from scipy.fft import fft, fftfreq, ifft
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import time

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    df = pd.read_csv("ExampleSignal_21_N=2000001_22n.txt", header=None)
    #x = df[0]
    y = df[0]

    yf = fft(np.array(y))
    N = len(y)
    print(N)
    T = 2e-4
    start_time = time.time()
    xf = fftfreq(N, T)[:N//2]

    print("--- %s seconds ---" % (time.time() - start_time))

    plt.plot(xf[:30], np.abs(yf[0:30]) / 2e4)

    plt.xlabel("Hz")
    plt.ylabel("PowerSpectrum,%")

    plt.grid()

    plt.show()
# See PyCharm help at https://www.jetbrains.com/help/pycharm/
