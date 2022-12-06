

from scipy.fft import fft, fftfreq, ifft
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import time

if __name__ == '__main__':
    df = pd.read_csv("PositiveGauss.txt", header=None, delimiter="\t")
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

