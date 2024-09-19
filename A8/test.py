import matplotlib.pyplot as plt
import numpy as np


def dft_plotter(dft_Y, freq_array, func, filename, xlim=15):
    plt.figure()
    plt.subplot(2, 1, 1)
    plt.plot(freq_array, abs(dft_Y), lw=2)
    plt.xlim([-xlim, xlim])
    plt.title("Spectrum of signal y = {}".format(func))
    plt.ylabel(r"$|Y|$", size=16)
    plt.grid(True)
    plt.subplot(2, 1, 2)
    ii = np.where(
        abs(dft_Y) > 1e-3
    )  # find the indices of the appreciably large elements
    plt.plot(freq_array[ii], np.angle(dft_Y[ii]), "go", lw=2)
    plt.xlim([-xlim, xlim])
    plt.grid(True)
    plt.ylabel(r"$\angle Y$", size=16)
    plt.xlabel(r"$k$", size=16)
    plt.savefig(filename)
    plt.show()


# this is the correct way
t = np.linspace(-4 * np.pi, 4 * np.pi, 513)
t = t[:-1]  # remove the last element
y = (1 + 0.1 * np.cos(t)) * np.cos(10 * t)
Y = np.fft.fftshift(np.fft.fft(y)) / 512
w = np.linspace(-64, 64, 513)
w = w[:-1]  # remove the last element
dft_plotter(Y, w, func=r"$(1+0.1\cos(t))\cos(10t)$", filename="test1.png")


# this method increased the number of points; higher sampling freq
t = np.linspace(0, 2 * np.pi, 513)
t = t[:-1]  # remove the last element
y = (1 + 0.1 * np.cos(t)) * np.cos(10 * t)
Y = np.fft.fftshift(np.fft.fft(y)) / 512
w = np.linspace(
    -64, 64, 513
)  # but why is this 64? shouldnt be because sampling rate changed
w = w[:-1]  # remove the last element
dft_plotter(Y, w, func=r"$(1+0.1\cos(t))\cos(10t)$", filename="test2.png")
# plt.show()

# changing sampling freq in time domain; 64->256 as wmax = (512/2pi)*2pi halved into + and -
t = np.linspace(0, 2 * np.pi, 513)
t = t[:-1]  # remove the last element
y = (1 + 0.1 * np.cos(t)) * np.cos(10 * t)
Y = np.fft.fftshift(np.fft.fft(y)) / 512
w = np.linspace(-256, 256, 513)
w = w[:-1]  # remove the last element
dft_plotter(Y, w, func=r"$(1+0.1\cos(t))\cos(10t)$", filename="test3.png")
# plt.show()
# you'll see that still we didnt get 4 peaks as we expected
# the plot is exactly the same as we got with 128 samples
# reason - we didnt get higher resolution in w domain
