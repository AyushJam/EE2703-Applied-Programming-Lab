################################################
"""
EE2703: Applied Programming Lab 
Assignment 8: The Digital Fourier Transform
Submitted by: Ayush Jamdar EE20B018
"""
################################################

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


# The DFT in Python
# PART 1: Spectrum of signal y = sin(5x)

# first, the direct way
x = np.linspace(0, 2 * np.pi, 128)
y = np.sin(5 * x)
Y = np.fft.fft(y) / 128
plt.figure()
plt.subplot(2, 1, 1)
plt.plot(abs(Y), lw=2)
plt.title("Spectrum of signal y = sin(5x)")
plt.ylabel(r"$|Y|$", size=16)
plt.grid(True)
plt.subplot(2, 1, 2)
plt.plot(np.unwrap(np.angle(Y)), lw=2)
# unwrap adds multiples of ±2π when the phase difference
# between consecutive elements of P are greater than or equal to the jump threshold π radians
plt.grid(True)
plt.ylabel(r"$\angle Y$", size=16)
plt.xlabel("$k$", size=16)
plt.savefig("a8_1.png")
plt.show()


# Using the numpy.fftshift function
x = np.linspace(0, 2 * np.pi, 129)
x = x[:-1]  # remove the last element
y = np.sin(5 * x)
Y = np.fft.fftshift(np.fft.fft(y)) / 128
dft_plotter(Y, np.linspace(-64, 63, 128), func="sin(5x)", filename="a8_2.png")


# PART 2: Amplitude Modulation
# f(t) = (1 +0.1 cos (t))cos (10t)
t = np.linspace(0, 2 * np.pi, 129)
t = t[:-1]  # remove the last element
y = (1 + 0.1 * np.cos(t)) * np.cos(10 * t)
Y = np.fft.fftshift(np.fft.fft(y)) / 128
w = np.linspace(-64, 63, 128)
dft_plotter(Y, w, func=r"$(1+0.1\cos(t))\cos(10t)$", filename="a8_3.png")

# using more samples
t = np.linspace(-4 * np.pi, 4 * np.pi, 513)
t = t[:-1]  # remove the last element
y = (1 + 0.1 * np.cos(t)) * np.cos(10 * t)
Y = np.fft.fftshift(np.fft.fft(y)) / 512
w = np.linspace(-64, 64, 513)
w = w[:-1]  # remove the last element
dft_plotter(Y, w, func=r"$(1+0.1\cos(t))\cos(10t)$", filename="a8_4.png")


# PART 3: Spectrum of sin^3t and cos^3t
t = np.linspace(-4 * np.pi, 4 * np.pi, 513)
t = t[:-1]  # remove the last element
y = np.sin(t) ** 3
Y = np.fft.fftshift(np.fft.fft(y)) / 512
w = np.linspace(-64, 64, 513)
w = w[:-1]  # remove the last element
dft_plotter(Y, w, func=r"$\sin^3t$", filename="a8_5.png")

t = np.linspace(-4 * np.pi, 4 * np.pi, 513)
t = t[:-1]  # remove the last element
y = np.cos(t) ** 3
Y = np.fft.fftshift(np.fft.fft(y)) / 512
w = np.linspace(-64, 64, 513)
w = w[:-1]  # remove the last element
dft_plotter(Y, w, func=r"$\cos^3t$", filename="a8_6.png")

# PART 4: Spectrum of cos(20t+5cos(t))
# Frequency Modulation
t = np.linspace(-4 * np.pi, 4 * np.pi, 513)
t = t[:-1]  # remove the last element
y = np.cos(20 * t + 5 * np.cos(t))
Y = np.fft.fftshift(np.fft.fft(y)) / 512
w = np.linspace(-64, 64, 513)
w = w[:-1]  # remove the last element
dft_plotter(Y, w, xlim=40, func=r"$\cos(20t+5\cos(t))$", filename="a8_7.png")

# PART 5: The DFT of a Gaussian
# f(t) = exp(-t^2/2)
# F(w) = sqrt(2pi) exp(-w^2/2)
# Take - 1: tlim=4pi, wlim=64, N=512
t = np.linspace(-4 * np.pi, 4 * np.pi, 513)
t = t[:-1]  # remove the last element
y = np.exp(-(t**2) / 2)
Y = abs(np.fft.fftshift((np.fft.fft(y)))) / 512
Y = Y * np.sqrt(2 * np.pi) / np.max(Y)
w = np.linspace(-64, 64, 513)
w = w[:-1]  # remove the last element
Y_actual = np.sqrt(2 * np.pi) * np.exp(-(w**2) / 2)
print("Max Error T1 = {}".format((abs(Y - Y_actual).max())))
# Max Error = 5.428817237852918e-16
dft_plotter(Y, w, xlim=15, func=r"$\exp(-t^2/2)$", filename="a8_8.png")

# Take - 2: tlim=8pi, wlim=32, N=512
t = np.linspace(-8 * np.pi, 8 * np.pi, 513)
t = t[:-1]  # remove the last element
y = np.exp(-(t**2) / 2)
# Normalise the spectrum
Y = np.fft.fftshift(abs(np.fft.fft(y))) / 512
Y = Y * np.sqrt(2 * np.pi) / np.max(Y)
w = np.linspace(-32, 32, 513)
w = w[:-1]  # remove the last element
Y_actual = np.sqrt(2 * np.pi) * np.exp(-(w**2) / 2)
print("Max Error T2 = {}".format(np.max(abs(Y - Y_actual))))
# Max Error = 1.010436804753547e-15
dft_plotter(Y, w, xlim=15, func=r"$\exp(-t^2/2)$", filename="a8_9.png")

# Take - 3: tlim=8pi, wlim=64, N=512
t = np.linspace(-8 * np.pi, 8 * np.pi, 513)
t = t[:-1]  # remove the last element
y = np.exp(-(t**2) / 2)
Y = np.fft.fftshift(abs(np.fft.fft(y))) / 512
Y = Y * np.sqrt(2 * np.pi) / np.max(Y)
w = np.linspace(-64, 64, 513)
w = w[:-1]  # remove the last element
Y_actual = np.sqrt(2 * np.pi) * np.exp(-(w**2) / 2)
print("Max Error T3 = {}".format(np.max(abs(Y - Y_actual))))
# Max Error = 1.692676175087995
dft_plotter(Y, w, xlim=15, func=r"$\exp(-t^2/2)$", filename="a8_10.png")

# Take - 4: tlim=8pi, wlim=32, N=512
t = np.linspace(-8 * np.pi, 8 * np.pi, 513)
t = t[:-1]  # remove the last element
y = np.exp(-(t**2) / 2)
Y = np.fft.fftshift(abs(np.fft.fft(y))) / 512
Y = Y * np.sqrt(2 * np.pi) / np.max(Y)
w = np.linspace(-32, 32, 513)
w = w[:-1]  # remove the last element
Y_actual = np.sqrt(2 * np.pi) * np.exp(-(w**2) / 2)
print("Max Error T4 = {}".format(np.max(abs(Y - Y_actual))))
dft_plotter(Y, w, xlim=15, func=r"$\exp(-t^2/2)$", filename="a8_11.png")

# Take - 5: tlim=4pi, wlim=32, N=1024
t = np.linspace(-4 * np.pi, 4 * np.pi, 1025)
t = t[:-1]  # remove the last element
y = np.exp(-(t**2) / 2)
Y = np.fft.fftshift(abs(np.fft.fft(y))) / 1024
w = np.linspace(-32, 32, 1025)
w = w[:-1]  # remove the last element
Y = Y * np.sqrt(2 * np.pi) / np.max(Y)
Y_actual = np.sqrt(2 * np.pi) * np.exp(-(w**2) / 2)
print("Max Error T5 = {}".format(np.max(abs(Y - Y_actual))))
dft_plotter(Y, w, xlim=15, func=r"$\exp(-t^2/2)$", filename="a8_12.png")

##################################################################################

# EDIT/TEST after submission
def gaussian_dft_plotter(tlim, N):
    wlim = N * np.pi / (2 * tlim)
    t = np.linspace(-tlim, tlim, N)
    t = t[:-1]  # remove the last element
    y = np.exp(-(t**2) / 2)
    Y = np.fft.fftshift(abs(np.fft.fft(y))) / N
    Y = Y * np.sqrt(2 * np.pi) / np.max(Y)
    w = np.linspace(-wlim, wlim, N)
    w = w[:-1]  # remove the last element
    Y_actual = np.sqrt(2 * np.pi) * np.exp(-(w**2) / 2)
    print("Max Error = {}".format(np.max(abs(Y - Y_actual))))
    dft_plotter(Y, w, xlim=15, func=r"$\exp(-t^2/2)$", filename="a8_13.png")


# retaking take - 5: tlim=4pi, wlim=128, N=1024
gaussian_dft_plotter(4 * np.pi, 1024)
