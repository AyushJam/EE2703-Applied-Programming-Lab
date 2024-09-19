####################################################
"""
EE2703 - Applied Programming Lab 
Assignment 9: Spectra of Non-Periodic Signals
Author: Ayush Mukund Jamdar EE20B018
"""
####################################################

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm
import mpl_toolkits.mplot3d.axes3d as axes3d


# Starting with the DFT of y = sin(sqrt2*t)
t = np.linspace(-np.pi, np.pi, 65)
t = t[:-1]
dt = t[1] - t[0]
fmax = 1 / dt
y = np.sin(np.sqrt(2) * t)
y[0] = 0  # the sample corresponding to -tmax is 0
Y = np.fft.fftshift(y)  # to make y start with 0
Y = np.fft.fftshift(np.fft.fft(y)) / 64  # normalize
w = np.linspace(-np.pi * fmax, np.pi * fmax, 65)
w = w[:-1]
plt.figure()
plt.subplot(2, 1, 1)
plt.plot(w, abs(Y), lw=2)
plt.xlim([-10, 10])
plt.ylabel("|Y|", size=16)
plt.title("Spectrum of y = sin(sqrt(2)*t)")
plt.grid()

plt.subplot(2, 1, 2)
plt.plot(w, np.angle(Y), "ro", lw=2)
plt.xlim([-10, 10])
plt.ylabel("Phase of Y", size=16)
plt.xlabel(r"$\omega$", size=16)
plt.grid()
plt.savefig("A9_1.png")
plt.show()

# using the Hamming Window
t1 = np.linspace(-np.pi, np.pi, 65)
t1 = t1[:-1]
t2 = np.linspace(-3 * np.pi, -np.pi, 65)
t2 = t2[:-1]
t3 = np.linspace(np.pi, 3 * np.pi, 65)
t3 = t3[:-1]
n = np.arange(64)
wnd = np.fft.fftshift(0.54 + 0.46 * np.cos(2 * np.pi * n / 63))
y = np.sin(np.sqrt(2) * t1) * wnd
plt.figure(3)
plt.plot(t1, y, "bo", lw=2)
plt.plot(t2, y, "ro", lw=2)
plt.plot(t3, y, "ro", lw=2)
plt.ylabel(r"$y$", size=16)
plt.xlabel(r"$t$", size=16)
plt.title(r"$\sin\left(\sqrt{2}t\right)\times w(t)$ with $t$ wrapping every $2\pi$ ")
plt.grid(True)
plt.savefig("A9_2.png")
plt.show()

# DFT of y = sin(sqrt2*t) with the Hamming Window
t = np.linspace(-np.pi, np.pi, 65)
t = t[:-1]
dt = t[1] - t[0]
fmax = 1 / dt
n = np.arange(64)
wnd = np.fft.fftshift(0.54 + 0.46 * np.cos(2 * np.pi * n / 63))
# the below line was missing in the main code
y = np.sin(np.sqrt(2) * t) * wnd
y[0] = 0  # the sample corresponding to -tmax is 0
y = np.fft.fftshift(y)  # to make y start with 0
Y = np.fft.fftshift(np.fft.fft(y)) / 64  # normalize
w = np.linspace(-np.pi * fmax, np.pi * fmax, 65)
w = w[:-1]
plt.figure()
plt.subplot(2, 1, 1)
plt.plot(w, abs(Y), lw=2)
plt.xlim([-8, 8])
plt.ylabel("|Y|", size=16)
plt.title("Spectrum of y = sin(sqrt(2)*t) with Hamming Window")
plt.grid()
plt.subplot(2, 1, 2)
plt.plot(w, np.angle(Y), "ro", lw=2)
plt.xlim([-8, 8])
plt.ylabel("Phase of Y", size=16)
plt.xlabel(r"$\omega$", size=16)
plt.grid()
plt.savefig("A9_3.png")
plt.show()

# The Assignment
# Spectrum of y = cos^3(wt); w = 0.86
# first, without the Hamming Window
t = np.linspace(-np.pi, np.pi, 65)
t = t[:-1]
dt = t[1] - t[0]
fmax = 1 / dt
wo = 0.86
y = np.cos(wo * t) ** 3
y[0] = 0  # the sample corresponding to -tmax is 0
Y = np.fft.fftshift(y)  # to make y start with 0
Y = np.fft.fftshift(np.fft.fft(y)) / 64  # normalize
w = np.linspace(-np.pi * fmax, np.pi * fmax, 65)
w = w[:-1]
plt.figure()
plt.subplot(2, 1, 1)
plt.plot(w, abs(Y), lw=2)
plt.xlim([-10, 10])
plt.ylabel("Magnitude |Y|", size=16)
plt.title("Spectrum of y = cos^3(0.86t)")
plt.grid()
plt.subplot(2, 1, 2)
plt.plot(w, np.angle(Y), "ro", lw=2)
plt.xlim([-10, 10])
plt.ylabel("Phase of Y", size=16)
plt.xlabel(r"$\omega$", size=16)
plt.grid()
plt.savefig("A9_4.png")
plt.show()


# with the Hamming Window
t = np.linspace(-np.pi, np.pi, 65)
t = t[:-1]
dt = t[1] - t[0]
fmax = 1 / dt
wo = 0.86
n = np.arange(64)
wnd = np.fft.fftshift(0.54 + 0.46 * np.cos(2 * np.pi * n / 63))
y = np.cos(wo * t) ** 3 * wnd
y[0] = 0  # the sample corresponding to -tmax is 0
Y = np.fft.fftshift(y)  # to make y start with 0
Y = np.fft.fftshift(np.fft.fft(y)) / 64  # normalize
w = np.linspace(-np.pi * fmax, np.pi * fmax, 65)
w = w[:-1]
plt.figure()
plt.subplot(2, 1, 1)
plt.plot(w, abs(Y), lw=2)
plt.xlim([-10, 10])
plt.ylabel("Magnitude |Y|", size=16)
plt.title("Spectrum of y = cos^3(0.86t) with Hamming Window")
plt.grid()
plt.subplot(2, 1, 2)
plt.plot(w, np.angle(Y), "ro", lw=2)
plt.xlim([-10, 10])
plt.ylabel("Phase of Y", size=16)
plt.xlabel(r"$\omega$", size=16)
plt.grid()
plt.savefig("A9_5.png")
plt.show()


# Using more samples for the same
t = np.linspace(-4 * np.pi, 4 * np.pi, 257)
t = t[:-1]
dt = t[1] - t[0]
fmax = 1 / dt
wo = 0.86
y = np.cos(wo * t) ** 3
y[0] = 0  # the sample corresponding to -tmax is 0
Y = np.fft.fftshift(y)  # to make y start with 0
Y = np.fft.fftshift(np.fft.fft(y)) / 256  # normalize
w = np.linspace(-np.pi * fmax, np.pi * fmax, 257)
w = w[:-1]
plt.figure()
plt.subplot(2, 1, 1)
plt.plot(w, abs(Y), lw=2)
plt.xlim([-10, 10])
plt.ylabel("Magnitude |Y|", size=16)
plt.title("Spectrum of y = cos^3(0.86t)")
plt.grid()
plt.subplot(2, 1, 2)
plt.plot(w, np.angle(Y), "ro", lw=2)
plt.xlim([-10, 10])
plt.ylabel("Phase of Y", size=16)
plt.xlabel(r"$\omega$", size=16)
plt.grid()
plt.savefig("A9_4.png")
plt.show()


# with the Hamming Window
t = np.linspace(-4 * np.pi, 4 * np.pi, 257)
t = t[:-1]
dt = t[1] - t[0]
fmax = 1 / dt
wo = 0.86
n = np.arange(256)
wnd = np.fft.fftshift(0.54 + 0.46 * np.cos(2 * np.pi * n / 255))
y = np.cos(wo * t) ** 3 * wnd
y[0] = 0  # the sample corresponding to -tmax is 0
Y = np.fft.fftshift(y)  # to make y start with 0
Y = np.fft.fftshift(np.fft.fft(y)) / 256  # normalize
w = np.linspace(-np.pi * fmax, np.pi * fmax, 257)
w = w[:-1]
plt.figure()
plt.subplot(2, 1, 1)
plt.plot(w, abs(Y), lw=2)
plt.xlim([-10, 10])
plt.ylabel("Magnitude |Y|", size=16)
plt.title("Spectrum of y = cos^3(0.86t) with Hamming Window")
plt.grid()
plt.subplot(2, 1, 2)
plt.plot(w, np.angle(Y), "ro", lw=2)
plt.xlim([-10, 10])
plt.ylabel("Phase of Y", size=16)
plt.xlabel(r"$\omega$", size=16)
plt.grid()
plt.savefig("A9_5.png")
plt.show()


# Question 3
# Program to estimate frequency and phase of a signal
# cos(wo + d)
def extract_parameters(Y, w):
    # find the peak
    peak_index = np.argmax(abs(Y))
    # find the peak frequency
    peak_freq = w[peak_index]  # this is wo
    # now to find the phase
    phase = np.angle(Y[peak_index])
    return peak_freq, phase


# test the above function
x = np.linspace(0, 2 * np.pi, 129)
x = x[:-1]  # remove the last element
y = np.cos(10 * x + np.pi * 0.3)
Y = np.fft.fftshift(np.fft.fft(y)) / 128
wo, d = extract_parameters(Y, np.linspace(-64, 63, 128))
print("The frequency is:", wo, "and the phase is:", d)
# negative because it came first in the array

# Question 4
# White Gaussian Noise
# Generate white Gaussian noise
noise = 0.1 * np.random.randn(len(Y))
# add noise to the signal
Y_noise = Y + noise
wo_noise, d_noise = extract_parameters(Y_noise, np.linspace(-64, 63, 128))
print("(Noise added) The frequency is:", wo_noise, "and the phase is:", d_noise)

# Question 5
# The Chirped Signal
t = np.linspace(-np.pi, np.pi, 1025)
t = t[:-1]
dt = t[1] - t[0]
fmax = 1 / dt
y = np.cos(16 * t * (1.5 + t / (2 * np.pi)))
Y = np.fft.fftshift(np.fft.fft(y)) / 1024
w = np.linspace(-np.pi * fmax, np.pi * fmax, 1025)
w = w[:-1]
plt.figure()
plt.subplot(2, 1, 1)
plt.plot(w, abs(Y), lw=2)
plt.ylabel("Magnitude |Y|", size=16)
plt.title("Spectrum of y = cos(16t*(1.5+t/2pi))")
plt.xlim([-60, 60])
plt.grid()
plt.subplot(2, 1, 2)
plt.plot(w, np.angle(Y), "ro", lw=2)
plt.ylabel("Phase of Y", size=16)
plt.xlabel(r"$\omega$", size=16)
plt.xlim([-60, 60])
plt.grid()
plt.savefig("A9_6.png")
plt.show()


# Question 6
# Broken Chirped Signal; 16 parts
Y_array = []
for i in range(16):
    tlim1 = -np.pi + (2 * np.pi) * i / 16
    tlim2 = -np.pi + (2 * np.pi) * (i + 1) / 16
    t = np.linspace(tlim1, tlim2, 65)
    t = t[:-1]
    dt = t[1] - t[0]
    fmax = 1 / dt
    y = np.cos(16 * t * (1.5 + t / (2 * np.pi)))
    Y = np.fft.fftshift(np.fft.fft(y)) / 64
    Y_array.append((Y))

Y_array = np.array(Y_array)
t1 = np.linspace(-np.pi, np.pi, 16)
t = np.linspace(-np.pi, np.pi, 1025)
t = t[:-1]
dt = t[1] - t[0]
fmax = 1 / dt
w = np.linspace(-np.pi * fmax, np.pi * fmax, 65)
w = w[:-1]
t1, w = np.meshgrid(t1, w)
# inds = np.where(abs(w) > 150)
# Y_array[:, inds] = np.NaN
surface = axes3d.Axes3D(plt.figure())
s = surface.plot_surface(
    t1,
    w,
    abs(Y_array.T),
    rstride=1,
    cstride=1,
    cmap=cm.coolwarm,
    linewidth=0,
    antialiased=False,
)
plt.xlabel("Frequency", size=16)
plt.ylabel("Time", size=16)
plt.savefig("A9_7.png")
plt.show()
