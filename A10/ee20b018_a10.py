"""
EE2703: Applied Programming Lab
Assignment 10: Convolution
Author: Ayush Mukund Jamdar EE20B018

"""
########################################################
# Imports
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as sig
import csv

# The filter response h.csv
with open("h.csv") as f:
    h = np.loadtxt(f, delimiter=",")

# plotting the filter response
# USE: scipy.signal.freqz()
w, H = sig.freqz(h)  # takes the fourrier transform of the filter
plt.plot(w, 20 * np.log10(abs(H)))
plt.xlabel("Frequency (rad/sample)")
plt.ylabel("Magnitude (dB)")
plt.title("Magnitude Response")
plt.grid()
plt.savefig("h.png")
plt.show()

plt.plot(w, np.unwrap(np.angle(H)))
# unwrap ensures that concecutive phase angles don't differ by more than pi
# it takes 2 pi complement if diff > pi
plt.xlabel("Frequency (rad/sample)")
plt.ylabel("Phase (radians)")
plt.title("Phase Response")
plt.grid()
plt.savefig("h_phase.png")
plt.show()

"""
Generate and plot the input signal x[n]
x[n] = cos(0.2*pi*n) + cos(0.85*pi*n)
n = 1, 2, 3, ..., 2**10 
"""
n = np.arange(1, 2**10 + 1)
x = np.cos(0.2 * np.pi * n) + np.cos(0.85 * np.pi * n)
plt.plot(n, x)
plt.xlabel("n")
plt.ylabel("x[n]")
plt.title("Input Signal")
plt.savefig("x.png")
plt.grid()
plt.show()

# Now, pass the sequence x through the filter h
# using linear convolution
y = np.zeros(len(x))
for i in range(len(x)):
    for j in range(len(h)):
        if i + j < len(x):
            y[i] += x[i + j] * h[j]

plt.plot(n, y)
plt.xlabel("n")
plt.ylabel("y[n]")
plt.title("Output Signal")
plt.savefig("y.png")
plt.grid()
plt.show()


# Now, repeat the above process using circular convolution
# This way, we take fourier transforms, multilpy and then take inverse fourier transform
Y = np.fft.fft(x) * np.fft.fft(np.concatenate((h, np.zeros(len(x) - len(h)))))
y_circ = np.fft.ifft(Y)
plt.plot(n, y_circ)
plt.xlabel("n")
plt.ylabel("y[n]")
plt.title("Output Signal - Circular Convolution")
plt.savefig("y_circ.png")
plt.grid()
plt.show()


# Now, doing the linear convolution using circular convolution
P = len(h)
print("P = ", P)
n1 = int(np.floor(np.log2(P))) + 1  # process to pad zeros to h
print("n1 = ", n1)
h_with_zeros = np.concatenate((h, np.zeros(int((2**n1)) - P)))
len_h = len(h_with_zeros)  # = P
print("len_h = ", len_h)
L = int(np.floor(len(x) / 2**n1))  # length of each section of x
print("L = ", L)
x_with_zeros = np.concatenate((x, np.zeros(L * (int(2**n1)) - len(x))))
y_circ_conv = np.zeros(
    len(x_with_zeros) + len(h_with_zeros) - 1
)  # using property of convolution, we know the length
print("len(x_w_zeros) = ", len(x_with_zeros))

for i in range(L):
    x_ = np.concatenate(
        (x_with_zeros[i * len_h : (i + 1) * len_h], np.zeros(len_h - 1))
    )
    y_circ_conv[i * len_h : (i + 1) * len_h + len_h - 1] += np.fft.ifft(
        np.fft.fft(x_)
        * np.fft.fft(
            np.concatenate((h_with_zeros, np.zeros(len(x_) - len(h_with_zeros))))
        )
    ).real

"""
if .real is not used, the following error is thrown:
    numpy.core._exceptions.UFuncTypeError: Cannot cast ufunc 'add' output 
    from dtype('complex128') to dtype('float64') with casting rule 'same_kind' 
"""
plt.plot(n, (y_circ_conv[:1024]).real)  # as n goes from 1 to 2**10
plt.xlabel("n")
plt.ylabel("y[n]")
plt.title("Linear Convolution from Circular Convolution")
plt.savefig("lin_from_cir.png")
plt.grid()
plt.show()


# Circular correlation
# Zadoff Chu sequences, first reading the numebers as floats from strings
with open("x1.csv") as f:
    csvreader = csv.reader(f)
    lines = np.array([row for row in csvreader])

# print(lines, type(lines), lines.shape)
# this tells that the array is a numpy array and numbers are given as a+ib
# we need to convert them to complex numbers
zandoff_chu_seq = []
for line in lines:
    try:
        a, bi = line[0].split("+")
        a = float(a)
        bi = float(bi[:-1])  # leaving the i
    except ValueError:
        try:
            if line[0][0] == "-":
                _, a, bi = line[0].split("-")
            else:
                a, bi = line[0].split("-")
            a = float(a)
            bi = -float(bi[:-1])  # leaving the i
        except ValueError:
            a = float(line[0])  # there is a 1 in the list
            bi = 0.0

    zandoff_chu_seq.append([a + 1j * bi])

zandoff_chu_seq = np.array(zandoff_chu_seq)
# print(zandoff_chu_seq, len(zandoff_chu_seq), zandoff_chu_seq.shape)
# correlation of this sequence with a shifted version of itself
shifted_zandoff_chu_seq = np.roll(zandoff_chu_seq, 5)
correlation = np.fft.ifftshift(
    np.correlate(shifted_zandoff_chu_seq[:, 0], zandoff_chu_seq[:, 0], "full")
)
print(len(correlation))
plt.plot(np.arange(0, len(correlation)), abs(correlation), "ro")
plt.xlabel("n")
plt.ylabel("correlation")
plt.xlim(0, 30)
plt.title("Correlation of Zadoff-Chu Sequence")
plt.savefig("correlation.png")
plt.grid()
plt.show()

###############################################################################
