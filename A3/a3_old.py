"""
           Applied Programming Lab EE2703
        ____________________________________
        Assignment 3: Fitting Data to Models
        Submitted by
          Ayush Mukund Jamdar
          EE20B018

"""

import numpy as np
import scipy.special as sp
from pylab import *

time = np.loadtxt(fname='fitting.dat', usecols=0)
# time is numpy array of 101 numbers 0.0 to 10.0

# PART 2
data = np.loadtxt(fname='fitting.dat', usecols=(range(1, 10)))
# data is a numpy array of shape 101 x 9


# PART 3
noise = []
k = []
for i in range(data.size // len(data)):  # 9 iterations
    noise.append(data.T[i] + (0.105 * time) - (1.05 * sp.jn(2, time)))
# noise shape = 9x101; where each column is a different noise
# each row is the timestamp of each noise at that time


noise = np.array(noise)
plot(time, noise.T)
xlabel(r'$t$', size=20)
ylabel(r'$n(t)$', size=20)
title(r'Figure 0: Noise')
legend(['Noise_{}'.format(i) for i in range(len(noise))])
grid()
show()

sigma = logspace(-1, -3, 9)


# PART 4
def g(t, a, b):
    return np.array(a * sp.jn(2, t) + (b * t))


g1 = g(time, 1.05, -0.105)  # g1.shape = (101, )
# print(g1.shape)
for i in range(len(noise)):
    n_t = [noise[i]]
    plot(time, (n_t + g1).T, label="σ_{} = {}".format(i, format(round(sigma[i], 3), '.3f')))

plot(time, g1, 'black', label="True Value")
legend(loc='upper right')
xlabel(r'$t$', size=20)
ylabel(r'$f(t) + noise$', size=20)
title("Q4: Data to be fitted to theory")
grid()
show()

# PART 5:
# plot the first column of data with error bars
# I use every fifth data item
plot(time, g1, 'black', label="f(t)")
errorbar(time[::5], data.T[0][::5], sigma[0], fmt='ro', label='Error bar')
title("Q5: Data points for \u03C3 = 0.10 along with exact function")
xlabel(r't$\rightarrow$')
grid()
show()

# PART 6
# Obtaining g(t, A, B) as a column vector
# Starting with constructing M
x = np.array([sp.jn(2, time[i]) for i in range(len(time))]).T
y = time.T
M = c_[x, y]
A0 = 1.05
B0 = -0.105
g2 = dot(M, np.array([A0, B0]).T)

if g2.all() == g1.all():
    print("Part 6: Vectors are equal")
else:
    print("Part 6: Vectors are unequal")

# PART 7
# Computing the mean squared error
A = arange(0, 2.1, 0.1)
B = arange(-0.2, 0.01, 0.01)


mean_squared_error = []
for i in range(len(A)):
    mean_squared_error.append(zeros(len(B)))

mean_squared_error = np.array(mean_squared_error)
# print(mean_squared_error.shape)


for i in range(len(A)):
    for j in range(len(B)):
        for k in range(101):
            mean_squared_error[i][j] += (data.T[0][k] - g(time[k], A[i], B[j])) ** 2

mean_squared_error /= 101

# PART 8
# Contour plot of mean_squared_error
cs = contour(A, B, mean_squared_error)
clabel(cs, fontsize=10)
p = scatter(A0, B0) # to plot the exact point
annotate("Exact Location", (A0, B0))
# TeX renderer in matplotlib
xlabel(r'A$\rightarrow$')
ylabel(r'B$\rightarrow$')
grid()
title("Q8: contour plot of \u03B5_ij")
show()


# PART 9 and 10
# Best estimate of A and B
'''
the lstsq fun basically solves a matrix equation ax=b 
by computing a vector x that minimizes |b-ax|**2
The equation may be under-, well-, or over-determined 
(i.e., the number of linearly independent rows of a can be less than, equal to,
 or greater than its number of linearly independent columns).
If a is square and full rank, then x (but for round-off error) is 
the “exact” solution of the equation. If not, then x gets minimized to Euclidean 2-norm 
Here, x is the column (Ai, Bj)
'''
best_estimate = np.array([np.linalg.lstsq(M, (data.T[i]).T)[0] for i in range(9)])
error_in_A = abs(A0 - best_estimate[:, 0])
error_in_B = abs(B0 - best_estimate[:, 1])

# Error vs Noise plot
plot(sigma, error_in_A, 'ro', label='Aerr', linestyle='dashed', linewidth='1')
# ro is used to plot with red circle markers
plot(sigma, error_in_B, 'go', label='Berr', linestyle='dashed', linewidth='1')
xlabel(r'Noise standard deviation $\rightarrow$')
ylabel(r'MS error$\rightarrow$')
title('Q10: Variation of error with noise')
grid()
legend()
show()

# PART 11
# plotting part 10 plot in loglog
loglog(sigma, error_in_A, 'ro', label='Aerr')
stem(sigma, error_in_A, '-ro')
stem(sigma, error_in_B, '-go')
loglog(sigma, error_in_B, 'ro', label='Aerr')
xlabel(r'Noise standard deviation $\rightarrow$')
ylabel(r'MS error$\rightarrow$')
title('Q11: Variation of error with noise')
legend()
grid()
show()

########################################################################################
