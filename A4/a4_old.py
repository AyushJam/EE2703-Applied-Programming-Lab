"""
           Applied Programming Lab EE2703
        ____________________________________
        Assignment 4: Fourier Series
        Submitted by
          Ayush Mukund Jamdar
          EE20B018

"""

import numpy as np
import scipy.integrate
from matplotlib.pyplot import *


def exponential(x):
    return np.exp(x)


def coscos(x):
    return np.cos(np.cos(x))


# PART 1: Plotting the two functions
# The exponential function in the given range -2pi to 4pi
# playing around with the number of elements in x helps realise
# how much the accuracy of plots changes

x = np.linspace(-2 * np.pi, 4 * np.pi, 300)
y1 = exponential(x)
# periodic extension of y1 over 0 to 2pi means its values over this range
# should repeat
y1_periodic = y1[100: 200]
y1_periodic = np.append(y1_periodic, y1_periodic)
y1_periodic = np.append(y1_periodic, y1[100: 200])  # because I am plotting over three periods
semilogy(x, y1, 'red', label='Main function')
semilogy(x, y1_periodic, label='Periodic over 0 to 2\u03C0')
grid()
xlabel("x")
ylabel("e^x")
legend()
title("Q1: Exponential(x)")
show()

# The cos(cos(x)) function in the given range -2pi to 4pi
x = np.linspace(-2 * np.pi, 4 * np.pi, 300)
y2 = coscos(x)
y2_periodic = y2[100:200]
y2_periodic = np.append(y2_periodic, y2_periodic)
y2_periodic = np.append(y2_periodic, y2[100: 200])
plot(x, y2, label='Main function')
plot(x, y2_periodic, 'red', label='Periodic Extension')
grid()
legend()
xlabel("x")
ylabel("cos(cos(x))")
title("Q1: cos(cos(x))")
show()

# PART 2: Fourier Series Coefficients
# First, the exponential function
f0 = lambda a: np.exp(a)
a0_exp = (scipy.integrate.quad(f0, 0, 2 * np.pi)[0]) / (2 * np.pi)
# The quad function returns the two values, in which
# the first number is the value of integral and the second value
# is the estimate of the absolute error in the value of integral.

# 'a' coefficients for this series
a_n_exp = []
for i in range(1, 26):
    f = lambda x, k: np.exp(x) * np.cos(x * k)
    a_n_exp.append((scipy.integrate.quad(f, 0, 2 * np.pi, args=(i))[0]) / np.pi)

b_n_exp = []
for i in range(1, 26):
    g = lambda x, k: np.exp(x) * np.sin(x * k)
    b_n_exp.append((scipy.integrate.quad(g, 0, 2 * np.pi, args=(i))[0]) / np.pi)

# Plot a_n
scatter(np.array(range(25)), abs(np.array(a_n_exp)))
xlabel('n')
ylabel('a_n')
title('a_n for exp(x)')
show()

# Plot b_n
scatter(np.array(range(25)), abs(np.array(b_n_exp)))
xlabel('n')
ylabel('b_n')
title('b_n for exp(x)')
show()

# Second, the coscos function
p0 = lambda x: np.cos(np.cos(x))
a0_cos = (scipy.integrate.quad(p0, 0, 2 * np.pi)[0]) / (2 * np.pi)

a_n_cos = []
for i in range(1, 26):
    p = lambda x, k: np.cos(np.cos(x)) * np.cos(x * k)
    a_n_cos.append((scipy.integrate.quad(p, 0, 2 * np.pi, args=(i))[0]) / np.pi)

b_n_cos = []
for i in range(1, 26):
    q = lambda x, k: np.cos(np.cos(x)) * np.sin(x * k)
    b_n_cos.append((scipy.integrate.quad(q, 0, 2 * np.pi, args=(i))[0]) / np.pi)

# Plot a_n
scatter(np.array(range(25)), abs(np.array(a_n_cos)))
xlabel('n')
ylabel('a_n')
title('a_n for cos(cos(x))')
show()

# Plot b_n
scatter(np.array(range(25)), abs(np.array(b_n_cos)))
xlabel('n')
ylabel('b_n')
title('b_n for cos(cos(x))')
show()

# PART3: Plotting coefficients
# First the exponential function
coeffs_of_exp = [abs(a0_exp)]
for i in range(25):
    coeffs_of_exp.append(abs(a_n_exp[i]))
    coeffs_of_exp.append(abs(b_n_exp[i]))

# Plot-1
semilogy(np.array(range(51)), coeffs_of_exp, 'ro')
xlabel(r'n$\rightarrow$')
ylabel(r'Magnitude of coefficients$\rightarrow$')
title('Figure 3: Coeffs of exp(x) in semilog')
grid()
show()

# Plot-2
loglog(np.array(range(51)), coeffs_of_exp, 'ro')
xlabel(r'n$\rightarrow$')
ylabel(r'Magnitude of coefficients$\rightarrow$')
title('Figure 4: Coeffs of exp(x) in loglog')
grid()
show()

# Second the cos(cos(x)) function
# Important: Note the order of coefficients
# in the plot, the order is a0, a1, b1,..., a25, b25.
coeffs_of_coscos = [abs(a0_cos)]
for i in range(25):
    coeffs_of_coscos.append(abs(a_n_cos[i]))
    coeffs_of_coscos.append(abs(b_n_cos[i]))

# Plot-1
semilogy(np.array(range(51)), coeffs_of_coscos, 'ro')
xlabel(r'n$\rightarrow$')
ylabel(r'Magnitude of coefficients$\rightarrow$')
title('Figure 5: Coeffs of cos(cos(x)) in Semilog')
grid()
show()

# Plot-2
loglog(np.array(range(51)), coeffs_of_coscos, 'ro')
xlabel(r'n$\rightarrow$')
ylabel(r'Magnitude of coefficients$\rightarrow$')
title('Figure 6: Coeffs of cos(cos(x)) in Loglog')
grid()
show()


# PART 4 and 5: Least Squares Approach
X = np.linspace(0, 2*np.pi, 400)

# First, the exponential function
b_matrix_for_exp = exponential(X)
# build matrix A
"""
# This is the method I came up with, but the assign says not to use nested loop
A = []
for i in range(400):
    row = []
    row.append(1)
    for j in range(25):
        row.append(np.cos((j+1)*X[i]))
        row.append(np.sin((j+1)*X[i]))

    A.append(row)

A = np.array(A)  # this gives me the matrix
"""
# This is the way given in the assignment
X = np.linspace(0,2*np.pi,401)
X = X[:-1] # drop last term to have a proper periodic integral
b_matrix_for_exp = exponential(X)  # f has been written to take a vector
A_mat_for_exp = np.zeros((400, 51))  # allocate space for A
A_mat_for_exp[:, 0] = 1  # col 1 is all ones
for k in range(1,26):
    A_mat_for_exp[:, 2 * k - 1] = np.cos(k * X)  # cos(kx) column
    A_mat_for_exp[:, 2 * k] = np.sin(k * X)  # sin(kx) column

coeff_for_exp_lstsq = np.linalg.lstsq(A_mat_for_exp, b_matrix_for_exp)[0]  # the ’[0]’ is to pull out the
# best fit vector. lstsq returns a list.

# now plotting it
semilogy(np.array(range(51)), (coeffs_of_exp), 'ro', alpha=0.4,  label='Original Value')
semilogy(np.array(range(51)), abs(coeff_for_exp_lstsq), 'go', alpha=0.4, label='Least Squares Approach')
xlabel(r'n$\rightarrow$')
ylabel(r'Magnitude of coefficients$\rightarrow$')
title('Figure 7: Coeffs of exp(x) lstsq approach')
legend()
grid()
show()

loglog(np.array(range(51)), coeffs_of_exp, 'ro', alpha=0.4, label='Original Value')
loglog(np.array(range(51)), abs(coeff_for_exp_lstsq), 'go', alpha=0.4, label='Least Squares Approach')
xlabel(r'n$\rightarrow$')
ylabel(r'Magnitude of coefficients&\rightarrow&')
title('Figure 4: Coeffs of exp(x) in loglog')
legend()
grid()
show()


# Second, the cos(cosx) function

# build matrix A
# We'll use X from the previous part
b_matrix_for_coscos = coscos(X)  # f has been written to take a vector
A_mat_for_coscos = np.zeros((400, 51))  # allocate space for A
A_mat_for_coscos[:, 0] = 1  # col 1 is all ones
for k in range(1,26):
    A_mat_for_coscos[:, 2 * k - 1] = np.cos(k * X)  # cos(kx) column
    A_mat_for_coscos[:, 2 * k] = np.sin(k * X)  # sin(kx) column

coeff_for_coscos_lstsq = np.linalg.lstsq(A_mat_for_coscos, b_matrix_for_coscos)[0]
# the ’[0]’ is to pull out the
# best fit vector. lstsq returns a list.

# now plotting it
semilogy(np.array(range(51)), (coeffs_of_coscos), 'ro', alpha=0.4, label='Original Value')
semilogy(np.array(range(51)), abs(coeff_for_coscos_lstsq), 'go', alpha=0.4, label='Least Squares Approach')
xlabel(r'n$\rightarrow$')
ylabel(r'Magnitude of coefficients$\rightarrow$')
title('Figure 8: Coeffs of cos(cos(x)) lstsq approach')
legend()
grid()
show()

loglog(np.array(range(51)), coeffs_of_coscos, 'ro', label='Original Value', alpha=0.4)
loglog(np.array(range(51)), abs(coeff_for_coscos_lstsq), 'go', label='Least Squares Approach', alpha=0.4)
xlabel(r'n$\rightarrow$')
ylabel(r'Magnitude of coefficients&\rightarrow&')
title('Figure 6: Coeffs of cos(cos(x)) in Loglog')
legend()
grid()
show()

# PART 6: Deviation
# First the exponential
error_exp = coeffs_of_exp - abs(coeff_for_exp_lstsq)
plot(range(51), abs(error_exp), 'ro')
xlabel('n')
ylabel('real coeffs - lstsq coeffs')
title('Deviation in coeffs of exp(x)')
grid()
show()
print(np.amax(abs(error_exp)))

# Second the coscos
error_coscos = coeffs_of_coscos - abs(coeff_for_coscos_lstsq)
plot(range(51), abs(error_coscos), 'ro')
xlabel('n')
ylabel('real coeffs - lstsq coeffs')
grid()
title('Deviation in coeffs of exp(x)')
show()
print(np.amax(abs(error_coscos)))


# PART 7: Compute Ac
# Exp function
Ac_exp = np.dot(A_mat_for_exp, coeff_for_exp_lstsq)

semilogy(X, Ac_exp, 'go', label='Obtained from lstsq and Fourier', alpha=0.4)
semilogy(X, exponential(X), label='True Value')
xlabel('X')
ylabel('Function values at Xi')
legend()
show()

# CosCos function
Ac_coscos = np.dot(A_mat_for_coscos, coeff_for_coscos_lstsq)

plot(X, Ac_coscos, 'go', label='Obtained from lstsq and Fourier', alpha=0.4)
plot(X, coscos(X), label='True Value')
xlabel('X')
ylabel('Function values at Xi')
legend()
show()

########################################################################################
