# -*- coding: utf-8 -*-
"""
Created on Thu May 12 10:28:07 2022

@author: Sasuke
"""

import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import inv

# from main import RiN, Rij


### Independent Parameters
l = 0.5
c = 2.9979e8
mu0 = 4e-7 * np.pi
N = 4
Im = 1
a = 0.01

### Dependent Parameters
Lambda = l * 4
Freq = c / Lambda
k = 2 * np.pi / Lambda
dz = l / N
print(dz)

### Helper section and variables


def get_PB(RiN, a=a, dz=dz):
    k = np.pi
    return (mu0 / (4 * np.pi) * np.exp(-complex(0, k) * RiN) * dz / RiN).reshape(-1, 1)


def get_P(R_u, dz=dz, a=a):
    k = np.pi
    return (mu0 / (4 * np.pi) * np.exp(-complex(0, k) * R_u) * dz) / (R_u)


def get_Rz(z, z_k, a=a):
    return np.sqrt(a**2 + (z_k - z) ** 2)


def get_Ru(z, z_u, a=a):
    return np.sqrt(a**2 + (z_u - z) ** 2)


def get_Q(R_u, P, dz=dz, a=a, mu0=mu0):
    k = np.pi
    x = complex(0, -k) / R_u - 1 / R_u**2
    return -a * P / mu0 * x


def get_QB(PB, RiN, mu0=mu0, a=a):
    k = np.pi
    x = complex(0, -k) / RiN - 1 / RiN**2
    return -PB * a / mu0 * x


### Question 1
z = np.linspace(-l, l, num=2 * N + 1)
# print(z)
z_k = z[[0, N, -1]]
I_approx = Im * np.sin(k * (l - z))
idx = np.ones((2 * N + 1,), dtype=bool)
idx[[0, N, -1]] = False
J = np.zeros_like(idx)
z_u = z[idx]
# print(z_u)
### Question 2
M = 1 / (2 * np.pi * a) * np.identity(2 * N - 2)

### Testing  shit
z = np.linspace(-l, l, 2 * N + 1)
u = np.concatenate((z[1:N], z[N + 1 : 2 * N]))
Ru = get_Ru(u, u.reshape(-1, 1))
Rz = get_Rz(z_u, z.reshape(-1, 1))
Pb = get_PB(Rz[N, :])
P = get_P(Ru)

Qb = get_QB(Pb, Rz[N, :].reshape(-1, 1))
Q = get_Q(Ru, P)


###

J = inv((M - Q)).dot(Qb * Im)
I_exact = np.zeros((2 * N + 1, 1), dtype=complex)
I_exact[1 : 2 * N - 1] += J
I_exact[N] = Im
plt.figure()
plt.plot(z, I_approx, label="I_formula")
plt.plot(z, I_exact, label="I_exact")
plt.title("I vs z")
plt.xlabel("z")
plt.ylabel("I")
plt.grid()
plt.legend()
# plt.show()
print(I_exact)
