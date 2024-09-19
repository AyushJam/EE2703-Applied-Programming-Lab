"""
           Applied Programming Lab EE2703
        ____________________________________
        Assignment 5: The Resistor Problem
        Submitted by
          Ayush Mukund Jamdar
          EE20B018

"""

from sys import argv, exit
import mpl_toolkits.mplot3d.axes3d as p3
from matplotlib.pyplot import *
import argparse
import numpy as np


# reference for commandline argument parsing
# https://docs.python.org/3/library/argparse.html
parser = argparse.ArgumentParser(description='Processing plate data and number of iterations')
parser.add_argument('--Nx', type=int, default=25, help='size of plate along x')
parser.add_argument('--Ny', type=int, default=25, help='size of plate along y')
parser.add_argument('--radius', type=int, default=8, help='radius of central lead')
parser.add_argument('--Niter', type=int, default=1500, help='Number of iterations')
args = parser.parse_args()

Nx = args.Nx  # number of points taken along x in units
Ny = args.Ny  # number of points taken along y in units
radius = args.radius  # radius of central lead in units
Niter = args.Niter  # number of iterations to perform

# plate dimension is 1 cm by 1 cm

phi = np.zeros((Ny, Nx))
# integer points are taken because phi[i, j] represents the potential
# at the point(i,j)
x = np.linspace(-((Nx-1)//2), Nx//2, Nx)
y = np.linspace(-((Ny-1)//2), Ny//2, Ny)
X, Y = np.meshgrid(x, y)

ii = np.where(X**2 + Y**2 <= (radius**2))
# ii will have two columns,
# one giving the x-coordinate and the second the y-coordinate

phi[ii] = 1.0  # set potential of these points to 1

# making a contour plot for this
# note that origin of the coordinates is at the plate's center
# reference https://python-course.eu/numerical-programming/contour-plots-with-matplotlib.php
# reference https://matplotlib.org/2.0.2/examples/color/colormaps_reference.html
contourf(x, y, phi)  # plot x coordinates of the plate on x axis
colorbar()
scatter(X[ii], Y[ii], c='r', marker='o')
xlabel(r'x (plate dimension) $\rightarrow$')
ylabel(r'y (plate dimension) $\rightarrow$')
title("Contour plot of Potential")
show()


#  PART 2: Perform the iteration and update potential
oldphi = phi.copy()
errors = np.zeros(Niter)

for k in range(Niter):
    oldphi = phi.copy()
    '''
    # first iterating over every row 
    for i in range(1, Ny-1):  # interior nodes
        # iterating over every column
        for j in range(1, Nx-1):
            phi[i,j] = (phi[i,j-1]+phi[i,j+1]+phi[i-1,j]+phi[i+1,j])/4
    errors[k] = (abs.(phi-oldphi)).max()
    '''
    # using python subarrays
    left_neighbours = phi[1:-1, 0:-2]
    right_neighbours = phi[1:-1, 2:]
    top_neighbours = phi[0:-2, 1:-1]
    bottom_neighbours = phi[2:, 1:-1]
    phi[1:-1, 1:-1] = (left_neighbours+right_neighbours+top_neighbours+bottom_neighbours)/4

    # assert boudaries
    phi[1:-1, 0] = phi[1:-1, 1]  # left side
    phi[1:-1, -1] = phi[1:-1, -2]  # right side
    phi[0,1:-1] = phi[1, 1:-1]  # top
    phi[ii] = 1.0
    phi[-1, :] = 0  # ground at bottom edge

    errors[k] = (abs(phi - oldphi)).max()

# plotting the error
plot(range(Niter)[::50], errors[::50])
xlabel("Number of iterations")
ylabel("Error in \u03A8")
title("Change in error with iterations")
legend()
show()


# extracting the fit
# fit 1 - the entire error vector
# logy = log A + Bx; x is 1, 2, ..., Niter
# but the first argument of the lstsq should be a NiterxNiter matrix
# so I use a diagonal matrix for x
x_fit = np.zeros((Niter, 2))
x_fit[:,0] = list(range(1, Niter+1))
x_fit[:, 1] = 1

# to check if error ever goes to zero, and plot accordingly
for i in range(Niter):
    if errors[i] == 0:
        (B, A) = np.linalg.lstsq(x_fit[:i,:], np.log(errors[:i]))[0]
        A = np.exp(A)
        semilogy(range(i)[::50], A * (np.exp(B * range(i)))[::50], 'ro-',
                 label='fit1 = entire error vector when error vanishes')
        semilogy(range(i), errors[:i], 'black', label='errors')
        xlabel("Number of iterations")
        ylabel("Error in \u03A8")
        title("Fit for error vector (Semilog) when error vanishes")
        legend()
        show()
        break


(B, A) = np.linalg.lstsq(x_fit, np.log(errors))[0]
A = np.exp(A)
# A = np.exp(np.linalg.lstsq(np.diag(x_fit), np.log(errors))[1])
semilogy(range(Niter)[::50], A * (np.exp(B * range(Niter)))[::50], 'ro-', label='fit1 = entire error vector')


# fit for error entries after the 500th iteration

Bafter500, Aafter500 = np.linalg.lstsq(x_fit[500:,:], np.log(errors[500:]))[0]
Aafter500 = np.exp(Aafter500)
semilogy(range(Niter)[500::50], Aafter500*(np.exp(Bafter500*(range(Niter)[500:])))[::50], 'go-', label='fit2 = fit for entries after 500')


# plotting the real error value
semilogy(range(Niter), errors, 'black', label='errors')
xlabel("Number of iterations")
ylabel("Error in \u03A6")
title("Fit for error vector (Semilog)")
legend()
show()


# Surface plot of potential
fig1=figure(4) # open a new figure
ax=p3.Axes3D(fig1) # Axes3D is the means to do a surface plot

# print(X.shape, Y.shape, phi.T.shape)
surf = ax.plot_surface(X, Y, phi, rstride=1, cstride=1, cmap=cm.jet, label='Potential')
# Setting a stride to zero causes the data to be not sampled in the corresponding direction,
# producing a 3D line plot rather than a wireframe plot.
xlabel(r'x $\rightarrow$')
ylabel(r'y $\rightarrow$')
title('The 3-D surface plot of the potential')
show()

# contour plot of potential
contourf(X, -Y, phi)
scatter(X[ii], Y[ii], c='r', marker='o', label='Central Electrode')
colorbar()
xlabel(r'x $\rightarrow$')
ylabel(r'y $\rightarrow$')
title('Contour plot of \u03A6')
legend()
show()

# vector plot of current
Jx = np.zeros((Ny, Nx))  # since x current along parallel sides is 0
Jx[:, 1:-1] = (phi[:, :-2] - phi[:, 2:])/2

Jy = np.zeros((Ny, Nx))  # since y current along top and bottom sides is 0
Jy[1:-1, :] = (phi[:-2, :] - phi[2:, :])/2

print("Jx.shape = {}, Jy.shape = {}".format(Jx.shape, Jy.shape))
quiver(x, y, -Jx[::-1, :], -Jy[::-1, :], scale=5)
scatter(X[ii], Y[ii], c='r', marker='o', label='Central Electrode')
legend()
xlabel(r'x $\rightarrow$')
ylabel(r'y $\rightarrow$')
title("The Vector Plot of the Current Flow")
show()
