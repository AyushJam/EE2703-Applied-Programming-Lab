"""
           Applied Programming Lab EE2703
        ____________________________________
        Assignment 6: The Laplace Transform
        Submitted by
          Ayush Mukund Jamdar
          EE20B018

"""

from sys import argv, exit
import matplotlib.pyplot as plt
import numpy as np
import scipy.signal as sp

# Question 1
# define the F(s) function (Laplace domain)
F1_s = sp.lti([1, 0.5], [1, 1, 2.5])
X1_s = sp.lti([1, 0.5], np.polymul([1, 0, 2.25], [1, 1, 2.5]))
t, x1 = sp.impulse(X1_s, None, np.linspace(0, 50, 1000))
# this gives the inverse laplace transform
plt.plot(t, x1)
plt.title("Time Response")
plt.xlabel(r'time $\rightarrow$')
plt.ylabel(r'x(t) $\rightarrow$')
plt.show()

# Question 2
# doing question 1 with smaller decay
F2_s = sp.lti([1, 0.05], [1, 0.1, 2.2525])
X2_s = sp.lti([1, 0.05], np.polymul([1, 0, 2.25], [1, 0.1, 2.2525]))
t, x2 = sp.impulse(X2_s, None, np.linspace(0, 50, 1000))
# this gives the inverse laplace transform
plt.plot(t, x2)
plt.title("Time Response For Smaller Decay")
plt.xlabel(r'time $\rightarrow$')
plt.ylabel(r'x(t) $\rightarrow$')
plt.show()

# Question 3
freq = np.linspace(1.4,1.6,5)
for f in freq:
    F_s = sp.lti([1, 0.5], [1, 0.1, (f**2)+0.0025])
    X_s = sp.lti([1, 0.5], np.polymul([1, 0, 2.25], [1, 0.1, (f**2)+0.0025]))
    H_s = sp.lti([1],[1,0,2.25])  # Obtain the transfer function X(s)/F(s)
    time_interval = np.linspace(0,50,1000)
    f_t = np.cos(f*time_interval)*(np.exp(-0.05*time_interval))
    t,x_t,svec = sp.lsim(H_s,f_t,time_interval)
    plt.subplot(1,2,1)
    plt.plot(time_interval, x_t,label='Output function')
    plt.xlabel(r'time $\rightarrow$')
    plt.ylabel(r'x(t) $\rightarrow$')
    plt.title('x(t) output')
    plt.subplot(1,2,2)
    plt.plot(time_interval,f_t,'red',label='Input function')
    plt.legend()
    plt.ylabel('f(t)')
    plt.title("f(t) at frequency {}".format(round(f, 2)))
    plt.show()

# Question 4
# Coupled Spring
X_s_q4 = sp.lti([1,0,2],[1,0,3,0])
t_q4, x_q4 = sp.impulse(X_s_q4,None,np.linspace(0, 20, 100))
plt.plot(t_q4, x_q4,'red')
plt.xlabel('time')
plt.ylabel('x(t)')
plt.title('x(t) - Coupled Spring Problem')
plt.show()

Y_s_q4 = sp.lti([2],[1,0,3,0])
t_q4, y_q4 = sp.impulse(Y_s_q4,None,np.linspace(0, 20, 100))
plt.plot(t_q4, y_q4,'red')
plt.xlabel('time')
plt.ylabel('y(t)')
plt.title('y(t) - Coupled Spring Problem')
plt.show()

# Question 5
trans_func = sp.lti([1],[1e-12,1e-4,1])
w,S,phi = trans_func.bode()
plt.subplot(2,1,1)
plt.semilogx(w,S,'red')
plt.title('Magnitude Plot')
plt.xlabel('w (log scale)')
plt.ylabel(r'|H(jw)|_{dB}')
plt.subplot(2,1,2)
plt.semilogx(w,phi)
plt.title('Phase Plot')
plt.xlabel('w (log scale)')
plt.ylabel(r'arg(H(jw))')
plt.show()

# Question 6
t_q6 = np.linspace(0,30e-6,1000)
vin = np.cos(t_q6*1000)-np.cos(t_q6*(10**6))
t_q6,vout,svec = sp.lsim(trans_func,vin,np.linspace(0,30e-6,1000))
plt.plot(t_q6,vin,'red',label='Vin')
plt.plot(t_q6,vout,'green',label='Vout')
plt.xlabel('time')
plt.ylabel('voltage')
plt.title('Question 6')
plt.legend()
plt.show()

t_q6 = np.linspace(0,0.01,1000)
vin = np.cos(t_q6*1000)-np.cos(t_q6*(10**6))
t_q6,vout,svec = sp.lsim(trans_func,vin,np.linspace(0,0.01,1000))
plt.plot(t_q6,vin,'red',label='Vin')
plt.plot(t_q6,vout,'green',label='Vout')
plt.xlabel('time')
plt.ylabel('voltage')
plt.title('Question 6')
plt.legend()
plt.show()

###############################################################################