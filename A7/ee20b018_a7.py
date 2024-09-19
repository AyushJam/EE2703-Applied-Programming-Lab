"""

Assignment 7: Applied Programming Lab EE2703
"Circuit Analysis Using Sympy"
Submitted by Ayush Mukund Jamdar EE20B018

"""

from sympy import *
import matplotlib.pyplot as plt
import numpy as np
import scipy.signal as sp

x, y, z, t = symbols('x y z t')
k, m, n = symbols('k m n', integer=True)
f, g, h = symbols('f g h', cls=Function)
init_printing()
s = symbols('s')  # This symbol will be used throughout


def lowpass(R1, R2, C1, C2, G, Vi):
    # A*V = b
    # R1 R2 C2 C2 are with reference to Figure 1 of assignment
    A = Matrix([[0, 0, 1, -1 / G], [-1 / (1 + s * R2 * C2), 1, 0, 0], [0, -G, G, 1],
                [-1 / R1 - 1 / R2 - s * C1, 1 / R2, 0, s * C1]])
    b = Matrix([0, 0, 0, -Vi / R1])
    V = A.inv() * b
    return (A, b, V)


def convert_sympy_to_lti(symbolic_exprn, s = symbols('s')):
    num, den = simplify(symbolic_exprn).as_numer_denom()
    # expressions in reduced form, then separate numerator and denominator
    p_num_den = poly(num, s), poly(den, s)  # polynomials
    c_num_den = [p.all_coeffs() for p in p_num_den]  # coefficients
    # elements of c_num_den are <class 'sympy.core.numbers.One'>, not float
    l_num, l_den = [lambdify((), c)() for c in c_num_den]  # convert to floats
    return sp.lti(l_num, l_den)


'''
For Example symbolic-exprn = s/(s^2+1)
num, den = s, s**2 + 1
p_num_den = (Poly(s, s, domain='ZZ'), Poly(s**2 + 1, s, domain='ZZ'))
c_num_den = [[1, 0], [1, 0, 1]]
lnum, lden = [1, 0] [1, 0, 1]
'''
# Question 0
# using sympy to solve the circuit problem
# input is delta(t)
A, b, V = lowpass(10000, 10000, 1e-9, 1e-9, 1.586, 1)
Vo = V[3]  # output voltage in s domain
print('Vo(s) = {}'.format(Vo))
ww = np.logspace(0, 8, 801)  # frequency range under observation
ss = 1j * ww  # s = jω
transfer_func_lowpass = lambdify(s, Vo, 'numpy')  # returns a function that operates over s variable
v = transfer_func_lowpass(ss)  # compute output at the given frequency set

plt.loglog(ww, abs(v), lw=2)
plt.title('Frequency Response of Lowpass Filter')
plt.xlabel('Frequencies on log scale')
plt.ylabel('Magnitude of H(jω) on a log scale')
plt.grid()
plt.show()

# The Assignment
'''
Question 1 : Obtain the step response
Vi(t) = u(t)
Vi(s) = 1/s
'''
# (conductance_matrix)*(Voltage_Vector) = (Current_Vector)
conductance_matrix_q1, I_q1, V_q1 = lowpass(10000, 10000, 1e-9, 1e-9, 1.586, 1/s)
print('Unit Response Vo(s) = {}'.format(V_q1[3]))
# Vo_func = lambdify(s,V_q1[3],'numpy')
t, unit_response_in_t = sp.impulse(convert_sympy_to_lti(V_q1[3]), None, np.linspace(0, 0.005, 1000))
plt.plot(t, unit_response_in_t, label='Output')
plt.plot(t,np.ones(1000),'red',label='Input')
plt.xlabel('Time')
plt.ylabel('voltage')
plt.title('Step Response of LPF (t)')
plt.grid()
plt.legend()
plt.show()

'''
Question 2: 
Input Signal => Vi = (sin(2000pi*t)+cos(2*10^6pi*t))u(t)
Vi(s) = -(2000pi)^2/(s^2+(2000pi)^2) + s/(s^2+(2*1e6pi)^2)j
Determine output voltage when passed through the 'Low Pass Filter'
'''

# t, Vo_t_q2 = sp.impulse(convert_sympy_to_lti(variable_voltages_q2[3], s), None, np.linspace(0, 0.005, 1000))
t = np.linspace(0,0.005,100000)
Vi_t_q2 = np.sin(2000*np.pi*t) + np.cos(2e6*np.pi*t)
transfer_func_lowpass = convert_sympy_to_lti(Vo)  # reference to Question 0
t,Vo_t_q2,_ = sp.lsim(transfer_func_lowpass, Vi_t_q2, t)
plt.plot(t,Vi_t_q2,'red', label='input')
plt.plot(t, Vo_t_q2, label='output')
plt.xlabel('time')
plt.ylabel('voltage')
plt.title('LPF Input - Output')
plt.legend()
plt.grid()
plt.show()


'''
Question 3: The High Pass Filter
The same Vi of Q2 is now passed through a HPF
'''

# the below function parameter names are wrt Figure Two in A7
def highpass(R1, R3, C1, C2, G, Vi):
    # A*V = b
    A = Matrix([[0, -1, 0, 1 / G], [G, -G, 0, -1], [s * C2 + 1 / R3, 0, -s * C2, 0],
                [-C2 / C1, 0, 1 + C2 / C1 + 1 / (s * C1 * R1), -1 / (s * C1 * R1)]])
    b = Matrix([0, 0, 0, Vi])
    V = A.inv() * b
    return (A, V, b)

# start by finding the transfer function, Vi(t) = delta => 1 in s domain
conductance_matrix_q3, variable_voltages_q3, b_q3 = highpass(1e4, 1e4, 1e-9, 1e-9, 1.586,1)
transfer_func_highpass = lambdify(s, variable_voltages_q3[3], 'numpy')  # output at Vi(s) = 1
plt.loglog(ww, abs(transfer_func_highpass(ss)), lw=2)
plt.xlabel('frequencies')
plt.ylabel('|H(jω)|')
plt.title('HPF |H(jω)| Plot')
plt.grid()
plt.show()

'''
Question 4: Damped Sinusoid
Vi = e^(-5)*cos(2000*np.pi*t)
'''

# first, for a low frequency signal
damping_fac = 5
t1 = np.linspace(0,1,1000)
V1_q4 = np.exp(-damping_fac*t1)*np.cos(20*np.pi*t1)
H = convert_sympy_to_lti(variable_voltages_q3[3])
t1,Vo1_q4,_ = sp.lsim(H, V1_q4,t1)

plt.plot(t1,V1_q4,label='input')
plt.plot(t1,Vo1_q4,label='output')
plt.xlabel('time')
plt.ylabel('voltage')
plt.title('Low Frequency Input')
plt.legend()
plt.grid()
plt.show()


# second, for a high frequency signal
damping_fac = 50000
t2 = np.linspace(0,0.0001,1000)
V2_q4 = np.exp(-damping_fac*t2)*np.cos(2e8*np.pi*t2)
t2,Vo2_q4,_ = sp.lsim(H, V2_q4,t2)

plt.plot(t2,V2_q4,label='input')
plt.plot(t2,Vo2_q4,label='output')
plt.xlabel('time')
plt.ylabel('voltage')
plt.title('High Frequency Input')
plt.legend()
plt.grid()
plt.show()

'''
Question 5: Unit Step Response of HPF
'''
conductance_matrix_q5, variable_voltages_q5, V_q5 = highpass(10000, 10000, 1e-9, 1e-9, 1.586, 1/s)
print('Unit Response Vo(s) = {}'.format(V_q1[3]))
# Vo_func = lambdify(s,V_q1[3],'numpy')
t, unit_response_hpf = sp.impulse(convert_sympy_to_lti(variable_voltages_q5[3]), None, np.linspace(0, 0.001, 10000))
plt.plot(t, unit_response_hpf, label='Output')
plt.plot(t,np.ones(len(t)),'red',label='Input')
plt.xlabel('Time')
plt.ylabel('voltages')
plt.title('Step Response of HPF')
plt.legend()
plt.grid()
plt.show()

################################################################################