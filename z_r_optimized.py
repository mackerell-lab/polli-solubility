# solve dM/dt and dr/dt together

import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp, odeint
import matplotlib.pyplot as plt

# Parameters
M0 = 5.0
#Cs = 0.001480764
Cs=0.05
V = 100
D = 0.000378
q = 1232
#r0 = 0.00434
r0=0.0001
hcrit = 0.0016

def h(r):
    return r if r < hcrit else hcrit

def zM(M):
    r = r0 * (M / M0)**(1/3)
    return 3 * D / (q * h(r) * r0)

# Define the differential equation
def dMdt(t, M, z, M0, Cs, V):
    return -z * M0**(1/3) * M**(2/3) * (Cs - (M0 - M) / V)

# Define the function to solve the differential equation and calculate the error
def solve_differential(z):
    t_span = (0, 300)
    t_eval = np.linspace(t_span[0], t_span[1], 300)
    solution = solve_ivp(dMdt, t_span, [M0], args=(z, M0, Cs, V), t_eval=t_eval, method='RK45')
    M = solution.y[0]
    return t_eval, M 

# Solve the differential equation with z
z=3 * D / (q * hcrit * r0)
t_eval, M = solve_differential(z)

# Define the system of differential equations
def system(t, y, M0, Cs, V):
    M, r = y
    z = zM(M)
    dMdt = -z * M0**(1/3) * M**(2/3) * (Cs - (M0 - M) / V)
    drdt = (1/3) * (r0**3 / M0)**(1/3) * M**(-2/3) * dMdt
    return [dMdt, drdt]

# Solve the differential equations
def solve_differential_system(r0, M0, Cs, V):
    t_span = (0, 300)
    t_eval = np.linspace(t_span[0], t_span[1], 300)
    solution = solve_ivp(system, t_span, [M0, r0], args=(M0, Cs, V), t_eval=t_eval, method='RK45')
    M = solution.y[0]
    r = solution.y[1]
    return t_eval, M, r

# Solve the system
t_eval, Mr, r = solve_differential_system(r0, M0, Cs, V)

# Plot the results
plt.figure()
plt.plot(t_eval, M, label='M(t,r0)')
plt.plot(t_eval, Mr, label='M(t,r)')
#plt.plot(t_eval, M-Mr, label='M(t,r0)-M(t,r)')
#plt.plot(t_eval, r, label='r(t)')
plt.xlabel('Time')
plt.ylabel('Values')
plt.grid()
plt.legend()
plt.show()
