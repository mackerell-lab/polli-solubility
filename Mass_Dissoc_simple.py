# Introduce distribution, using the simplest situation (with function z(r) called within dM/dt)
# ignore 'proper' definition for dz/dt or dr/dt - can double check if this is ok for drugs we are using

import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

# Parameters
M_0 = 5.0
Cs = 0.001480764
#Cs=2
V = 100
D = 0.000378
q = 1232
#r_0 = 0.00434
r_0=0.001
hcrit = 0.0016

tmax=100

# I will assume that h=r below.
def h(r):
    return r if r < hcrit else hcrit

def z_r(r):
    return 3 * D / (q * h(r) * r_0)

# Define the differential equation
def dMdt(t, M, z, M_0, Cs, V):
    return -z_r(r_0) * M_0**(1/3) * M**(2/3) * (Cs - (M_0 - M) / V) if M > 0 else 0

# Define the function to solve the differential equation and calculate the error
def solve_differential(z_var):
    t_span = (0, tmax)
    t_eval = np.linspace(t_span[0], t_span[1], tmax)
    solution = solve_ivp(dMdt, t_span, [M_0], args=(z_var, M_0, Cs, V), t_eval=t_eval, method='RK45')
    M = solution.y[0]
    return t_eval, M 

# Solve the differential equation with z
zinit=3 * D / (q * hcrit * r_0)
t_eval, M = solve_differential(zinit)
plt.plot(t_eval, (M_0-M)/M_0*100, label='M(t,r_0)')



plt.axhline(V*Cs/M_0*100, color='k',linestyle='--', label='Sat. Conc.')
plt.xlabel('Time (min)', fontweight='bold')
plt.ylabel('Mass Dissociated (%)', fontweight='bold')
plt.grid()
plt.legend()
plt.show()
