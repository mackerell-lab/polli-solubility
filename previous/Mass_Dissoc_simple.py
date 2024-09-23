# Introduce distribution, using the simplest situation (with function z(r) called within dM/dt)
# ignore 'proper' definition for dz/dt or dr/dt - can double check if this is ok for drugs we are using

import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

# Parameters
#0         Solubility (mg/mL)     0.27300
#1  Particle Density (mg/cm3)  1448.70000
#2                    M0 (mg)     5.00000
#3                     V (mL)   100.00000
#4                D (cm2/min)     0.00048
#5                 hcrit (cm)     0.00160
#M_0: 5.0, Cs: 0.273, V: 100.0, hcrit: 0.0016, zinit: 13654.40449788896

M_0 = 5.0
Cs = 0.27300
#Cs=2
V = 100
D = 0.00048
q = 1448.7
#r_0 = 0.00434
r_0=0.001
hcrit = 0.0016

tmax=20

# I will assume that h=r below.
def h(r):
    return r if r < hcrit else hcrit

def z_r(r, r_0):
    return 3 * D / (q * h(r) * r_0)

# Define the differential equation
def dMdt(t, M, z, M_0, Cs, V):
    return -z_r(r_0, r_0) * M_0**(1/3) * M**(2/3) * (Cs - (M_0 - M) / V) if M > 0 else 0

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



#plt.axhline(V*Cs/M_0*100, color='k',linestyle='--', label='Sat. Conc.')
plt.xlabel('Time (min)', fontweight='bold')
plt.ylabel('Mass Dissociated (%)', fontweight='bold')
plt.grid()
plt.legend()
plt.show()
