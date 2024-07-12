# solve dM/dt, accounting for affect of changing radii (properly)
# which required dr/dt and dz/dt. Note that, by Chain rule and when h=r:
#   dr/dt = dr/dM * dM/dt
#   dz/dt = dz/dr * dr/dM * dM/dt
# Both dr and dz change with dM. Because of that, what we did before was mathematically incorrect (if minor).
# as the particle shrinks (dr) the rate of dissolution also changes (dM). This is accounted for properly by these three diff eqs.

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


######################################################
# Define the system of differential equations
# Note that dM/dt is a function of z/r/h, which themselves are functions of M
# fundamentally, that is why we need to evaluate all three as differentials in this way
# I am assuming that h=r, since r will decrease, if r0<h then we are ok
def MDsystem(t, y):
    M, r, z = y
    dMdt = -z * M_0**(1/3) * M**(2/3) * (Cs - (M_0 - M) / V)
    drdt = (1/3) * (r_0**3 / M_0)**(1/3) * M**(-2/3) * dMdt
    dzdt = -3 * D / (q * r * r_0) * drdt
    return [dMdt, drdt, dzdt]

initial_conditions = [M_0, r_0, z_r(r_0)]
soln = solve_ivp(MDsystem, (0, tmax), initial_conditions, t_eval=t_eval)
M = soln.y[0]
r = soln.y[1]
z = soln.y[2]

plt.plot(soln.t, (M_0-M)/M_0*100, label='dM/dt properly')

plt.axhline(V*Cs/M_0*100, color='k',linestyle='--', label='Sat. Conc.')
plt.xlabel('Time (min)', fontweight='bold')
plt.ylabel('Mass Dissociated (%)', fontweight='bold')
plt.grid()
plt.legend()
plt.show()
