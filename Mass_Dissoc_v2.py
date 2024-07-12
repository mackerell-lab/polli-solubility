# solve dM/dt, dM1/dt, dM2/dt together (simple distribution of radii)
# don't account for change dr1dt affecting dM1dt

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

def h(r):
    return r if r < hcrit else hcrit

def z_Mr(M,r):
    return 3 * D / (q * h(r) * r_0)

# Define the differential equation
def dMdt(t, M, z, M_0, Cs, V):
    return -z_Mr(M, r_0) * M_0**(1/3) * M**(2/3) * (Cs - (M_0 - M) / V) if M > 0 else 0

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
def MDsystem(t, y, M_0, Cs, V):
    M, r = y
    z = z_Mr(M, r)
    dMdt = -z * M_0**(1/3) * M**(2/3) * (Cs - (M_0 - M) / V) if M > 0 else 0
    drdt = (1/3) * (r_0**3 / M_0)**(1/3) * M**(-2/3) * dMdt if M > 0 else 0
    return [dMdt, drdt]

# Solve the differential equations
def solve_MD_system(r_0, M_0, Cs, V):
    t_span = (0, tmax)
    t_eval = np.linspace(t_span[0], t_span[1], tmax)
    solution = solve_ivp(MDsystem, t_span, [M_0, r_0], args=(M_0, Cs, V), t_eval=t_eval, method='RK45')
    M = solution.y[0]
    r = solution.y[1]
    return t_eval, M, r

# Solve the system
t_eval, Mr, r = solve_MD_system(r_0, M_0, Cs, V)
plt.plot(t_eval, (M_0-Mr)/M_0*100, label='M(t,r)')

######################################################
# Define the polydispersed system with 2 sizes, no dr1/dt, etc
def PD2system(t, y, M_0, Cs, V):
    M1, M2, = y
    M = M1 + M2
    r1 = r1_0 * (M1 / M1_0)**(1/3)
    r2 = r2_0 * (M2 / M2_0)**(1/3)
    z1 = z_Mr(M1, r1)
    z2 = z_Mr(M2, r2)
    dM1dt = -z1 * M1_0**(1/3) * M1**(2/3) * (Cs - (M_0 - M) / V) if M1 > 0 else 0
    dM2dt = -z2 * M2_0**(1/3) * M2**(2/3) * (Cs - (M_0 - M) / V) if M2 > 0 else 0
    return [dM1dt, dM2dt]

# Solve the differential equations
def solve_PD2_system(M_0, M1_0, M2_0, Cs, V):
    t_span = (0, tmax)
    t_eval = np.linspace(t_span[0], t_span[1], tmax)
    solution = solve_ivp(PD2system, t_span, [M1_0, M2_0], args=(M_0, Cs, V), t_eval=t_eval, method='RK45')
    M1 = solution.y[0]
    M2 = solution.y[1]
    M = M1 + M2
    return t_eval, M, M1, M2

# Solve the system
M1_0 = M_0 / 2
M2_0 = M_0 / 2
r1_0 = r_0*1.2
r2_0 = r_0*0.8 
t_eval, Mtot, M1, M2 = solve_PD2_system(M_0, M1_0, M2_0, Cs, V)

plt.plot(t_eval, (M_0-Mtot)/M_0*100, label='M1 + M2')

plt.plot(t_eval, (M1_0-M1)/M_0*100, '--', label='M1')
plt.plot(t_eval, (M2_0-M2)/M_0*100, '.', label='M2')

######################################################
# Define the polydispersed system with dr1/dt etc
def PD2_dr_system(t, y, M_0, Cs, V):
    M1, M2, = y
    M = M1 + M2 # also not sure if this is correct

    r1 = r1_0 * (M1 / M1_0)**(1/3)
    r2 = r2_0 * (M2 / M2_0)**(1/3) # should just be one-time variables here not dynamically updated properly

    z1 = z_Mr(M1, r1) # also need to work out deriative and have it update with the sovler as dz/dt
    z2 = z_Mr(M2, r2)
    dM1dt = -z1 * M1_0**(1/3) * M1**(2/3) * (Cs - (M_0 - M) / V) if M1 > 0 else 0
    dM2dt = -z2 * M2_0**(1/3) * M2**(2/3) * (Cs - (M_0 - M) / V) if M2 > 0 else 0

    dr1dt = (1/3) * (r1_0**3 / M1_0)**(1/3) * M1**(-2/3) * dM1dt
    dr2dt = (1/3) * (r2_0**3 / M2_0)**(1/3) * M2**(-2/3) * dM2dt

    return [dM1dt, dM2dt]

# Solve the differential equations
def solve_PD2_dr_system(M_0, M1_0, M2_0, Cs, V): # should add the r0_i's
    t_span = (0, tmax)
    t_eval = np.linspace(t_span[0], t_span[1], tmax)
    solution = solve_ivp(PD2_dr_system, t_span, [M1_0, M2_0], args=(M_0, Cs, V), t_eval=t_eval, method='RK45')
    M1 = solution.y[0]
    M2 = solution.y[1]
    M = M1 + M2
    return t_eval, M, M1, M2

# Solve the system
M1_0 = M_0 / 2
M2_0 = M_0 / 2
r1_0 = r_0*1.2
r2_0 = r_0*0.8 
t_eval, Mtot_dr, M1_dr, M2_dr = solve_PD2_dr_system(M_0, M1_0, M2_0, Cs, V) # should add the r0_i's

plt.plot(t_eval, (M_0-Mtot_dr)/M_0*100, label='M1 + M2 - dr')

plt.plot(t_eval, (M1_0-M1_dr)/M_0*100, '--', label='M1 - dr')
plt.plot(t_eval, (M2_0-M2_dr)/M_0*100, '.', label='M2 - dr')

plt.axhline(V*Cs/M_0*100, color='k',linestyle='--', label='Sat. Conc.')
plt.xlabel('Time (min)', fontweight='bold')
plt.ylabel('Mass Dissociated (%)', fontweight='bold')
plt.grid()
plt.legend()
plt.show()
