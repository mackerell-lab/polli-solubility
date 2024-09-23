# Introduce distribution, using the simplest situation (with function Z(r0) called within dM/dt)
# a constant Z(r0)
# note these variables are not exactly consistent with the next script

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

def Z(r):
  return 3 * D / (q * h(r) * r_0)

# Define the differential equation
def dMdt(t, M, r_0, Cs, V):
  return -Z(r_0) * M_0**(1/3) * M**(2/3) * (Cs - (M_0 - M) / V)# if M > 0 else 0

# Solve the differential equation with z
t_eval = np.linspace(0, tmax)
solution = solve_ivp(dMdt, (0, tmax), [M_0], args=(r_0, Cs, V), t_eval=t_eval, method='Radau')
M = solution.y[0]
plt.plot(t_eval, (M_0-M)/M_0*100, label='M(t,r_0)')

def system(t, y, r1_0, r2_0, Cs, V):
  M, M1, M2 = y
  dM1dt = -Z(r1_0) * M1_0**(1/3) * M1**(2/3) * (Cs - (M_0 - M) / V)# if M > 0 else 0
  dM2dt = -Z(r2_0) * M2_0**(1/3) * M2**(2/3) * (Cs - (M_0 - M) / V)# if M > 0 else 0
  dMdt  = dM1dt + dM2dt
  return [dMdt, dM1dt, dM2dt]

M1_0 =  M_0/2
M2_0 =  M_0/2
r1_0 =  r_0*1.4
r2_0 =  r_0*0.6
initial_conditions = [M_0, M1_0, M2_0]
soln = solve_ivp(system, (0, tmax), initial_conditions, args=(r1_0, r2_0, Cs, V), t_eval=t_eval, method='Radau')
M  = soln.y[0]
M1 = soln.y[1]
M2 = soln.y[2]

plt.plot(soln.t, (M_0-M)/M_0*100, label='Simple with 2 slightly-different sizes')

# solve with N-system
def Nsystem(t, M, r0, M0, Cs, V):
  N = len(M)
  dMdt = np.zeros(N) # initialize then assign
  for i in range(1,N):
    dMdt[i] = -Z(r0[i]) * M0[i]**(1/3) * M[i]**(2/3) * (Cs - (M0[0] - M[0]) / V)# if M[0] > 0 else 0
  dMdt[0] = sum(dMdt[1:])
  return dMdt

r0_array = [r_0, r1_0, r2_0]
soln = solve_ivp(Nsystem, (0,tmax), initial_conditions, args=(r0_array, initial_conditions, Cs, V), t_eval=t_eval, method='Radau', rtol=1e-4)
M = soln.y[0]

plt.plot(soln.t, (M_0-M)/M_0 * 100, label=f'N = {len(initial_conditions)-1}')


plt.axhline(V*Cs/M_0*100, color='k',linestyle='--', label='Sat. Conc.')
plt.xlabel('Time (min)', fontweight='bold')
plt.ylabel('Mass Dissociated (%)', fontweight='bold')
plt.grid()
plt.legend()
plt.show()
