# Introduce distribution, using the simplest situation (with function Z(r0_i) called within dM/dt)
# a constant Z(r0)

import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp
#from scipy.special import binom
from scipy.stats import norm, binom
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

def z(r):
  return 3 * D / (q * h(r) * r_0)

# Define the differential equation
def dMdt(t, M, M_0, Cs, V):
  return -z(r_0) * M_0**(1/3) * M**(2/3) * (Cs - (M_0 - M) / V)# if M > 0 else 0

# Define the function to solve the differential equation and calculate the error
def solve_differential(z_var):
  t_span = (0, tmax)
  t_eval = np.linspace(t_span[0], t_span[1], tmax)
  solution = solve_ivp(dMdt, t_span, [M_0], args=(M_0, Cs, V), t_eval=t_eval, method='Radau')
  M = solution.y[0]
  return t_eval, M 

# Solve the differential equation with z
zinit=3 * D / (q * hcrit * r_0)
t_eval, M = solve_differential(zinit)
plt.plot(t_eval, (M_0-M)/M_0*100, label='Simple')


# Define a number of bins and calculate binomial/normal distribution of M0/r0
N = 10
tmax=100
r_0_std = r_0*0.1 # standard-deviation of r_0 distribution
r0 = norm.rvs(loc=r_0, scale=r_0_std, size=N+1)
M0 = M_0*np.ones(N+1)
weights = binom.pmf(np.arange(N), N-1, 0.5) # equivalent to: w=scipy.special.binom(N-1,np.arange(N)); w/=w.sum()
M0[1:] = M0[1:] * weights

def Nsystem(t, M, M0, r0, Cs, V):
  N = len(M)
  dMdt = np.zeros(N) # initialize then assign
  for i in range(1,N):
    dMdt[i] = -z(r0[i]) * M0[i]**(1/3) * M[i]**(2/3) * (Cs - (M0[0] - M[0]) / V)# if M[0] > 0 else 0
  dMdt[0] = sum(dMdt[1:])
  return dMdt

t_eval = np.linspace(0, tmax)
soln = solve_ivp(Nsystem, (0,tmax), M0, args=(M0, r0, Cs, V), t_eval=t_eval, method='Radau', rtol=1e-4)

M = soln.y[0]

plt.plot(soln.t, (M_0-M)/M_0 * 100, label=f'N = {N}')
plt.axhline(V*Cs/M_0*100, color='k',linestyle='--', label='Sat. Conc.')
plt.xlabel('Time (min)', fontweight='bold')
plt.ylabel('Mass Dissociated (%)', fontweight='bold')
plt.grid()
plt.legend()
plt.show()
