# Introduce distribution, using the simplest situation (with function Z(r0_i) called within dM/dt)
# dM_i/dt = -Z(r0_i) M0_i**(1/3) M_i**(2/3) ( Cs - (M0 - M)/V)

import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp
#from scipy.special import binom
from scipy.stats import norm, binom
import matplotlib.pyplot as plt

# read in parameters
paramfile = 'data/Meloxicam_PBS.param'
params = pd.read_csv(paramfile, header=None)
print(params)
# Format:
#                           0           1
#0         Solubility (mg/mL)       0.273
#1  Particle Density (mg/cm3)      1448.7
#2                    M0 (mg)           5
#3                     V (mL)         100
#4                D (cm2/min)     0.00048
Cs = params.iloc[0,1]
q  = params.iloc[1,1]
M_0 = params.iloc[2,1]
V = params.iloc[3,1]
D = params.iloc[4,1]
hcrit = params.iloc[5,1]

tmax=1200
###################
# Read in data
filename='data/Meloxicam.csv'
r0_read, M0_read = np.loadtxt(filename, dtype=float, delimiter=',', unpack=True, skiprows=1)
N = len(M0_read)
M0 = np.zeros(N+1)
r0 = np.zeros(N+1)
r_0 = r0_read.mean()


def h(r):
  return r if r < hcrit else hcrit

def z(r):
  return 3 * D / (q * h(r) * r_0)

# Define the differential equation
def dMdt(t, M, M_0, Cs, V):
  return -z(r_0) * M_0**(1/3) * M**(2/3) * (Cs - (M_0 - M) / V) if M > 0 else 0

# Define the function to solve the differential equation and calculate the error
def solve_differential(z_var):
  t_span = (0, tmax)
  t_eval = np.arange(0, tmax)
  #solution = solve_ivp(dMdt, t_span, [M_0], args=(M_0, Cs, V), t_eval=t_eval, method='Radau')
  solution = solve_ivp(dMdt, t_span, [M_0], args=(M_0, Cs, V), t_eval=t_eval, method='RK45')
  M = solution.y[0]
  return t_eval, M 

# Solve the differential equation with z
zinit=3 * D / (q * hcrit * r_0)
print(f"M_0: {M_0}, Cs: {Cs}, V: {V}, hcrit: {hcrit}, zinit: {zinit}")
t_eval, M = solve_differential(zinit)
plt.plot(t_eval, (M_0-M)/M_0*100, label='Simple')


# distribution of N equations
def Nsystem(t, M, M0, r0, Cs, V):
  n = len(M) # N + 1, for [M, M_1, M_2, ..., M_N]
  dMdt = np.zeros(n) # initialize then assign
  for i in range(1,n):
    dMdt[i] = -z(r0[i]) * M0[i]**(1/3) * M[i]**(2/3) * (Cs - (M0[0] - M[0]) / V) if M[i] > 0 else 0
  dMdt[0] = sum(dMdt[1:])
  return dMdt

t_eval = np.arange(0, tmax)
for iter in range(10):
  print(iter+1,'\r', end='')

  M0[0] = M0_read.sum()
  r0[0] = r0_read.mean()
  M0[1:] = M0_read
  M0 *= M_0/100 # convert mass percent to mass, % to ratio (0 to 1) and multiply by M_0
  r0[1:] = r0_read

  #soln = solve_ivp(Nsystem, (0,tmax), M0, args=(M0, r0, Cs, V), t_eval=t_eval, method='Radau', rtol=1e-4, first_step=1e-6, max_step=1e-2)
  soln = solve_ivp(Nsystem, (0,tmax), M0, args=(M0, r0, Cs, V), t_eval=t_eval, method='RK45', rtol=1e-4)
  m = soln.y[0]
  plt.plot(soln.t, (M_0-m)/M_0 * 100, label=f'N = {N}', c='C1', alpha=0.6)

if V*Cs/M_0*100 >  100: lim=100
else: lim= V*Cs/M_0*100
plt.axhline(lim, color='k',linestyle='--', label='Sat. Conc.')
plt.xlabel('Time (min)', fontweight='bold')
plt.ylabel('Mass Dissociated (%)', fontweight='bold')
#plt.ylim([0, V*Cs/M_0*100*1.2])
plt.grid()
#plt.legend()
plt.show()
