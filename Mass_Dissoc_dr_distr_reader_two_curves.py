# Distribution, with diminishing h (r_i decreases)
# Key equations:
# r_i = r0_i * (M_i/M0_i)**(1/3)
# Z(r_i, r0_i) = 3 * D / (q * h(r_i) * r0_i)
# dM_i/dt = -Z(r_i, r0_i) M0_i**(1/3) M_i**(2/3) ( Cs - (M0 - M)/V)

import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp
#from scipy.special import binom
from scipy.stats import norm, binom
import matplotlib.pyplot as plt
import sys

# read in command-line arguments, filenames and solver method
drug='Meloxicam'#sys.argv[1] # e.g. Ritonavir
solvent='POE'#sys.argv[2] # e.g. PBS
method='RK45' #sys.argv[3] # such as RK45 or Radau

# 
filename='data/'+drug+'.csv'
paramfile='data/'+drug+'_'+solvent+'.param' # params, see below
exptfile='data/expt_'+drug+'_'+solvent+'.csv' # experimental data file, Time (min),% dissolved,SEM

# read in parameters
params = pd.read_csv(paramfile, header=None)
print(params)
# Format:
#                           0           1
#0         Solubility (mg/mL)       0.273
#1  Particle Density (mg/cm3)      1448.7
#2                    M0 (mg)           5
#3                     V (mL)         100
#4                D (cm2/min)     0.00048
#5                 hcrit (cm)     0.00160
Cs = params.iloc[0,1]
q  = params.iloc[1,1]
M_0 = params.iloc[2,1]
V = params.iloc[3,1]
D = params.iloc[4,1]
hcrit = params.iloc[5,1]

tmax=300
interval=5
###################
# Read in data
r0_read, M0_read = np.loadtxt(filename, dtype=float, delimiter=',', unpack=True, skiprows=1)
r0_read = r0_read/2

N = len(M0_read)
M0 = np.zeros(N+1)
r0 = np.zeros(N+1)
r0_weighted = r0_read * (M0_read/M0_read.sum())
r_0 = r0_weighted.sum() # r_0 calculated as weighted mean of r_0 distribution

def h(r):
  return r if r < hcrit else hcrit

def z(r, r_0):
  return 3 * D / (q * h(r) * r_0)

# Define the differential equation
def dMdt(t, M, M_0, Cs, V, Csabl):
  r = r_0 * (M/M_0)**(1/3)
  return -z(r, r_0) * (Csabl/Cs) * M_0**(1/3) * M**(2/3) * (Cs - (M_0 - M) / V) if M > 0 else 0

# Define the function to solve the differential equation and calculate the error
def solve_differential(z_var, Csabl):
  t_span = (0, tmax)
  t_eval = np.arange(0, tmax, interval)
  solution = solve_ivp(dMdt, t_span, [M_0], args=(M_0, Cs, V, Csabl), t_eval=t_eval, method=method)
  #solution = solve_ivp(dMdt, t_span, [M_0], args=(M_0, Cs, V), t_eval=t_eval, method='RK45')
  M = solution.y[0]
  return t_eval, M 

# Solve the differential equation with z
# Csabl = Cs
zinit=3 * D / (q * hcrit * r_0)
print(f"M_0: {M_0}, Cs: {Cs}, V: {V}, r_0:{r_0}, hcrit: {hcrit}, zinit: {zinit}")
t_eval, M = solve_differential(zinit, Cs)
plt.plot(t_eval, (M_0-M)/M_0*100, label='Simple')

# Csabl = Cs/10
zinit=3 * D / (q * hcrit * r_0)
t_eval, M = solve_differential(zinit, Cs/10)
plt.plot(t_eval, (M_0-M)/M_0*100, label='Simple, Csabl=Cs/10', color='C3')


# distribution of N equations
def Nsystem(t, M, M0, r0, Cs, V, Csabl):
  n = len(M) # N + 1, for [M, M_1, M_2, ..., M_N]
  dMdt = np.zeros(n) # initialize then assign
  for i in range(1,n):
    #dMdt[i] = -z(r0[i], r0[i]) * M0[i]**(1/3) * M[i]**(2/3) * (Cs - (M0[0] - M[0]) / V) if M[i] > 0 else 0
    r_i = r0[i] * (M[i]/M0[i])**(1/3) if M[i] > 0 else 0
    dMdt[i] = -z(r_i, r0[i]) * (Csabl/Cs) * M0[i]**(1/3) * M[i]**(2/3) * (Cs - (M0[0] - M[0]) / V) if M[i] > 0 else 0
  dMdt[0] = sum(dMdt[1:])
  return dMdt

# first distribution
t_eval = np.arange(0, tmax, interval)
for iter in range(1):
  print(iter+1,'\r', end='')

  M0[0] = 100 # still in Mass Percent now
  r0[0] = r_0
  M0[1:] = M0_read
  M0 *= M_0/100 # convert mass percent to mass, % to ratio (0 to 1) and multiply by M_0
  r0[1:] = r0_read

  soln = solve_ivp(Nsystem, (0,tmax), M0, args=(M0, r0, Cs, V, Cs), t_eval=t_eval, method=method)# rtol=1e-4, first_step=1e-6, max_step=1e-2)
  #soln = solve_ivp(Nsystem, (0,tmax), M0, args=(M0, r0, Cs, V), t_eval=t_eval, method='RK45', rtol=1e-4)
  m = soln.y[0]
  PD = (M_0-m)/M_0 * 100
  plt.plot(soln.t, PD, label=f'Distribution', c='C1', alpha=1.0)

# second distribution
t_eval = np.arange(0, tmax, interval)
for iter in range(1):
  print(iter+1,'\r', end='')

  M0[0] = 100 # still in Mass Percent now
  r0[0] = r_0
  M0[1:] = M0_read
  M0 *= M_0/100 # convert mass percent to mass, % to ratio (0 to 1) and multiply by M_0
  r0[1:] = r0_read

  soln = solve_ivp(Nsystem, (0,tmax), M0, args=(M0, r0, Cs, V, Cs/10), t_eval=t_eval, method=method)# rtol=1e-4, first_step=1e-6, max_step=1e-2)
  #soln = solve_ivp(Nsystem, (0,tmax), M0, args=(M0, r0, Cs, V), t_eval=t_eval, method='RK45', rtol=1e-4)
  m = soln.y[0]
  PD = (M_0-m)/M_0 * 100
  plt.plot(soln.t, PD, label=f'Distribution Csabl = Cs/10', c='C4', alpha=1.0)


# read in time series (true curve)
exptdata = pd.read_csv(exptfile)

plt.errorbar(exptdata['Time (min)'], exptdata['% dissolved'], yerr=exptdata['SEM'], label='Experiment', color='C2')


printtable = True
if printtable:
  print('==================')
  print('Printing predicted % dissolved data for time points matching these files:')
  print(filename)
  print(paramfile)
  print(exptfile)
  print('Time (min),% dissolved')
  for time,pd in zip(soln.t, PD):
    # https://stackoverflow.com/questions/55239065/checking-if-a-specific-float-value-is-in-list-array-in-python-numpy
    if np.any(np.isclose(time, exptdata['Time (min)'], rtol=1e-4)):
      print('%i,%.3f'%(time, pd))

if V*Cs/M_0*100 >  100: lim=100
else: lim= V*Cs/M_0*100
plt.axhline(lim, color='k',linestyle='--', label='Sat. Conc.')
plt.xlabel('Time (min)', fontweight='bold')
plt.ylabel('Mass Dissociated (%)', fontweight='bold')
#plt.ylim([0, V*Cs/M_0*100*1.2])
plt.grid()
plt.legend(title=solvent)

plt.show()
