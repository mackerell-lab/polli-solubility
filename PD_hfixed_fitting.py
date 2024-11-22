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
drug=sys.argv[1] # e.g. Ritonavir
solvent=sys.argv[2] # e.g. PBS
method=sys.argv[3] # such as RK45 or Radau

# 
filename='data/'+drug+'.csv'
paramfile='data/'+drug+'_'+solvent+'.param' # params, see below
exptfile='data/expt_'+drug+'_'+solvent+'.csv' # experimental data file, Time (min),% dissolved,SEM

# read in parameters
params = pd.read_csv(paramfile, header=None)
print(params)
# Format (example)
#                           0           1
#0         Solubility (mg/mL)       0.273
#1  Particle Density (mg/cm3)      1448.7
#2                    M0 (mg)           5
#3                     V (mL)         100
#4                D (cm2/min)     0.00048
#5                 hcrit (cm)     0.00160
#6                    r0 (cm)     0.00120
Cs = params.iloc[0,1]
q  = params.iloc[1,1]
M_0 = params.iloc[2,1]
V = params.iloc[3,1]
D = params.iloc[4,1]
hcrit = params.iloc[5,1]
r_0 = params.iloc[6,1]
tmax=300
interval=5
###################
# Read in data
r0_read, M0_read = np.loadtxt(filename, dtype=float, delimiter=',', unpack=True, skiprows=1)
r0_read = r0_read
N = len(M0_read)
M0 = np.zeros(N+1)
r0 = np.zeros(N+1)
#r0_weighted = r0_read * (M0_read/M0_read.sum())
#r_0 = r0_weighted.sum() # r_0 calculated as weighted mean of r_0 distribution

# read in time series (true curve)
exptdata = pd.read_csv(exptfile)

#def h(r):
 # return r if r < hcrit else hcrit

def z(h, r_0):
  return 3 * D / (q * h * r_0)

# Define the differential equation
def dMdt(t, M, M_0, Cs, V, h):
  #r = r_0 * (M/M_0)**(1/3)
  return -z(h, r_0)  * M_0**(1/3) * M**(2/3) * (Cs - (M_0 - M) / V) if M > 0 else 0

# Define the function to solve the differential equation and calculate the error
def solve_differential(z_var, h):
  t_span = (0, tmax)
  t_eval = np.arange(0, tmax+interval, interval)
  solution = solve_ivp(dMdt, t_span, [M_0], args=(M_0, Cs, V, h), t_eval=t_eval, method=method)
  #solution = solve_ivp(dMdt, t_span, [M_0], args=(M_0, Cs, V), t_eval=t_eval, method='RK45')
  M = solution.y[0]
  return t_eval, M 

# Solve the differential equation with z
zinit=3 * D / (q * hcrit * r_0)
#print(f"M_0: {M_0}, Cs: {Cs}, V: {V}, r_0:{r_0}, hcrit: {hcrit}, zinit: {zinit}")
t_eval, M = solve_differential(zinit, hcrit)
#plt.plot(t_eval, (M_0-M)/M_0*100, label='Simple')


# distribution of N equations
def Nsystem(t, M, M0, r0, Cs, V, h):
  n = len(M) # N + 1, for [M, M_1, M_2, ..., M_N]
  dMdt = np.zeros(n) # initialize then assign
  for i in range(1,n):
    #dMdt[i] = -z(r0[i], r0[i]) * M0[i]**(1/3) * M[i]**(2/3) * (Cs - (M0[0] - M[0]) / V) if M[i] > 0 else 0
    #r_i = r0[i] * (M[i]/M0[i])**(1/3) if M[i] > 0 else 0
    dMdt[i] = -z(h, r0[i]) * M0[i]**(1/3) * M[i]**(2/3) * (Cs - (M0[0] - M[0]) / V) if M[i] > 0 else 0
  dMdt[0] = sum(dMdt[1:])
  return dMdt

# predicted has many more time points than expt
def rmse(pred, expt):
  
  # collect subset of time points
  pd_subset = []
  for t,pd in zip(time, PD):
    if np.any(np.isclose(t, exptdata['Time (min)'], rtol=1e-4)): pd_subset.append(pd)
  pd_subset = np.array(pd_subset)
  return np.sqrt(np.mean((pd_subset-expt)**2))

# predict percent dissolved as function of const h
# return time and PD
def predict_PD(h):
  r0 = np.zeros(N+1)
  M0 = np.zeros(N+1)
  M0[0] = 100 # still in Mass Percent now
  r0[0] = r_0
  M0[1:] = M0_read
  M0 *= M_0/100 # convert mass percent to mass, % to ratio (0 to 1) and multiply by M_0
  r0[1:] = r0_read

  soln = solve_ivp(Nsystem, (0,tmax), M0, args=(M0, r0, Cs, V, h), t_eval=t_eval, method=method)# rtol=1e-4, first_step=1e-6, max_step=1e-2)
  #soln = solve_ivp(Nsystem, (0,tmax), M0, args=(M0, r0, Cs, V), t_eval=t_eval, method='RK45', rtol=1e-4)
  m = soln.y[0]
  PD = (M_0-m)/M_0 * 100
  return soln.t, PD

t_eval = np.arange(0, tmax+interval, interval)
prec=1.0
min_err = np.inf
error_tolerance = 0.5
h_opt = -1
h_array = np.arange(1, 100+prec, prec)
errorray= np.zeros(h_array.size)
for i in range(h_array.size):
  h=h_array[i]/10000 # convert to cm units
  time, PD = predict_PD(h)
  #plt.plot(time, PD, c='C1', alpha=0.2)
  err = rmse(PD, exptdata['% dissolved'])
  errorray[i] = err  
  if err + error_tolerance < min_err:
    min_err = err
    h_opt = h

# predict PD with optimal h
time, PD = predict_PD(h_opt) # must give h in cm!!

# NOTE: these are in microns, since h_array is also in microns (for convenience)
print("Tabulated h and RMSE")
print("h (micron), RMSE")
for h_interest in [7., 12., 16.]:
  print('%.0f,%.2f'%(h_interest, errorray[np.where(h_interest == h_array)][0]))
print("\n****Optimal h: %.0f"%(h_opt*10**4), "RMSE: %.2f"%min_err)

printtable = True
if printtable:
  print('==================')
  print('Printing predicted % dissolved data for time points matching these files:')
  print(filename)
  print(paramfile)
  print(exptfile)
  print('Time (min),% dissolved')
  for t,pd in zip(time, PD):
    # https://stackoverflow.com/questions/55239065/checking-if-a-specific-float-value-is-in-list-array-in-python-numpy
    if np.any(np.isclose(t, exptdata['Time (min)'], rtol=1e-4)):
      print('%i,%.3f'%(t, pd))


#################################################################

# RMSE(h_fixed) curve
plt.plot(h_array, errorray, c="black", linewidth = 2)
plt.xlabel(r'h$\bf_{fixed}$ ($\bf{\mu}$m)', fontweight='bold', fontsize=22)
plt.ylabel('RMSE (% PD)', fontweight='bold', fontsize=22)
plt.xticks(fontweight='bold', fontsize=15)
plt.yticks(fontweight='bold', fontsize=15)
ax = plt.gca()  # Get the current axis
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.tight_layout()

# Figure in desired manuscript format
plt.figure(figsize=(10, 8))
plt.rcParams['font.family'] = 'sans-serif'
plt.plot(t_eval, (M_0 - M) / M_0 * 100, c = "black", ls="--", label='Predicted (mean only)', linewidth = 2)
plt.plot(time, PD, label='Predicted (entire PSD)', c="black", linewidth = 2)
plt.errorbar(exptdata['Time (min)'], exptdata['% dissolved'], yerr=exptdata['SEM'], label='Observed', marker="o", markersize=10, capsize=2, elinewidth = 1, color='black', linewidth=2)

#plt.errorbar(exptdata2['Time (min)'], exptdata2['% dissolved'], yerr=exptdata2['SEM'], label='Observed (aggregated)', color='C3')
# Limits
plt.xlim(0, 303)
plt.ylim(0, 103)
plt.xticks(np.arange(0, 303, 60), fontweight='bold', fontsize=15)
#plt.xticks(fontweight='bold', fontsize=15)
plt.yticks(fontweight='bold', fontsize=15)

# Labels and Grid
plt.xlabel('Time (min)', fontweight='bold', fontsize=22)
plt.ylabel('% Drug dissolved', fontweight='bold', fontsize=22)
plt.legend(bbox_to_anchor=(1.7, 0.95),frameon=False, prop={ "size":22})
plt.grid(False)

# Despining
ax = plt.gca()  # Get the current axis
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# Plot Show
plt.savefig("simulation.png", dpi=450, bbox_inches='tight')
plt.savefig("simulation.svg", bbox_inches='tight')
plt.show()
