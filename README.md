# polli-solubility
Erik Nordquist, enord@outerbanks.umaryland.edu
Jim Polli, jpolli@rx.umaryland.edu

Drug solubility differential equation solver

We are using a python library called scipy to solve these systems of differential equations.

HINTZ, R., & JOHNSON, K. (1989). The effect of particle size distribution on dissolution rate and oral absorption. International Journal of Pharmaceutics, 51(1), 9–17. doi:10.1016/0378-5173(89)90069-0 


Contents:
=============

* data/ : contains the parameter files, distribution of particle radii and % mass, and k
* Mass_Dissoc_dr_distribution_reader.py : reads data (distribution, params, and experimental data), solved and plots % dissolved over time

Previous versions:
* Mass_Dissoc_distribution_reader.py : 
* Mass_Dissoc_dMdrdz.py : (monodisperse) solves dM/dt the 'simple' way with r=r0, then accounting for the fact that dr & dz are functions of dM
* Mass_Dissoc_v0.py solve dM/dt and dr/dt for homogeneous particle distribution with radius r = r0(M/M0)**1/3
*  formerly z_r_optimized.py :solve dM/dt and dr/dt for homogeneous particle distribution with radius r = r0(M/M0)**1/3
* Mass_Dissoc_v1.py : solve system of (2) particle radii: dM/dt = dM1/dt, dM2/dt
* Mass_Dissoc_v2.py : solve system of 2 with the dr/dt effect


