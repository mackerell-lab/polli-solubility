# polli-solubility
Drug solubility differential equation solvers

We are using a python library called scipy to solve these systems of differential equations.

Erik Nordquist, enord@outerbanks.umaryland.edu
Prabin Baral, pbaral@outerbanks.umaryland.edu

"The effectof particle size distribution on dissolution rate and oral absorption" Hintz and Johnson 1989 IJP elsevier

Contents:
=============

* Mass_Dissoc_v0.py solve dM/dt and dr/dt for homogeneous particle distribution with radius r = r0(M/M0)**1/3
*  formerly z_r_optimized.py :solve dM/dt and dr/dt for homogeneous particle distribution with radius r = r0(M/M0)**1/3
* Mass_Dissoc_v1.py : solve system of (2) particle radii: dM/dt = dM1/dt, dM2/dt
* Mass_Dissoc_v2.py : solve system of 2 with the dr/dt effect
* TODO: think properly about dM + dM1 + dM2 and dzdr!!

