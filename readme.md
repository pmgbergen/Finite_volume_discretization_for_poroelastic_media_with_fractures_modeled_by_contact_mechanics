[![DOI](https://zenodo.org/badge/179075991.svg)](https://zenodo.org/badge/latestdoi/179075991)

This repository contains the run-scripts for the paper:

Finite volume discretization for poroelastic media with fractures
modeled by contact mechanics

by
Runar L. Berge, Inga Berre, Eirik Keilegavlen, Jan M. Nordbotten, and Barbara Wohlmuth

To run the examples contained in this repository a working version of PorePy (which can
be downloaded from https://github.com/pmgbergen/porepy) must be installed on the computer.
The code within this repository was develop for the PorePy version given by the commit:
bcdc83b01c295cfb6b1189e7934308df83f83a74

The PorePy install requires installations of further packages, see Install instructions
in the PorePy repository.

Moreover, the simulations use the PyAMG package to solve the linear system resulting 
from the contact mechanics problem. Install by 

	pip install pyamg

or 

	conda install pyamg

This repository contains the following files:
contact.py:	Calculate the Robin boundary condition resulting from the Newton scheme  
main_1.py:   	Run script Example 1 in paper  
main_2.py:     	Run script Example 2 in paper  
main_3.py:     	Run script Example 3 in paper  
models.py:     	Solve the elastic/biot equations with contact using Newton  
my_meshing.py: 	Create a mesh with mortars on the fractures   
setup_1.py:    	Setup file for example 1  
setup_2.py:    	Setup file for example 2  
setup_3.py:    	Setup file for example 3  
solvers.py:    	Linear solvers for the elastic/biot system with contact  
utils.py:      	Various utility functions  
viz.py:        	Module for vizualization  

The subfolders reference_plot_ex_* contains the vtk files created by running the main_* scripts.

Note that the meshsizes set in main_*.py are larger than the ones used in the paper to reduce
the run times.
