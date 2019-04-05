This repository contains the run-scripts for the paper:

Finite volume discretization for poroelastic media with fractures
modeled by contact mechanics

by
Runar L. Berge, Barbara Wohlmuth, Eirik Keilegavlen, Inga Berre, and Jan M. Nordbotten

To run the examples contained in this repository a working version of PorePy (which can
be downloaded from https://github.com/pmgbergen/porepy) must be installed on the computer.
The code within this repository was develop for the porepy version given by the commit:
bcdc83b01c295cfb6b1189e7934308df83f83a74

The PorePy install requires installations of further packages, see Install instructions
in the PorePy repository.

Moreover, the simulations use the PyAMG package to solve the linear system resulting 
from the contact mechanics problem. Install by 

	pip install pyamg

or 

	conda install pyamg

This repository contains the following files:
contact.py    Calculate the Robin boundary condition resulting from the Newton scheme
models.py     Solve the elastic/biot equations with contact using Newton
my_meshign.py Create a mesh with mortars on the fractures
solvers.py    Linear solvers for the elastic/biot system with contact
utils.py      Various utility functions
viz.py        Module for vizualization

The examples are found in files 
	main_1.py, 
	main_2.py 
	main_3.py 

corresponding to the examples in the paper. The setups are the correspondingly named setup_*.py.

Note that the meshsizes set in main_*.py are larger than the ones used in the paper to reduce
the run times.
