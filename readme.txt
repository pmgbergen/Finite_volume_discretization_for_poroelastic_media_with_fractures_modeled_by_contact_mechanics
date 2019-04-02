This repository contains the run-scripts for the paper:

Finite volume discretization for poroelastic media with fractures
modeled by contact mechanics

by
Runar L. Berge, Barbara Wohlmuth, Eirik Keilegavlen, Inga Berre, and Jan M. Nordbotten

To run the examples contained in this repository a working version of porepy (which can
be downloaded from https://github.com/pmgbergen/porepy) must be installed on the computer.
The code within this repository was develop for the porepy version given by the commit:
bcdc83b01c295cfb6b1189e7934308df83f83a74

This repository contains the following files:
contact.py    Calculate the Robin boundary condition resulting from the Newton scheme
models.py     Solve the elastic/biot equations with contact using Newton
my_meshign.py Create a mesh with mortars on the fractures
solvers.py    Linear solvers for the elastic/biot system with contact
utils.py      Various utility functions
viz.py        Module for vizualization

In addition there are three subfolders, example_1, example_2 and example_3 with the same
structure:
example_*/
    setup.py    Setup of example * in the paper listed above
    main.py     Run this script to run example *

Note that the meshsizes set in main.py is larger than the ones used in the paper to reduce
the run times.
