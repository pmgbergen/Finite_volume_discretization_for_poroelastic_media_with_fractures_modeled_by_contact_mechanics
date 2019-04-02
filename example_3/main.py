"""
Main run script for running example 2.
"""
import setup
import models

mesh_args = {'mesh_size_frac': 0.5, 'mesh_size_min': 0.1 * 0.1, 'mesh_size_bound': 0.8}
run_name = 'example_3'
setup = setup.Example3Setup(mesh_args, run_name)

models.run_biot(setup)
