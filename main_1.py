"""
Main run script for running example 1.
"""
import setup_1
import models

run_name = 'example_1'
mesh_args = {'mesh_size_frac': 0.02, 'mesh_size_min': 0.001, 'mesh_size_bound':0.04}
setup = setup_1.Example1Setup(mesh_args, run_name)

models.run_mechanics(setup)
