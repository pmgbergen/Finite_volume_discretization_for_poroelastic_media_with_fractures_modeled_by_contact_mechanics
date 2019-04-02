"""
Main run script for running example 2.
"""
import setup
import models

run_name = 'example_1'
mesh_args = {'mesh_size_frac': 0.02, 'mesh_size_min': 0.001, 'mesh_size_bound':0.04}
setup = setup.Example1Setup(mesh_args, run_name)

models.run_mechanics(setup)
