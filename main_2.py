"""
Main run script for running example 2.
"""
import setup_2
import models

mesh_args = {'mesh_size_frac': 0.5, 'mesh_size_min': 0.1 * 0.1, 'mesh_size_bound': 0.8}
run_name = 'example_2'
setup = setup_2.Example2Setup(mesh_args, run_name)

models.run_mechanics(setup)
