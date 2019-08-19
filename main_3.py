"""
Main run script for running example 3.
"""
import setup_3
import models

run_name = 'example_3'
# Mesh size in paper:
# size = 0.005
size = 0.05
mesh_args = {'mesh_size_frac': size,
             'mesh_size_bound': 2 * size,
             'mesh_size_min': 0.1 * size}

setup_biot = setup_3.Example3Setup(mesh_args, run_name)

models.run_biot(setup_biot)
