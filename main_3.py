"""
Main run script for running example 3.
"""
import setup
import commons
import setup

run_name = 'example_4_0.005'
# Mesh size in paper:
# size = 0.005
size = 0.5
mesh_args = {'mesh_size_frac': size,
             'mesh_size_bound': 2 * size,
             'mesh_size_min': 0.1 * size}

setup_biot = setup.Example3Setup(mesh_args, run_name)

commons.run_biot(setup_biot)
