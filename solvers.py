#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Linear solvers. Includes an amg solver for pure elasticity with contact mechanics and a
fixed stress solver for Biot with contact mechanics
"""

import pickle
import numpy as np
import scipy.sparse as sps
import time
import porepy as pp
import pyamg

def amg(gb, A, b):
    """
    amg solver for elasticity with contact mechaics.
    """
    tic = time.time()

    # Linear solver starts here
    nd = gb.dim_max()
    g = gb.grids_of_dimension(nd)[0]
    num_cells = g.num_cells
    el_ind = np.arange(num_cells * nd)
    mortar_ind = np.arange(el_ind.size, A.shape[1])

    A_el = A[:, el_ind][el_ind, :]

    A_el_m = A[el_ind, :][:, mortar_ind]
    A_m_el = A[mortar_ind, :][:, el_ind]
    A_m_m = A[mortar_ind, :][:, mortar_ind]

    b_el = b[el_ind]

    n = num_cells * nd

    u = np.zeros(el_ind.size)
    m = np.zeros(mortar_ind.size)

    solution = np.zeros_like(b)

    def resid(x, y=None):
        if y is None:
            y = x[mortar_ind]
            x = x[el_ind]
        return np.sqrt(np.sum((b - A * np.hstack((x, y)))**2))

    residuals = []
    def callback(r):
#        print(r)
        residuals.append(r)

    # Preconditioning of the linear system, based on a Schur complement reduction to a
    # system in the elasticity variables.

    # Approximation of the inverse of A_m_m, by a diagonal matrix.
    # The quality of this for various configurations of the mortar variable is not clear.
    iA_m_m = sps.dia_matrix((1.0 / A_m_m.diagonal(), 0), A_m_m.shape)

    # Also create a factorization of the mortar variable. This is relatively cheap, so why not
    A_m_m_solve = sps.linalg.factorized(A_m_m)

    # Schur complement formulation, using the approximated inv(A_m_m)
    S = A_el - A_el_m * iA_m_m * A_m_el

    if False:
        # Activate this to use a direct solver for the Schur complement system. It will be
        # efficient, but memory demanding for large systems.
        solve_el = sps.linalg.factorized(S)
    else:
        # AMG preconditioner. We have not experimented with various parameters in the AMG
        # construction; the performance out of the box was considered satisfactory for the
        # moment.

        # Create an AMG hierarchy
        amg_args = {'presmoother': ('gauss_seidel', {'sweep': 'symmetric', 'iterations': 1}),
                'postsmoother':  ('gauss_seidel', {'sweep': 'symmetric', 'iterations': 1}),
                }
        ml = pyamg.smoothed_aggregation_solver(S, symmetry='nonsymmetric', **amg_args)
        print(ml)
        # Use as preconditioner

        solve_el = ml.aspreconditioner(cycle= 'V')

    def precond_schur(r):
        # Mortar residual
        rm = r[mortar_ind]
        # Residual for the elasticity is the local one, plus the mapping of the mortar residual
        r_el = r[el_ind] - A_el_m * A_m_m_solve(rm)
        # Solve, using specified solver
        du = solve_el(r_el)
        # Map back to mortar residual
        dm = A_m_m_solve(rm - A_m_el * du)
        return np.hstack((du, dm))

    # Maximum number of GMRES iterations. Set high, so as not to use restart
    max_it = 1000

    M = sps.linalg.LinearOperator(A.shape, precond_schur)
    solution, info = sps.linalg.gmres(
        A, b, M=M, restart=max_it, callback=callback, maxiter=max_it, tol=1e-12
    )

    print("linear solver residual: ", residuals[-1])
    print('Time: ' + str(time.time() - tic))
    return solution

def fixed_stress(gb, A, b, block_dof, full_dof, x0=None):
    """
    Fixed stress solver for Biot coupled with elasticity
    """
    tic = time.time()

    # Data loading step
    # Linear solver starts here
    nd = gb.dim_max()
    g = gb.grids_of_dimension(nd)[0]
    data = gb.node_props(g)
    num_cells = g.num_cells

    # Get indices of different dofs
    # full_dof contains the number of dofs per block. To get a global ordering, use
    global_dof = np.r_[0, np.cumsum(full_dof)]

    # split global variable
    block_u = block_dof[(g, "u")]
    block_p = block_dof[(g, "p")]
    block_lam_u = block_dof[((g, g), "lam_u")]
    if ((g, g), "lam_p") in block_dof:
        block_lam_p = block_dof[((g, g), "lam_p")]
        mortar_p_ind = np.arange(global_dof[block_lam_p], global_dof[block_lam_p + 1])
    else: # if we don't have a mortar variable for pressure (non-permeable fractures)
        mortar_p_ind = np.array([], dtype=np.int)
    # Get the global displacement and pressure dofs
    el_ind= np.arange(global_dof[block_u], global_dof[block_u + 1])
    p_ind = np.arange(global_dof[block_p], global_dof[block_p + 1])
    mortar_el_ind = np.arange(global_dof[block_lam_u], global_dof[block_lam_u + 1])


    num_mortar = full_dof[-1]

    # merge pressure and pressure mortar indices
    p_full_ind = np.hstack((p_ind, mortar_p_ind))
    rest_ind = np.hstack((el_ind, p_full_ind))

    # Extract submatrices
    A_rest_rest = A[rest_ind][:, rest_ind]
    A_rest_m = A[rest_ind][:, mortar_el_ind]
    A_m_rest = A[mortar_el_ind][:, rest_ind]

    A_m_m = A[mortar_el_ind, :][:, mortar_el_ind]

    b_el = b[el_ind]

    n = num_cells * nd

    solution = np.zeros_like(b)

    residuals = []
    def callback(r):
        residuals.append(r)

    # Preconditioning of the linear system, based on a Schur complement reduction to a
    # system in the elasticity variables.

    # Approximation of the inverse of A_m_m, by a diagonal matrix.
    # The quality of this for various configurations of the mortar variable is not clear.
    iA_m_m = sps.dia_matrix((1.0 / A_m_m.diagonal(), 0), A_m_m.shape)

    # Also create a factorization of the mortar variable. This is relatively cheap, so why not
    A_m_m_solve = sps.linalg.factorized(A_m_m)

    # Schur complement formulation, using the approximated inv(A_m_m)
    S = A_rest_rest - A_rest_m * iA_m_m * A_m_rest

    S_el = S[:, el_ind][el_ind, :]

    p_schur_ind = np.arange(el_ind.size, S.shape[0])

    S_p = S[:, p_schur_ind][p_schur_ind]

    S_el_p = S[el_ind][:, p_schur_ind]
    S_p_el = S[p_schur_ind][:, el_ind]

    # Create an AMG hierarchy
    amg_args = {'presmoother': ('gauss_seidel', {'sweep': 'symmetric', 'iterations': 2}),
            'postsmoother':  ('gauss_seidel', {'sweep': 'symmetric', 'iterations': 2}),
            }
    #ml = pyamg.smoothed_aggregation_solver(S_el, symmetry='nonsymmetric', **amg_args)
    ml = pyamg.rootnode_solver(S_el, symmetry='nonsymmetric', **amg_args)
    print(ml)
    # Use as preconditioner

    solve_el = ml.aspreconditioner(cycle= 'W')

    biot_alpha = data[pp.PARAMETERS]['mech']['biot_alpha']
    rock_type = data[pp.PARAMETERS]['mech']['rock']

    stab_size = p_schur_ind.size
    stab_vec = (
        biot_alpha**2 / (2 * (2 * rock_type.MU / nd + rock_type.LAMBDA)) * np.ones(stab_size)
        )
    stabilization = sps.dia_matrix((stab_vec, 0), shape=(stab_size, stab_size))

    solve_p = sps.linalg.factorized(S_p + stabilization)

    def solve_fixed_stress(r):
        r_p = r[p_schur_ind]
        dp = solve_p(r_p)
        r_el = r[el_ind] - S_el_p * dp

        d_el = solve_el(r_el)

        return np.hstack((d_el, dp))


    def precond_schur(r):
        # Mortar residual
        rm = r[mortar_el_ind]
        # Residual for the elasticity is the local one, plus the mapping of the mortar residual
        r_rest = r[rest_ind] - A_rest_m * A_m_m_solve(rm)
        # Solve, using specified solver
        du = solve_fixed_stress(r_rest)
        # Map back to mortar residual
        dm = A_m_m_solve(rm - A_m_rest * du)
        return np.hstack((du, dm))


    max_it = 1000

    M = sps.linalg.LinearOperator(A.shape, precond_schur)

    # Inital guess
    if x0 is None:
        x0 = np.zeros(A.shape[1])

    solution, info = sps.linalg.gmres(
        A, b, M=M, x0=x0, restart=max_it, callback=callback, maxiter=max_it, tol=1e-11
    )

    if len(residuals)>0:
        print("Linear solver residual: ", residuals[-1])
    print("Linear solver iterations: ", len(residuals))
    print('Linear solver time: ' + str(time.time() - tic))
    return solution, info, residuals
