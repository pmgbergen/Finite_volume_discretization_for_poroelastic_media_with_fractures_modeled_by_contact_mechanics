"""
This module contains functions for solving a contact problem. It has functions
for solving the elastic equations and the time dependent Biot equations, both with a
non-linear contact condition. The non-linearity is handled by applying a semi-smooth
Newton method.
"""
import numpy as np
import scipy.sparse as sps
import porepy as pp
import pickle

import utils
from contact import contact_coulomb
import viz
import solvers

def run_mechanics(setup):
    """
    Function for solving linear elasticity with a non-linear Coulomb contact.
    
    There are some assumtions on the variable and discretization names given to the
    grid bucket:
        'u': The displacement variable
        'lam': The mortar variable
        'mpsa': The mpsa discretization
    
    In addition to the standard parameters for mpsa we also require the following
    under the contact mechanics keyword (returned from setup.set_parameters):
        'friction_coeff' : The coefficient of friction
        'c' : The numerical parameter in the non-linear complementary function.

    Arguments:
        setup: A setup class with methods:
                set_parameters(g, data_node, mg, data_edge): assigns data to grid bucket.
                    Returns the keyword for the linear elastic parameters and a keyword
                    for the contact mechanics parameters.
                create_grid(): Create and return the grid bucket
                initial_condition(): Returns initial guess for 'u' and 'lam'.
            and attributes:
                out_name(): returns a string. The data from the simulation will be
                written to the file 'res_data/' + setup.out_name and the vtk files to
                'res_plot/' + setup.out_name
    """
    gb = setup.create_grid()
    # Extract the grids we use
    dim = gb.dim_max()
    g = gb.grids_of_dimension(dim)[0]
    data_node = gb.node_props(g)
    data_edge = gb.edge_props((g, g))
    mg = data_edge['mortar_grid']

    # set parameters
    key, key_m = setup.set_parameters(g, data_node, mg, data_edge)

    # Get shortcut to some of the parameters
    F = data_edge[pp.PARAMETERS][key_m]['friction_coeff']
    c_num = data_edge[pp.PARAMETERS][key_m]['c']

    # Define rotations
    M_inv, nc = utils.normal_tangential_rotations(g, mg)
    # Set up assembler and discretize
    mpsa = data_node[pp.DISCRETIZATION]['u']['mpsa']
    mpsa.discretize(g, data_node)
    assembler = pp.Assembler()

    # prepare for iteration
    u0, uc, Tc = setup.initial_condition(g, mg, nc)
    T_contact = []
    u_contact = []
    save_sliding = []

    errors = []

    counter_newton = 0
    converged_newton = False
    max_newton = 15

    while counter_newton <= max_newton and not converged_newton:
        print('Newton iteration number: ', counter_newton, '/', max_newton)
        counter_newton += 1
        # Calculate numerical friction bound used in  the contact condition
        bf = F * np.clip(np.sum(nc * (-Tc + c_num * uc) , axis=0), 0, np.inf)
        # Find the robin weight
        mortar_weight, robin_weight, rhs = contact_coulomb(Tc, uc, F, bf, c_num, c_num, M_inv)
        data_edge[pp.PARAMETERS][key_m]['robin_weight'] = robin_weight
        data_edge[pp.PARAMETERS][key_m]['mortar_weight'] = mortar_weight
        data_edge[pp.PARAMETERS][key_m]['robin_rhs'] = rhs

        # Re-discretize and solve
        A, b, block_dof, full_dof = assembler.assemble_matrix_rhs(gb)

        if gb.num_cells()>4000:
            sol = solvers.amg(gb, A, b)
        else:
            sol = sps.linalg.spsolve(A, b)
        # Split solution into displacement variable and mortar variable
        assembler.distribute_variable(gb, sol, block_dof, full_dof)
        u = data_node['u'].reshape((g.dim, -1), order='F')
        Tc = data_edge['lam'].reshape((g.dim, -1), order='F')

        # Reconstruct the displacement on the fractures
        uc = reconstruct_mortar_displacement(u, Tc, g, mg, data_node, data_edge, key, key_m)

        # Calculate the error
        if np.sum((u - u0)**2 * g.cell_volumes) / np.sum(u**2* g.cell_volumes) < 1e-10:
            converged_newton = True

        print('error: ', np.sum((u - u0)**2) / np.sum(u**2))
        errors.append(np.sum((u - u0)**2) / np.sum(u**2))

        # Prepare for next iteration
        u0 = u
        T_contact.append(Tc)
        u_contact.append(uc)

    # Store vtk of solution:
    viz.export_nodal_values(g, mg, data_node, u, None, Tc, key, key_m, setup.out_name, "res_plot")
    if dim==3:
        m_exp_name = setup.out_name + "_mortar_grid"
        viz.export_mortar_grid(g, mg, data_edge, uc, Tc, key, key_m, m_exp_name, "res_plot")


def run_biot(setup):
    """
    Function for solving the time dependent Biot equations with a non-linear Coulomb
    contact condition on the fractures.
    
    There are some assumtions on the variable and discretization names given to the
    grid bucket:
        'u': The displacement variable
        'p': Fluid pressure variable
        'lam': The mortar variable
    Furthermore, the parameter keyword from the elasticity is assumed the same as the
    parameter keyword from the contact condition.

    In addition to the standard parameters for Biot we also require the following
    under the mechanics keyword (returned from setup.set_parameters):
        'friction_coeff' : The coefficient of friction
        'c' : The numerical parameter in the non-linear complementary function.
    and for the parameters under the fluid flow keyword:
        'time_step': time step of implicit Euler.

    Arguments:
        setup: A setup class with methods:
                set_parameters(g, data_node, mg, data_edge): assigns data to grid bucket.
                    Returns the keyword for the linear elastic parameters and a keyword
                    for the contact mechanics parameters.
                create_grid(): Create and return the grid bucket
                initial_condition(): Returns initial guess for 'u' and 'lam'.
            and attributes:
                out_name(): returns a string. The data from the simulation will be
                    written to the file 'res_data/' + setup.out_name and the vtk files
                    to 'res_plot/' + setup.out_name
                end_time: End time time of simulation.
    """
    gb = setup.create_grid()
    # Extract the grids we use
    dim = gb.dim_max()
    g = gb.grids_of_dimension(dim)[0]
    data_node = gb.node_props(g)
    data_edge = gb.edge_props((g, g))
    mg = data_edge['mortar_grid']

    # set parameters
    key_m, key_f = setup.set_parameters(g, data_node, mg, data_edge)
    # Short hand for some parameters
    F = data_edge[pp.PARAMETERS][key_m]['friction_coeff']
    c_num = data_edge[pp.PARAMETERS][key_m]['c']
    dt = data_node[pp.PARAMETERS][key_f]["time_step"]

    # Define rotations
    M_inv, nc = utils.normal_tangential_rotations(g, mg)

    # Set up assembler and get initial condition
    assembler = pp.Assembler()

    u0 = data_node[pp.PARAMETERS][key_m]['state']['displacement'].reshape((g.dim, -1), order='F')
    p0 = data_node[pp.PARAMETERS][key_f]['state']
    Tc = data_edge[pp.PARAMETERS][key_m]['state'].reshape((g.dim, -1), order='F')

    # Reconstruct displacement jump on fractures
    uc = reconstruct_mortar_displacement(
            u0, Tc, g, mg, data_node, data_edge, key_m, key_f, pressure=p0
        )
    uc0=uc.copy()
    # Define function for splitting the global solution vector
    def split_solution_vector(x, block_dof, full_dof):
        # full_dof contains the number of dofs per block. To get a global ordering, use
        global_dof = np.r_[0, np.cumsum(full_dof)]

        # split global variable
        block_u = block_dof[(g, "u")]
        block_p = block_dof[(g, "p")]
        block_lam = block_dof[((g, g), "lam_u")]
        # Get the global displacement and pressure dofs
        u_dof = np.arange(global_dof[block_u], global_dof[block_u + 1])
        p_dof = np.arange(global_dof[block_p], global_dof[block_p + 1])
        lam_dof = np.arange(global_dof[block_lam], global_dof[block_lam + 1])

        # Plot pressure and displacements
        u = x[u_dof].reshape((g.dim, -1), order="F")
        p = x[p_dof]
        lam = x[lam_dof].reshape((g.dim, -1), order="F")
        return u, p, lam

    # prepare for time loop
    sol = None # inital guess for Newton solver. 
    T_contact = []
    u_contact = []
    save_sliding = []
    errors = []
    exporter = pp.Exporter(g, setup.out_name, 'res_plot')
    t = 0.0
    T = setup.end_time
    k = 0
    times = []
    newton_it=0
    while t < T:
        t += dt
        k += 1
        print('Time step: ', k, '/', int(np.ceil(T / dt)))

        times.append(t)
        # Prepare for Newton
        counter_newton = 0
        converged_newton = False
        max_newton = 12
        newton_errors = []
        while counter_newton <= max_newton and not converged_newton:
            print('Newton iteration number: ', counter_newton, '/', max_newton)
            counter_newton += 1
            bf = F * np.clip(np.sum(nc * (-Tc + c_num * uc) , axis=0), 0, np.inf)
            ducdt = (uc - uc0) / dt
            # Find the robin weight
            mortar_weight, robin_weight, rhs = contact_coulomb(
                Tc, ducdt, F, bf, c_num, c_num, M_inv
            )
            rhs = rhs.reshape((g.dim, -1), order='F')
            for i in range(mg.num_cells):
                robin_weight[i] = robin_weight[i] / dt
                rhs[:, i] = rhs[:, i] + robin_weight[i].dot(uc0[:, i])
            data_edge[pp.PARAMETERS][key_m]['robin_weight'] = robin_weight
            data_edge[pp.PARAMETERS][key_m]['mortar_weight'] = mortar_weight
            data_edge[pp.PARAMETERS][key_m]['robin_rhs'] = rhs.ravel('F')

            # Re-discretize and solve
            A, b, block_dof, full_dof = assembler.assemble_matrix_rhs(gb)
            print('max A: ', np.max(np.abs(A)))
            print('max A sum: ', np.max(np.sum(np.abs(A), axis=1)))
            print('min A sum: ', np.min(np.sum(np.abs(A), axis=1)))

            if A.shape[0]>10000:
                sol, msg, err = solvers.fixed_stress(gb, A, b, block_dof, full_dof, sol)
                if msg!=0:
                    # did not converge.
                    print('Iterative solver failed.')
            else:
                sol = sps.linalg.spsolve(A, b)

            # Split solution in the different variables
            u, p, Tc = split_solution_vector(sol, block_dof, full_dof)

            # Reconstruct displacement jump on mortar boundary
            uc = reconstruct_mortar_displacement(
                u, Tc, g, mg, data_node, data_edge, key_m, key_f, pressure=p
            )
            # Calculate the error
            if np.sum((u - u0)**2 * g.cell_volumes) / np.sum(u**2* g.cell_volumes) < 1e-10:
                converged_newton = True

            print('error: ', np.sum((u - u0)**2) / np.sum(u**2))
            newton_errors.append(np.sum((u - u0)**2) / np.sum(u**2))
            # Prepare for nect newton iteration
            u0 = u
            newton_it += 1

        errors.append(newton_errors)
        # Prepare for next time step
        uc0 = uc.copy()
        T_contact.append(Tc)
        u_contact.append(uc)

        mech_bc = data_node[pp.PARAMETERS][key_m]['bc_values'].copy()
        data_node[pp.PARAMETERS][key_m]['bc_values'] = setup.bc_values(g, t + dt, key_m)
        data_node[pp.PARAMETERS][key_f]['bc_values'] = setup.bc_values(g, t + dt, key_f)

        data_node[pp.PARAMETERS][key_m]["state"]["displacement"] = u.ravel('F').copy()
        data_node[pp.PARAMETERS][key_m]["state"]["bc_values"] = mech_bc
        data_node[pp.PARAMETERS][key_f]["state"] = p.copy()
        data_edge[pp.PARAMETERS][key_m]["state"] = Tc.ravel('F').copy()

        if g.dim==2:
            u_exp = np.vstack((u, np.zeros(u.shape[1])))
        elif g.dim==3:
            u_exp = u.copy()
            m_exp_name = setup.out_name + "_mortar_grid"
            viz.export_mortar_grid(
                g, mg, data_edge, uc, Tc, key_m, key_m, m_exp_name, "res_plot", time_step=k
            )

        exporter.write_vtk({"u": u_exp, 'p': p}, time_step=k)
        
    exporter.write_pvd(np.array(times))

def reconstruct_mortar_displacement(
        u, Tc, g, mg, data_node, data_edge, key_m, key_f, pressure=None
):
    """
    Reconstruct the displacement jump on the mortar grid. The displacement is given by
    construction by the cell centered displacements (and pressures for Biot) and
    Lagrange multipliers.
    """
    # Define some convenient mappings
    s_t = pp.fvutils.SubcellTopology(g)
    sgn_bnd_hf, slv_2_mrt_nd, mstr_2_mrt_nd = utils.get_mappings(g, mg, s_t)

    # Get discretizations
    u_bc = data_node[pp.PARAMETERS][key_m]['bc_values']
    stress = data_node[pp.DISCRETIZATION_MATRICES][key_m]['stress']
    bound_stress =  data_node[pp.DISCRETIZATION_MATRICES][key_m]['bound_stress']
    disp_cell = data_node[pp.DISCRETIZATION_MATRICES][key_m]['bound_displacement_cell']
    disp_bound = data_node[pp.DISCRETIZATION_MATRICES][key_m]['bound_displacement_face']

    if pressure is not None:
        grad_p =  data_node[pp.DISCRETIZATION_MATRICES][key_m]['grad_p']
        disp_pressure = data_node[pp.DISCRETIZATION_MATRICES][key_m]['bound_displacement_pressure']

    # First add traction from mortars and boundary conditions
    T_slave = (slv_2_mrt_nd.T * Tc.ravel('F'))
    T_master = -(mstr_2_mrt_nd.T * Tc.ravel('F'))
    bound_val = u_bc + T_slave + T_master

    # Then reconstruct the traction and displacements. Note the tractions are just
    # used as in the sanity checks below.
    # The grad_p contribution should by construction cancel
    if pressure is not None:
        assert np.allclose(slv_2_mrt_nd * grad_p * pressure, 0)
    T_hf = (
        stress * u.ravel('F') + bound_stress * bound_val
    ).reshape((g.dim, -1), order='F')
    T_hf = sgn_bnd_hf * T_hf

    # Get the displacements on all subfaces
    u_hf = (disp_cell * u.ravel('F') + disp_bound * bound_val).reshape(
        (g.dim, -1), order='F')
    if pressure is not None:
        u_hf += (disp_pressure * pressure).reshape(g.dim, -1, order='F')

    # Map to the slave and master sides
    u_slave = (slv_2_mrt_nd * u_hf.ravel('F')).reshape((g.dim, -1), order='F')
    u_master = (mstr_2_mrt_nd * u_hf.ravel('F')).reshape((g.dim, -1), order='F')

    # Calculate the jump
    uc = u_slave - u_master

    # Some sanity checks
    assert np.allclose(Tc.ravel('F'), slv_2_mrt_nd * T_hf.ravel('F'))
    assert np.allclose(slv_2_mrt_nd * T_hf.ravel('F'),
                       -mstr_2_mrt_nd * T_hf.ravel('F'))                           

    return uc
