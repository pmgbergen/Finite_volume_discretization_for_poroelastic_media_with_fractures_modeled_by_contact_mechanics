"""
Module for visualizating the results
"""
import numpy as np
import scipy.sparse as sps
import matplotlib.pyplot as plt
import porepy as pp

import utils

def export_nodal_values(g, mg, data_node, u, p, Tc, key, key_m, *vargs, **kwargs):
    """
    Extrapolate the cell centered displacements to the nodes and export to vtk.
    """
    u_n = utils.construct_nodal_values(g, mg, data_node, u, p, Tc, key, key_m)
    # Export
    if g.dim==2:
        u_n_exp = np.vstack((u_n, np.zeros(u_n.shape[1])))
    else:
        u_n_exp = u_n
    exporter = pp.Exporter(g, *vargs, **kwargs)
    exporter.write_vtk({"u": u_n_exp}, point_data=True)


def export_mortar_grid(g, mg, data_edge, uc, Tc, key, key_m, *vargs, time_step=None, **kwargs):
    """
    Export the displacement jumps and mortar variable on the mortar grid.
    """
    # Get subcell topology and mappings to sub mortar grid
    s_t = pp.fvutils.SubcellTopology(g)
    sgn_bnd_hf, slv_2_mrt_nd, mstr_2_mrt_nd = utils.get_mappings(g, mg, s_t)

    # Get mortar mappings for faces also
    mg_f2c = data_edge['mortar_grid_f2c']
    slavef2mortarc_nd = sps.kron(mg_f2c.slave_to_mortar_int(), sps.eye(g.dim))
    hf2f = pp.fvutils.map_hf_2_f(g=g)

    # First get face valued vectors (from subface values)
    Tcc, ucc = utils.subface_to_face_mortar(g, mg, data_edge, Tc, uc)
    t_id = [0, 1]
    n_id = [2]
    M_inv, _ =  utils.normal_tangential_rotations(g, mg)
    M = utils.inverse_3dmatrix(M_inv)
    tc1 = M[:, 0, :]
    tc2 = M[:, 1, :]
    nc = M[:, 2, :]

    # Then get the tangential and normal components
    T_hat = np.sum(M_inv * Tc, axis=1)
    
    Tc_tau = T_hat[t_id[0]] * tc1 + T_hat[t_id[1]] *tc2
    Tc_n = T_hat[n_id] * nc
    Tcc_tau = slavef2mortarc_nd * hf2f * slv_2_mrt_nd.T * Tc_tau.ravel('F')
    Tcc_tau = Tcc_tau.reshape((g.dim, -1), order='F')
    Tcc_n = slavef2mortarc_nd * hf2f * slv_2_mrt_nd.T * Tc_n.ravel('F')
    Tcc_n = Tcc_n.reshape((g.dim, -1), order='F')

    # Sanity check
    assert np.allclose(Tcc_tau + Tcc_n, Tcc)

    # Export face values
    mg_plot = mg_f2c.side_grids['mortar_grid']

    exporter = pp.Exporter(mg_plot, *vargs, **kwargs)
    exporter.write_vtk({'Ttau': Tcc_tau / mg_f2c.cell_volumes,
                        'Tn': Tcc_n / mg_f2c.cell_volumes,
                        '[u]': ucc},
                       time_step=time_step)

def plot_bounding_box(g, plot_vars):
    """
    Plot the bounding box of a 2D grid.
    """
    x_min = np.min(g.nodes[0])
    x_max = np.max(g.nodes[0])
    y_min = np.min(g.nodes[1])
    y_max = np.max(g.nodes[1])

    corners_x = [x_min, x_max, x_max, x_min, x_min]
    corners_y = [y_min, y_min, y_max, y_max, y_min]
    plt.plot(corners_x, corners_y, 'k', **plot_vars)
