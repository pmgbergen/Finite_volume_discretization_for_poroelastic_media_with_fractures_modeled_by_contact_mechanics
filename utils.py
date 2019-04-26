"""
This module inclueds some utility functions and mappings.
"""
import numpy as np
import scipy.sparse as sps
import matplotlib.pyplot as plt
import porepy as pp

def get_mappings(g, mg, s_t):
    """
    Given a grid, mortar grid and subcell topology get mapping between slave and mortar
    and master and mortar.
    Parameters:
        g:  Grid
        mg: MortarGrid
        s_t: SubcellTopology
    Returns:
        sgn_bnd_hf: The sign on the mortar faces
        slv_2_mrt_nd: Mapping from slave to mortar (for nd vectors)
        mstr_2_mrt_nd: Mapping from master to mortar (for nd vectors)
    """
    # find sign of boundary faces
    sgn = pp.numerics.fracture_deformation.sign_of_faces(g, g.get_all_boundary_faces())
    sgn_bnd = np.ones(g.num_faces, dtype=np.int)
    sgn_bnd[g.get_all_boundary_faces()] = sgn
    sgn_bnd_hf = sgn_bnd[s_t.fno_unique]
    
    # Define some convenient mappings
    slv_2_mrt = mg.slave_to_mortar_int()
    slv_2_mrt_nd = sps.kron(slv_2_mrt, sps.eye(g.dim))
    mstr_2_mrt = mg.master_to_mortar_int()
    mstr_2_mrt_nd = sps.kron(mstr_2_mrt, sps.eye(g.dim))

    return sgn_bnd_hf, slv_2_mrt_nd, mstr_2_mrt_nd


def normal_tangential_rotations(g, mg):
    """
    Calculate the normal and two tangential directions on the mortar grid.
    The normal vector is taken from the corresponding slave boundary, while the
    tangential components are taken from the Gram Schimt procedure using the
    vectors n, t1* = [1,0,0]^T and t2* = [0,1,0]^T. This will give three orthonormal
    vectors: n, t1, and t2

    Parameters:
        g: Grid
        mg: MortarGrid

    Returns:
        M_inv: The inverse of the basis matrix [t1, t2, n] for each mortar subcell
        nc: the normal vectors n for each mortar subcell
    """
    # Get subcell topology
    s_t = pp.fvutils.SubcellTopology(g)
    # Define some convenient mappings
    sgn_bnd_hf, slv_2_mrt_nd, mstr_2_mrt_nd = get_mappings(g, mg, s_t)
    # Define normal and tangential directions
    nf = g.face_normals[:g.dim] / g.face_areas
    nc = slv_2_mrt_nd * (sgn_bnd_hf * nf[:, s_t.fno_unique]).ravel('F')
    nc = nc.reshape((g.dim, -1), order="F")
    if g.dim==3:
        t1 = np.tile(np.array([[1, 0, 0]]).T, (1, mg.num_cells))
        t2 = np.tile(np.array([[0, 1, 0]]).T, (1, mg.num_cells))
        nc, tc1, tc2 = gram_schmidt(nc, t1, t2)
        basis = np.hstack([tc1, tc2, nc])
    else:
        t1 = np.tile(np.array([[1,  0]]).T, (1, mg.num_cells))
        nc, tc1 = gram_schmidt(nc, t1)
        basis = np.hstack([tc1, nc])
    basis = basis.reshape((g.dim, g.dim, mg.num_cells))
    M_inv = inverse_3dmatrix(basis)
    return M_inv, nc

def color_mortar_grid(g, mg, mg_f2c):
    """
    Color the mortar grid according to connectivity, i.e., each fracture will be assigned
    one color. The color (an int) can be found for each mortar subcell as mg.color

    Parameters:
    g: Grid
    mg: MortarGrid (with mappings from subfaces to subcells)
    mg_f2c: MortarGrid (with mappings from faces to cells)

    Returns:
    None
    """
    # color the grids to be able to seperate them
    mg_side = mg.side_grids['mortar_grid']
    color = pp.fracs.split_grid.find_cell_color(mg_side, np.arange(mg_side.num_cells))
    hf2f = pp.fvutils.map_hf_2_f(nd=1, g=g)
    mg.color = mg.slave_to_mortar_avg() * hf2f.T * mg_f2c.slave_to_mortar_avg().T * color
    mg.color = mg.color.astype(int)

def subface_to_face_mortar(g, mg, data_edge, Tc, uc):
    """
    Map the subface values of traction Tc and displacement jumps uc (subcell on mortar
    grid) to the face values (cells on mortar grid). It is assumed that the mortar grid

    Parameters:
        g: Grid
        mg: MortarGrid
        data_edge: Data dictionary for mortar grid
        Tc: Traction on mortar subcells
        uc: Displacement jump on mortar subcells

    Returns:
        Tc: Traction on mortar cells (the sum of subcell-values)
        uc: Displacement jump on mortar cells (the average of subcell values)
    """
    s_t = pp.fvutils.SubcellTopology(g)
    sgn_bnd_hf, slv_2_mrt_nd, mstr_2_mrt_nd = get_mappings(g, mg, s_t)
    # Get mortar mappings for faces
    mg_f2c = data_edge['mortar_grid_f2c']

    slavef2mortarc_nd = sps.kron(mg_f2c.slave_to_mortar_int(), sps.eye(g.dim))
    hf2f = pp.fvutils.map_hf_2_f(g=g)
    num_nodes = mg_f2c.slave_to_mortar_int() * np.diff(g.face_nodes.indptr)
    # Get full face valued vectors
    Tcc = slavef2mortarc_nd * hf2f * slv_2_mrt_nd.T * Tc.ravel('F')
    Tcc = Tcc.reshape((g.dim, -1), order='F')
    ucc = slavef2mortarc_nd * hf2f * slv_2_mrt_nd.T * uc.ravel('F')
    ucc = ucc.reshape((g.dim, -1), order='F') / num_nodes
    return Tcc, ucc

def gram_schmidt(u1, u2, u3=None):
    """
    Perform a Gram Schmidt procedure for the vectors u1, u2 and u3 to obtain a set of
    orhtogonal vectors.
    
    Parameters:
        u1: ndArray
        u2: ndArray
        u3: ndArray

    Returns:
        u1': ndArray u1 / ||u1||
        u2': ndarray (u2 - u2*u1 * u1) / ||u2||
        u3': (optional) ndArray (u3 - u3*u2' - u3*u1')/||u3||
    """
    u1 = u1 / np.sqrt(np.sum(u1**2, axis=0))

    u2 = u2 - np.sum(u2 * u1, axis=0) * u1
    u2 = u2 / np.sqrt(np.sum(u2**2, axis=0))

    if u3 is None:
        return u1, u2
    u3 = u3 - np.sum(u3 * u1, axis=0) * u1 - np.sum(u3 * u2, axis=0) * u2
    u3 = u3 / np.sqrt(np.sum(u3**2, axis=0))
    
    return u1, u2, u3

def inverse_3dmatrix(M):
    """
    Find the inverse of the (m,m,k) 3D ndArray M. The inverse is intrepreted as the
    2d inverse of M[:, :, i] for i = 0...k
    
    Parameters:
    M: (m, m, k) ndArray

    Returns:
    M_inv: Inverse of M
    """
    M_inv = np.zeros(M.shape)
    for i in range(M.shape[-1]):
        M_inv[:, :, i] = np.linalg.inv(M[:, :, i])
    return M_inv
    

def transform_coordinates(vec, basis_inv):
    """
    Calculate the coordinates of a vector in given coordinate system.
    
    Parameters:
    vec: ndArray
    basis_inv: inverse of basis matrix (e1, e2, e2)

    Returns:
    vec_b: vector vec given in basis e1, e2, e3.    
    """
    vec_b = np.zeros(vec.shape)
    for i in range(vec.shape[1]):
        T_full = np.atleast_2d(vec[:, i]).T
        M_inv = basis_inv[:, :, i]
        vec_b[:, i] = M_inv.dot(T_full).ravel()
    return vec_b

def project_coordinates(vec, basis_inv):
    """
    Project coordinates given in a basis matrix basis_inv to Cartesian coordinates.

    Parameters:
    vec: Vector given by the inverse of basis_inv
    basis_inv: inverse of the basis matrix.

    Returns:
    vec: vector vec given in Cartesian basis.
    """
    vec_b = np.zeros(vec.shape)
    for i in range(vec.shape[1]):
        T_full = np.atleast_2d(vec[:, i]).T
        M = np.linalg.inv(basis_inv[:, :, i])
        vec_b[:, i] = M.T.dot(T_full).ravel()
    return vec_b

def sign_of_faces(g, faces):
    """
    returns the sign of faces as defined by g.cell_faces. 
    Parameters:
    g: (Grid Object)
    faces: (ndarray) indices of faces that you want to know the sign for. The 
           faces must be boundary faces.

    Returns:
    sgn: (ndarray) the sign of the faces
    """

    IA = np.argsort(faces)
    IC = np.argsort(IA)

    fi, _, sgn = sps.find(g.cell_faces[faces[IA], :])
    assert fi.size == faces.size, "sign of internal faces does not make sense"
    I = np.argsort(fi)
    sgn = sgn[I]
    sgn = sgn[IC]
    return sgn

def l2(x):
    """
    l2 vector norm
    """
    x = np.atleast_2d(x)
    return np.sqrt(np.sum(x**2, axis=0))


def construct_nodal_values(g, mg, data_node, u, p, Tc, key, key_m):
    """
    Given a solution reconstruct the nodel displacements.
    """
    # Get the subcell topology and mappings to and from mortar grid
    s_t = pp.fvutils.SubcellTopology(g)
    sgn_bnd_hf, slv_2_mrt_nd, mstr_2_mrt_nd = get_mappings(g, mg, s_t)

    ## Obtain the nodal values (eta = 0). Note that we have to rediscretize
    k = data_node[pp.PARAMETERS][key]['fourth_order_tensor']
    bc = data_node[pp.PARAMETERS][key]['bc']
    u_bc = data_node[pp.PARAMETERS][key]['bc_values']
    eta_vec = data_node[pp.PARAMETERS][key]['mpsa_eta']
    eta_node = 1 * np.ones(s_t.num_subfno_unique)
    _, _, disp_cell, disp_bound = pp.numerics.fv.mpsa.mpsa(
        g, k, bc, hf_eta=eta_node, eta=eta_vec
        )

    # Map mortar variable to slave and master side
    T_slave = (slv_2_mrt_nd.T * Tc.ravel('F'))
    T_master = -(mstr_2_mrt_nd.T * Tc.ravel('F'))
    bound_val = u_bc + T_slave + T_master

    # Get nodel displacements
    u_hf = (
        disp_cell * u.ravel('F') + disp_bound * bound_val.ravel('F')
    ).reshape((g.dim, -1), order='F')
    if p is not None:
        disp_pressure = data_node[pp.DISCRETIZATION_MATRICES][key_m]['bound_displacement_pressure']
        u_hf += (disp_pressure * p).reshape(g.dim, -1, order='F')

    # For each node, we have one value per cell connected to the node. Average these
    # values.
    u_n = np.zeros((g.dim, g.num_nodes))
    num_hit = np.zeros(g.num_nodes)
    for i, f in enumerate(s_t.subfno_unique):
        u_n[:, s_t.nno_unique[i]] += u_hf[:, f]
        num_hit[s_t.nno_unique[i]] += 1
    u_n /= num_hit
    return u_n


def plot_vec(vec, start, *vargs, **kwargs):
    if vec.shape[0]==3:
        _plot_3d(vec, start, *vargs, **kwargs)
    else:
        _plot_2d(vec, start, *vargs, **kwargs)

def _plot_2d(vec, start, *vargs, **kwargs):
    start_x = start[0]
    start_y = start[1]
    end_x = start_x + vec[0]
    end_y = start_y + vec[1]
    for i in range(start_x.size):
        plt.plot([start_x[i], end_x[i]], [start_y[i], end_y[i]], *vargs, **kwargs)

def _plot_3d(vec, start, *vargs, **kwargs):
    ax = plt.gca()
    start_x = start[0]
    start_y = start[1]
    start_z = start[2]
    end_x = start_x + vec[0]
    end_y = start_y + vec[1]
    end_z = start_z + vec[2]

    for i in range(start_x.size):
        ax.plot(
            [start_x[i], end_x[i]],
            [start_y[i], end_y[i]],
            [start_z[i], end_z[i]], *vargs, **kwargs)
