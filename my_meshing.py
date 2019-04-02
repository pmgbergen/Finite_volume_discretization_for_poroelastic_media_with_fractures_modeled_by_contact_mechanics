"""
Utility module for the meshing. Mainly to create the mortar grids given a grid bucket
with fractures.
"""

import numpy as np
import scipy.sparse as sps
import porepy as pp


def create_mortar_grids(gb):
    """
    Given a grid bucket, create new edges (in the grid bucket) that maps the slave side
    of the fractures to the master side.
    
    Arguments:
        gb : GridBucket
    """
    # First we add a edge from each grid to itself.
    for g, d in gb:
        # Define a mapping from faces to master and faces to slave
        master_id = g.frac_pairs[0]
        slave_id = g.frac_pairs[1]
        if master_id.size == 0:
            continue
        num_m = master_id.size
        if num_m != slave_id.size:
            raise ValueError("Mortar and slave side of fracture must match")
        rows = np.arange(num_m)
        master_face = sps.coo_matrix(
            (np.ones(num_m), (rows, master_id)), (num_m, g.num_faces)
        ).tocsc()
        slave_face = sps.coo_matrix(
            (np.ones(num_m), (rows, slave_id)), (num_m, g.num_faces)
        ).tocsc()
        # Find the mapping from slave to master
        slave_master = master_face.T * slave_face
        slave_grid, face_id, _ = pp.partition.extract_subgrid(g, slave_id, faces=True, is_planar=False)
        # do some checks that all mappings have been correct
        # Check that slave and master coincide

        if np.all(face_id != slave_id):
            raise AssertionError("Assume extract subgrid does not change ordering")

        if not np.all(
            [
                np.allclose((slave_face - master_face) * g.face_centers[dim], 0)
                for dim in range(g.dim)
            ]
        ):
            raise AssertionError("Slave and master do not conside")
        if not np.all(
            [
                np.allclose(
                    slave_master.T * g.face_centers[dim],
                    slave_face.T * g.face_centers[dim, slave_id]
                )
                for dim in range(g.dim)
            ]
        ):
            raise AssertionError("Slave faces do not coinside with mortar cells")

        if not np.all(
            [
                np.allclose(
                    slave_face * g.face_centers[dim], slave_grid.cell_centers[dim]
                )
                for dim in range(g.dim)
            ]
        ):
            raise AssertionError("Slave faces do not coinside with mortar cells")

        # Checks done.
        # Now create mortar grid
        mortar_grid = pp.BoundaryMortar(g.dim - 1, slave_grid, (slave_master.T))

        # Add edge to Grid bucket
        gb.add_edge((g, g), slave_master)
        d_e = gb.edge_props((g, g))
        d_e["mortar_grid"] = mortar_grid

    gb.assign_node_ordering()


# ------------------------------------------------------------------------------#

def map_mortar_to_submortar(gb):
    """
    Takes a GridBucket and changes the face_cell map of the mortar grid to a
    subface_to_subcell mapping.
    """
    for e, d, in gb.edges():
        mortar_grid = d['mortar_grid']
        g_slave, g_master = gb.nodes_of_edge(e)
        if g_slave.dim != g_master.dim:
            continue
        # We expand the mortar grid according to the subcell topology
        # First update master
        master_hf2f = pp.fvutils.map_hf_2_f(nd=1, g=g_master)
        master_s_t = pp.fvutils.SubcellTopology(g_master)
        master_hf2m = mortar_grid.master_to_mortar_int() * master_hf2f
        master2submortar = mortar_2_sub_mortar(master_s_t, master_hf2m)
        mortar_grid._master_to_mortar_int = master2submortar
        # Then update slave
        slave_hf2f = pp.fvutils.map_hf_2_f(nd=1, g=g_slave)
        slave_hf2m = mortar_grid.slave_to_mortar_int() * slave_hf2f

        # Then update mortar grid to be defined by the subcells
        slave_s_t = pp.fvutils.SubcellTopology(g_slave)
        slave2submortar = mortar_2_sub_mortar(slave_s_t, slave_hf2m)
        mortar_grid._slave_to_mortar_int = slave2submortar
        mortar_grid.num_cells = slave2submortar.shape[0]

def mortar_2_sub_mortar(s_t, proj):
    subfno_u = s_t.subfno_unique
    _, sf_ind = proj.nonzero()
    is_mortar = np.in1d(subfno_u, sf_ind)
    mortar_ind = np.arange(np.sum(is_mortar))
    sf2m = sps.coo_matrix(
        (np.ones(mortar_ind.size), (mortar_ind, sf_ind)),
        shape=(mortar_ind.size, s_t.num_subfno_unique)
    ).tocsc()
    return sf2m
