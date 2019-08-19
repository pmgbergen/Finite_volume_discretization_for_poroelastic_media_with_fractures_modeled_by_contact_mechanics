"""
This is a setup class for solving linear elasticity with contact between the fractures.

The domain $[0, 2]\times[0, 1]$ with six fractures. We do not consider any fluid, and
solve only for the linear elasticity coupled to the contact
"""
import numpy as np
import scipy.sparse as sps
from scipy.spatial.distance import cdist
import porepy as pp
import copy

import my_meshing as meshing
import utils

class Example1Setup():
    def __init__(self, mesh_args, out_name):
        self.mesh_args = mesh_args
        self.out_name = out_name
        self.pressure_scale = pp.GIGA

    def create_grid(self):
        """
        Method that creates and returns the GridBucket of a 2D domain with six
        fractures. The two sides of the fractures are coupled together with a
        mortar grid.
        """

        # Create grid
        self.frac_pts = np.array([[0.2, 0.7], [0.5, 0.7], [0.8, 0.65],
                             [1, 0.3], [1.8,0.4],
                             [0.2, 0.3], [0.6, 0.25],
                             [1.0, 0.4], [1.7, 0.85],
                             [1.5, 0.65], [2.0, 0.55],
                             [1.5, 0.05], [1.4, 0.25]]).T
        frac_edges = np.array([[0,1], [1, 2], [3, 4], [5, 6], [7, 8], [9, 10],  [11, 12]]).T
        box = {'xmin': 0, 'ymin': 0, 'xmax': 2, 'ymax': 1}

        network = pp.FractureNetwork2d(self.frac_pts, frac_edges, domain=box)
        # Generate the mixed-dimensional mesh
        gb = network.mesh(self.mesh_args)

        # Remove fracture grid as it is not needed
        for g1 in gb.grids_of_dimension(1):
            gb.remove_node(g1)
        for g0 in gb.grids_of_dimension(0):
            gb.remove_node(g0)

        # Define the mortar grid
        meshing.create_mortar_grids(gb)
        g = gb.grids_of_dimension(2)[0]
        data_edge = gb.edge_props((g, g))
        mg = data_edge['mortar_grid']
        # Map the mortars to the subcell grid
        data_edge['mortar_grid_f2c'] = copy.deepcopy(mg) #keep copy of face to cell mortar
        meshing.map_mortar_to_submortar(gb)
        return gb

    def set_parameters(self, g, data_node, mg, data_edge):
        """
        Set the parameters for the simulation. The stress is given in GPa.
        """
        # Define the finite volume sub grid
        s_t = pp.fvutils.SubcellTopology(g)

        # Rock parameters
        rock = pp.Granite()
        lam =  rock.LAMBDA * np.ones(g.num_cells) / self.pressure_scale
        mu =  rock.MU * np.ones(g.num_cells) / self.pressure_scale
        F = self._friction_coefficient(g, mg, data_edge, s_t)

        k = pp.FourthOrderTensor(g.dim, mu, lam)

        # Define boundary regions
        top = g.face_centers[g.dim-1] > np.max(g.nodes[1]) - 1e-9
        bot = g.face_centers[g.dim-1] < np.min(g.nodes[1]) + 1e-9

        top_hf = top[s_t.fno_unique]
        bot_hf = bot[s_t.fno_unique]

        # Define boundary condition on sub_faces
        bc = pp.BoundaryConditionVectorial(g, top + bot, 'dir')
        bc = pp.fvutils.boundary_to_sub_boundary(bc, s_t)

        # Set the boundary values
        u_bc = np.zeros((g.dim, s_t.num_subfno_unique))
        u_bc[1, top_hf] =  -0.002
        u_bc[0, top_hf] = 0.005

        # Find the continuity points
        eta = 1/3
        eta_vec =  eta * np.ones(s_t.num_subfno_unique)

        cont_pnt = g.face_centers[:g.dim, s_t.fno_unique] + eta_vec * (
            g.nodes[:g.dim, s_t.nno_unique] - g.face_centers[:g.dim, s_t.fno_unique]
        )

        # collect parameters in dictionary
        key = 'mech'
        key_m = 'mech'

        data_node = pp.initialize_data(
            g, data_node, key, 
            {'bc': bc,
             'bc_values': u_bc.ravel('F'),
             'source': 0,
             'fourth_order_tensor': k,
             'mpsa_eta': eta_vec,
             'cont_pnt': cont_pnt,
             'rock': rock,
        }
        )

        pp.initialize_data(
            mg, data_edge, key_m,
            {'friction_coeff': F,
             'c': 100}
        )

        # Define discretization
        # For the 2D domain we solve linear elasticity with mpsa.
        mpsa = pp.Mpsa(key)
        data_node[pp.PRIMARY_VARIABLES] = {'u': {'cells': g.dim}}
        data_node[pp.DISCRETIZATION] = {'u': {'mpsa': mpsa}}

        # And define a Robin condition on the mortar grid
        contact = pp.numerics.interface_laws.elliptic_interface_laws.RobinContact(
            key_m, mpsa)
        data_edge[pp.PRIMARY_VARIABLES] = {'lam': {'cells': g.dim}}
        data_edge[pp.COUPLING_DISCRETIZATION] = {
            'robin_disc': {
                g: ('u', 'mpsa'),
                g: ('u', 'mpsa'),
                (g, g): ('lam',  contact),
            }
        }
        return key, key_m

    def initial_condition(self, g, mg, nc):
        """
        Initial guess for Newton iteration.
        """
        # Initial guess: no sliding
        u0 = np.zeros((g.dim, g.num_cells))
        Tc = -100*nc
        uc = np.zeros((g.dim, mg.num_cells))
        return u0, uc, Tc

    def _friction_coefficient(self, g, mg, data_edge, s_t):
        utils.color_mortar_grid(g, mg, data_edge['mortar_grid_f2c'])
        tips = self.frac_pts[:, [0, 1, 1,  2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]]
        rate = 200 * np.ones(tips.shape[1])
        if np.unique(mg.color).size==6:
            tip_color = np.array([0,-1, -1,0, 1,1,2,2,3,3,4,-1,5,5])
        else:
            tip_color = np.array([0, -1, -1, 1, 2,2,3,3,4,4,5,-1,6,6])
        F = np.zeros(mg.num_cells)
        for color in np.unique(mg.color):
            face_color = (mg.slave_to_mortar_avg().T * (mg.color == color)).astype(bool)
            fc = g.face_centers[:g.dim, s_t.fno_unique[face_color]]
            D = cdist(fc.T, tips[:, tip_color==color].T)
            ID = np.argmin(D, axis=1)
            D = np.min(D, axis=1)
            R = rate[tip_color==color][ID]
            beta = 10
            F[mg.color==color] = 0.5 * (1 + beta*np.exp(-R*D**2))
        return F
