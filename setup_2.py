"""
This is a setup class for solving linear elasticity with contact between the fractures.
"""
import numpy as np
import scipy.sparse as sps
from scipy.spatial.distance import cdist
import porepy as pp
import copy

import my_meshing as meshing
import utils

class Example2Setup():
    def __init__(self, mesh_args, out_name):
        self.mesh_args = mesh_args
        self.out_name = out_name
        self.pressure_scale = pp.GIGA
        self.length_scale = 100 * pp.METER
        self.domain = {'xmin': -2, 'xmax': 3, 'ymin': -2, 'ymax': 3, 'zmin': -3, 'zmax': 3}

    def create_grid(self):
        """
        Method that creates and returns the GridBucket of a 3D domain with two
        fractures. The two sides of the fractures are coupled together with a
        mortar grid.
        """
        # define fractures
        f_1 = pp.EllipticFracture(np.array([-0.1, -0.3, -0.8]),
                                  1.5,
                                  1.5,
                                  np.pi/6.3,
                                  np.pi/2.2,
                                  np.pi/4.1,
                                  num_points=8)
        f_2 = pp.EllipticFracture(np.array([1.5, 0.6, 0.8]),
                                  1.5,
                                  1.5,
                                  0,
                                  np.pi/2.3,
                                  -np.pi/4.2,
                                  num_points=8)

        # Define a 3d FractureNetwork
        self.fractures = [f_1, f_2]
        network = pp.FractureNetwork3d([f_1, f_2], domain=self.domain)
        # Generate the mixed-dimensional mesh
        gb = network.mesh(self.mesh_args)
        
        # Remove fracture grid as it is not needed
        for g2 in gb.grids_of_dimension(2):
            gb.remove_node(g2)

        # Define the mortar grid
        meshing.create_mortar_grids(gb)

        g = gb.grids_of_dimension(3)[0]
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
        k = pp.FourthOrderTensor(g.dim, mu, lam)
        F = self._friction_coefficient(g, mg, data_edge, s_t)
        
        # Define boundary regions
        east = g.face_centers[0] > np.max(g.nodes[0]) - 1e-9
        west = g.face_centers[0] < np.min(g.nodes[0]) + 1e-9
        north = g.face_centers[1] > np.max(g.nodes[1]) - 1e-9
        south = g.face_centers[1] < np.min(g.nodes[1]) + 1e-9
        top = g.face_centers[2] > np.max(g.nodes[2]) - 1e-9
        bot = g.face_centers[2] < np.min(g.nodes[2]) + 1e-9

        top_hf = top[s_t.fno_unique]
        bot_hf = bot[s_t.fno_unique]

        # define boundary condition
        bc = pp.BoundaryConditionVectorial(g)
        bc.is_dir[0, east] = True      # Rolling
        bc.is_dir[0, west] = True      # Rolling
        bc.is_dir[1, north] = True     # Rolling
        bc.is_dir[1, south] = True     # Rolling
        bc.is_dir[:, bot] = True       # Dirichlet
        bc.is_neu[bc.is_dir + bc.is_rob] = False

        # extract boundary condition to subcells
        bc = pp.fvutils.boundary_to_sub_boundary(bc, s_t)

        # Give boundary condition values
        A = g.face_areas[s_t.fno_unique][top_hf] / 3
        u_bc = np.zeros((g.dim, s_t.num_subfno_unique))
        u_bc[2, top_hf] =  -4.5 * pp.MEGA * pp.PASCAL / self.pressure_scale * A

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
             'c': 100 * pp.GIGA * pp.PASCAL / self.pressure_scale}
        )

        # Define discretization
        # Solve linear elasticity with mpsa
        mpsa = pp.Mpsa(key)
        data_node[pp.PRIMARY_VARIABLES] = {'u': {'cells': g.dim}}
        data_node[pp.DISCRETIZATION] = {'u': {'mpsa': mpsa}}
        # Add a Robin condition to the mortar grid
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
        fs = self.fractures
        ccs = [f.center for f in fs]
        rs = [np.sqrt(np.sum((np.atleast_2d(f.p[:, 0]).T - f.center)**2)) for f in fs]
        F = np.zeros(mg.num_cells)
        for color in np.unique(mg.color):
            face_color = (mg.slave_to_mortar_avg().T * (mg.color == color)).astype(bool)
            fc = g.face_centers[:g.dim, s_t.fno_unique[face_color]]

            D = cdist(fc.T, ccs[color].T)
            ID = np.argmin(D, axis=1)
            D = np.min(D, axis=1)
            R = 0.1
            F[mg.color==color] = 0.5 * np.exp(R/(rs[color] - D) - R / rs[color])
        return F
