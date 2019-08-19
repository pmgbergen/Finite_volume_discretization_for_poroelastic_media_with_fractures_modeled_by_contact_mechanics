"""
This is a setup class for solving Biot with contact between the fractures.
"""
import numpy as np
import scipy.sparse as sps
import pickle
import porepy as pp

import discretizations
import setup_2


class Example4Setup(setup_2.Example2Setup):
    def __init__(self, mesh_args, out_name):
        super().__init__(mesh_args, out_name)
        self.mesh_args = mesh_args
        self.out_name = out_name
        self.store = True
        self.S = 1e-10 * 1 / pp.PASCAL
        self.k = 1e-11 * pp.METER**2
        self.viscosity = 1 * pp.MILLI * pp.PASCAL * pp.SECOND
        H = (self.domain['xmax'] - self.domain['xmin']) * self.length_scale
        self.end_time =  15 * self.viscosity * self.S * H**2 / self.k


    def set_parameters(self, g, data_node, mg, data_edge):
        """
        Set the parameters for the simulation. The stress is given in GPa.
        """
        # First set the parameters used in the pure elastic simulation
        key_m, key_c = super().set_parameters(g, data_node, mg, data_edge)
        key_f = 'flow'

        if not key_m == key_c:
            raise ValueError('Mechanics keyword must equal contact keyword')
        self.key_m = key_m
        self.key_f = key_f
        # Set fluid parameters
        kxx = self.k * np.ones(g.num_cells) / self.length_scale**2
        viscosity = self.viscosity / self.pressure_scale
        K = pp.SecondOrderTensor(g.dim, kxx / viscosity)

        # Define Biot parameters
        alpha = 1
        dt = self.end_time / 20
        # Define the finite volume sub grid
        s_t = pp.fvutils.SubcellTopology(g)

        # Define boundary conditions for flow
        top = g.face_centers[2] > np.max(g.nodes[2]) - 1e-9
        bot = g.face_centers[2] < np.min(g.nodes[2]) + 1e-9
        east = g.face_centers[0] > np.max(g.nodes[0]) - 1e-9
        bc_flow = pp.BoundaryCondition(g, top, 'dir')
        bc_flow = pp.fvutils.boundary_to_sub_boundary(bc_flow, s_t)    

        # Set boundary condition values.
        p_bc = self.bc_values(g, dt, key_f)

        # Set initial solution
        u0 = np.zeros(g.dim * g.num_cells)
        p0 = np.zeros(g.num_cells)
        lam_u0 = np.zeros(g.dim * mg.num_cells)
        u_bc0 = self.bc_values(g, 0, key_m)
        u_bc = self.bc_values(g, dt, key_m)
        # Collect parameters in dictionaries

        # Add biot parameters to mechanics
        data_node[pp.PARAMETERS][key_m]['biot_alpha'] = alpha
        data_node[pp.PARAMETERS][key_m]['time_step'] = dt
        data_node[pp.PARAMETERS][key_m]['bc_values'] = u_bc
        data_node[pp.PARAMETERS][key_m]['state']= {'displacement':  u0,
                                                   'bc_values': u_bc0}

        data_edge[pp.PARAMETERS][key_c]['state'] = lam_u0

        # Add fluid flow dictionary
        data_node = pp.initialize_data(
            g, data_node, key_f, 
            {'bc': bc_flow,
             'bc_values': p_bc.ravel('F'),
             'second_order_tensor': K,
             'mass_weight': self.S,
             'aperture': np.ones(g.num_cells),
             'biot_alpha': alpha,
             'time_step': dt,
             'state': p0,
        }
        )

        # Define discretization.
        # For the domain we solve linear elasticity with mpsa and fluid flow with mpfa.
        # In addition we add a storage term (ImplicitMassMatrix) to the fluid mass balance.
        # The coupling terms are:
        # BiotStabilization, pressure contribution to the div u term.
        # GrapP, pressure contribution to stress equation.
        # div_u, displacement contribution to div u term.
        data_node[pp.PRIMARY_VARIABLES] = {"u": {"cells": g.dim}, "p": {"cells": 1}}

        mpfa_disc = discretizations.ImplicitMpfa(key_f)
        data_node[pp.DISCRETIZATION] = {
            "u": {"div_sigma": pp.Mpsa(key_m)},
            "p": {
                "flux": mpfa_disc,
                "mass": discretizations.ImplicitMassMatrix(key_f),
                "stab": pp.BiotStabilization(key_f),
            },
            "u_p": {"grad_p": pp.GradP(key_m)},
            "p_u": {"div_u": pp.DivD(key_m)},
        }

        # On the mortar grid we define two variables and sets of equations. The first
        # adds a Robin condition to the elasticity equation. The second enforces full
        # fluid pressure and flux continuity over the fractures. We also have to be
        # carefull to obtain the contribution of the coupling discretizations gradP on
        # the Robin contact condition, and the contribution from the mechanical mortar
        # variable on the div_u term.

        # Contribution from fluid pressure on displacement jump at fractures
        gradP_disp = pp.numerics.interface_laws.elliptic_interface_laws.RobinContactBiotPressure(
            key_m, pp.numerics.fv.biot.GradP(key_m)
        )
        # Contribution from mechanics mortar on div_u term
        div_u_lam = pp.numerics.interface_laws.elliptic_interface_laws.DivU_StressMortar(
            key_m, pp.numerics.fv.biot.DivD(key_m)
        )
        # gradP_disp and pp.RobinContact will now give the correct Robin contact
        # condition.
        # div_u (from above) and div_u_lam will now give the correct div u term in the
        # fluid mass balance
        data_edge[pp.PRIMARY_VARIABLES] = {"lam_u": {"cells": g.dim}}
        data_edge[pp.COUPLING_DISCRETIZATION] = {
            "robin_discretization": {
                g: ("u", "div_sigma"),
                g: ("u", "div_sigma"),
                (g, g): ("lam_u", pp.RobinContact(key_m, pp.Mpsa(key_m))),
            },
            "p_contribution_to_displacement": {
                g: ("p", "flux"), # "flux" should be "grad_p", but the assembler does not
                g: ("p", "flux"), # support this. However, in FV this is not used anyway.
                (g, g): ("lam_u", gradP_disp),
            },
            "lam_u_contr_2_div_u": {
                g: ("p", "flux"), # "flux" -> "div_u"
                g: ("p", "flux"),
                (g, g): ("lam_u", div_u_lam),
            },
        }
        # Discretize with biot
        pp.Biot(key_m, key_f).discretize(g, data_node)
        return key_m, key_f

    def bc_values(self, g, t, key):
        # Define the finite volume sub grid
        s_t = pp.fvutils.SubcellTopology(g)
        # Define boundary conditions for flow
        top = g.face_centers[2] > np.max(g.nodes[2]) - 1e-9
        bot = g.face_centers[2] < np.min(g.nodes[2]) + 1e-9
        east = g.face_centers[0] > np.max(g.nodes[0]) - 1e-9

        if key==self.key_m:
            # Set boundary condition values. Remember stress is scaled with giga
            A = g.face_areas[s_t.fno_unique][top[s_t.fno_unique]] / 3
            u_bc = np.zeros((g.dim, s_t.num_subfno_unique))
            if t < 1e-12:
                return u_bc.ravel('F')
            u_bc[2, top[s_t.fno_unique]] = (
                -4.5 * pp.MEGA * pp.PASCAL / self.pressure_scale * A
                )
            return u_bc.ravel('F')
        elif key==self.key_f:
            # Set boundary condition values. Remember stress is scaled with giga
            p_bc = np.zeros(s_t.num_subfno_unique)
            p_bc[top[s_t.fno_unique]] = 0 * pp.MEGA * pp.PASCAL / self.pressure_scale
            return p_bc
        else:
            raise ValueError('Unknown keyword: ' +  key)
