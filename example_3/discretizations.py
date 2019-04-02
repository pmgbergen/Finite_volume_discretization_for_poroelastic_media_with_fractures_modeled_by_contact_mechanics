"""
Module for extending the MPFA and MassMatrix discretizations in Porepy to handle
implicit Euler time-stepping.
"""
import porepy as pp

class ImplicitMassMatrix(pp.MassMatrix):
    def assemble_rhs(self, g, data):
        """ Overwrite MassMatrix method to
        Return the correct rhs for an IE time discretization of the Biot problem.
        """

        parameter_dictionary = data[pp.PARAMETERS][self.keyword]
        matrix_dictionary = data[pp.DISCRETIZATION_MATRICES][self.keyword]
        previous_pressure = parameter_dictionary["state"]

        return matrix_dictionary["mass"] * previous_pressure


class ImplicitMpfa(pp.Mpfa):
    def assemble_matrix_rhs(self, g, data):
        """ Overwrite MPFA method to be consistent with the Biot dt convention.
        """
        a, b = super().assemble_matrix_rhs(g, data)
        dt = data[pp.PARAMETERS][self.keyword]["time_step"]
        a = a * dt
        b = b * dt
        return a, b

    def assemble_int_bound_flux(self, g, data, data_edge, grid_swap, cc, matrix, rhs, self_ind):
        """
        Overwrite the MPFA method to be consistent with the Biot dt convention
        """
        div = g.cell_faces.T

        bound_flux = data[pp.DISCRETIZATION_MATRICES][self.keyword]["bound_flux"]
        # Projection operators to grid
        mg = data_edge["mortar_grid"]

        if grid_swap:
            proj = mg.slave_to_mortar_avg()
        else:
            proj = mg.master_to_mortar_avg()
        dt = data[pp.PARAMETERS][self.keyword]["time_step"]

        if bound_flux.shape[0] != g.num_faces:
            # If bound stress is gven as sub-faces we have to map it from sub-faces
            # to faces
            hf2f = pp.fvutils.map_hf_2_f(nd=1, g=g)
            bound_flux = hf2f * bound_flux
        if bound_flux.shape[1] != proj.shape[1]:
            raise ValueError(
                """Inconsistent shapes. Did you define a
            sub-face boundary condition but only a face-wise mortar?"""
            )

        cc[self_ind, 2] += dt * div * bound_flux * proj.T

