"""

Tests surface kernel, specifically with varying inputs,
in CARDIAX.

# TODO: batch these tests to check for different element types
        and gauss quadrature order - this is required to ensure
        the method is robust.

"""

import numpy as onp
import numpy.testing as onptest
import jax
import jax.numpy as np
import unittest

from cardiax import box_mesh
from cardiax import FiniteElement, Problem

# sample problem class to test surface function on
class HyperElasticity(Problem):

    def get_tensor_map(self):

        def psi(F):
            E = 10.
            nu = 0.3
            mu = E / (2. * (1. + nu))
            kappa = E / (3. * (1. - 2. * nu))
            J = np.linalg.det(F)
            I1 = np.trace(F.T @ F)
            energy = (mu / 2.) * (I1 - 3. - 2 * np.log(J)) + (kappa / 2.) * (J - 1.)**2.
            return energy

        P_fn = jax.grad(psi)

        def first_PK_stress(u_grad):
            I = np.eye(u_grad.shape[0])
            F = u_grad + I
            P = P_fn(F)
            return P

        return first_PK_stress
    
    def get_surface_maps(self):

        # traction t is expected to have the correct shape.
        def surface_map(u, u_grad, x, t):
            return t
                
        return {"u": {"top": surface_map}}
    
class SurfaceKernelTest(unittest.TestCase):

    @classmethod
    def setUpClass(cls):

        # create mesh and FE object.
        mesh = box_mesh(Nx=3, Ny=3, Nz=3)
        # use large gauss order to ensure fe.face_nodes != fe.face_quads
        fe = FiniteElement(mesh, vec = 3, dim = 3, ele_type = "hexahedron", gauss_order = 2)

        # define dirichlet boundary conditions
        def bottom(point):
            return np.isclose(point[0], 0., atol=1e-5)
        # Define Dirichlet boundary values.
        def zero_dirichlet_val(point):
            return 0.
        bc1 = [[bottom] * 3, [0, 1, 2],
                [zero_dirichlet_val] * 3]
        dirichlet_bc_info = {"u": [bc1]}

        # define neumann / traction boundary conditions.
        def surf(point):
            return np.isclose(point[0], 1.0, atol = 1e-5)
        location_fns = {'u': {'top': surf}}

        # create problem class instance
        problem = HyperElasticity({"u": fe},
                                dirichlet_bc_info=dirichlet_bc_info,
                                location_fns=location_fns)

        # save the problem object.
        cls.problem = problem

        # define a traction to re-use everywhere
        cls.t = np.array([1.0, 0.0, 0.0])

        return super().setUpClass()
    
    # define a method to standardize the *structure* of the input
    # to problem.set_internal_vars_surfs
    def get_internal_vars_surf_dict(self, traction):

        # a -> 't' in the function signature of surface_map
        return {'u': {'top': {'a': traction}}}

    # standardizes checking for the correct traction shape
    def traction_is_correct_shape(self, fe_key = 'u', surface_map = 'top', var='a'):
        """ checks for the correct traction shape


        Parameters
        ----------
        fe : str, optional
            _description_, by default 'u'
        surface_map : str, optional
            _description_, by default 'top'
        var : str, optional
            _description_, by default 'a'
        """

        # using dictionaries re-orders the shape of inputs to
        # vmapped functions, which is incredibly annoying.
        reshaped_traction = self.problem.internal_vars_surfaces[fe_key][surface_map][var]

        # check that the traction has the correct shape
        traction_shape = reshaped_traction.shape
        expected_shape = (len(self.problem.cells_face_dict[fe_key][surface_map]), 
                          self.problem.fes[fe_key].num_face_quads, 
                          self.problem.fes[fe_key].vec)
                
        return (traction_shape == expected_shape)
    
    def test_single_traction(self):
        """ tests defining traction as a constant field.
        """

        # try setting internal surface variables and calling a specific
        # function in Problem to avoid actually solving a problem.
        #
        # this ^ might require a Mock test, but would be worth it IMO.

        # turn the traction into the appropriate dictionary / structure
        traction_dict = self.get_internal_vars_surf_dict(self.t)

        # set the traction
        self.problem.set_internal_vars_surfaces(traction_dict)

        # check that the shape of the internal variable is correct.
        assert self.traction_is_correct_shape()

    def test_single_traction_nodes(self):
        """ tests defining traction as a constant field,
            on mesh 'nodes'.
        """

        # turn the traction into the appropriate dictionary / structure
        traction_dict = self.get_internal_vars_surf_dict(self.t)

        # set the traction
        self.problem.set_internal_vars_surfaces(traction_dict, 1)

        # check that the shape of the internal variable is correct.
        assert self.traction_is_correct_shape()
    
    def test_traction_cells(self):
        """ tests defining traction that is constant on each cell.
        """

        # traction - constant on each cell on the 'top' patch.
        t_cells = np.tile(self.t, (len(self.problem.cells_face_dict['u']['top']),1))

        # turn the traction into the appropriate dictionary / structure
        traction_dict = self.get_internal_vars_surf_dict(t_cells)

        # set the traction
        self.problem.set_internal_vars_surfaces(traction_dict)

        # check that the shape of the internal variable is correct.
        assert self.traction_is_correct_shape()
    
    def test_traction_cells_quads(self):
        """ tests defining traction at quads on each cell facet.
        """

        # traction defined on each node (value is constant, but shape is not.).
        t_nodes = np.tile(self.t, (len(self.problem.cells_face_dict['u']['top']), 
                              self.problem.fes['u'].num_face_quads, 
                              1))
        
        # NOTE: variables should be:
        #       (num_faces, num_face_quads, a), right?

        # turn the traction into the appropriate dictionary / structure
        traction_dict = self.get_internal_vars_surf_dict(t_nodes)

        # set the traction
        self.problem.set_internal_vars_surfaces(traction_dict)

        # check that the shape of the internal variable is correct.
        assert self.traction_is_correct_shape()
    
    def test_traction_cells_nodes(self):
        """ tests defining traction at nodes on each cell facet.
        """

        # traction defined on each node (value is constant, but shape is not.).
        t_nodes = np.tile(self.t, (len(self.problem.cells_face_dict['u']['top']), 
                              self.problem.fes['u'].num_nodes, 
                              1))
        
        # NOTE: output should be:
        #       (num_faces, num_face_quads, a), right?

        # turn the traction into the appropriate dictionary / structure
        traction_dict = self.get_internal_vars_surf_dict(t_nodes)

        # set the traction
        # 1 -> defines things at NODES.
        self.problem.set_internal_vars_surfaces(traction_dict, 1)

        # check that the shape of the internal variable is correct.
        assert self.traction_is_correct_shape()
    
    def test_traction_all_nodes(self):
        """ tests defining traction at all nodes in the mesh.

        this is useful for problems in contat mechanics, not sure
        how useful this is for other problems.
        """

        # traction defined on each node (value is constant, but shape is not.).
        # TODO: test this with wrapped geometries.
        t_nodes = np.tile(self.t, (self.problem.fes['u'].num_total_nodes, 1))
        
        # NOTE: output should be:
        #       (num_faces, num_face_quads, a), right?

        # turn the traction into the appropriate dictionary / structure
        traction_dict = self.get_internal_vars_surf_dict(t_nodes)

        # set the traction
        # 1 -> defines things at NODES.
        self.problem.set_internal_vars_surfaces(traction_dict, 1)

        # check that the shape of the internal variable is correct.
        assert self.traction_is_correct_shape()

    def test_scalar_val_all_nodes(self):
        """ tests defining a pressure as a scalar field at all nodes
        """

        scalar_value = np.ones((self.problem.fes['u'].num_total_nodes,1))

        # turn the traction into the appropriate dictionary / structure
        scalar_val_dict = self.get_internal_vars_surf_dict(scalar_value)

        # set the traction
        # 1 -> defines things at NODES.
        self.problem.set_internal_vars_surfaces(scalar_val_dict, 1)

        scalar_shape = self.problem.internal_vars_surfaces['u']['top']['a'].shape

        # check that the shape of the internal variable is correct.
        expected_shape = (len(self.problem.cells_face_dict['u']['top']), 
                          self.problem.fes['u'].num_face_quads,
                          1)

        # NOTE: I'm not sure if we're going to support defining
        #       a scalar field at each node of shape (num_total_nodes,); 
        #       this becomes a bit ambiguous as what if we want to set
        #       something with one of the following shapes:
        #
        #       - (num_total_cells, )
        #       - (num_nodes_per_cell, )
        #       - (num_total_quads, )
        #
        #       there is a chance that num_total_cells == num_total_quads,
        #       so this would be impossible to distinguish.
        onptest.assert_(expected_shape == scalar_shape)

if __name__ == "__main__":
    unittest.main()