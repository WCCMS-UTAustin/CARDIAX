"""
This script is made to run an input file.
Call (python run.py input_file.yaml)
"""

import functools as fctls
from cardiax.input_file.config import ProblemConfig
import yaml
import pyvista as pv
from pathlib import Path
import importlib
import jax.numpy as np
import numpy as onp
import jax
from jaxtyping import ArrayLike
from .config import FEConfig, get_dict
from dataclasses import asdict

import cardiax
from cardiax import FiniteElement, Problem, Newton_Solver
from cardiax._solver import Solver_Base
from cardiax.input_file.input_file_helper import add_surface_kernels

class ProblemManager():
    """This class is responsible for loading the input file and generating the 
    appropriate classes: FiniteElement, Problem, and Solver, to then be used to 
    solve the PDE in a reproducible manner. When setup appropriately,
    it should make creation of a workflow much easier for the user.

    Args:
        degree (_type_): _description_

    Returns:
        _type_: _description_
    """

    def __init__(self, ProblemConfig: ProblemConfig):
        """Add documentation

        Args:
            degree (_type_): _description_

        Returns:
            _type_: _description_
        """

        self.config = ProblemConfig

        # process the input file; generate FE, Problem, and Solver classes!
        self.fes, self.internal_vars = self.fe_loader(config=self.config.fe_config)
        self.problem = self.problem_loader(config=self.config.pde_config)
        self.solver = self.solver_loader(config=self.config.solver_config)
        return

    @classmethod
    def from_yaml(cls, input_file: Path):
        return cls(ProblemConfig.from_yaml(input_file))

    def fe_loader(self, config: FEConfig) -> tuple[FiniteElement, dict]:
        """ loads FE object from provided mesh data.

        Raises
        ------
        ValueError
            _description_
        """

        # read in mesh as .vtk file
        if config.mesh_path is not None:
            mesh = pv.read((self.config.directory / config.mesh_path).resolve())
        else:
            meshgen = getattr(cardiax, config.mesh_generator["name"])
            mesh = meshgen(**config.mesh_generator["kwargs"])

        # create and save FiniteElement object
        fes = {config.var: FiniteElement(mesh, **config.kwargs)}
        internal_vars = {config.var: {}}
        return fes, internal_vars

    def dirichlet_bc_loader(self, dirichlet_bc_params: dict):
        """Add documentation

        Args:
            degree (_type_): _description_

        Returns:
            _type_: _description_
        """

        # Generic function to check if a point is on a surface
        def surf_tag(tagged_nodes, point):
            return np.isclose(point, tagged_nodes, atol=1e-6).all(axis=1).any()

        def value_fn(value, point):
            return np.array(value)

        dirichlet_bc_info = {}

        for fe_key in dirichlet_bc_params:
            dirichlet_bc_temp = []
            for bc_num in dirichlet_bc_params[fe_key]:
                bc = dirichlet_bc_params[fe_key][bc_num]
                assert len(bc["component"]) == len(bc["value"])
                if bc["surface_tag"] is not None:
                    tagged_nodes = self.get_tagged_nodes(bc, fe_key)
                else:
                    tagged_nodes = self.create_boundary_mask(bc["surface_fn"]["point_comp"], bc["surface_fn"]["where"], fe_key)
                dirichlet_tag = fctls.partial(surf_tag, tagged_nodes)

                value_fns = []
                for i in range(len(bc["value"])):
                    part_value_fn = fctls.partial(value_fn, bc["value"][i])
                    value_fns.append(part_value_fn)

                bc_temp = [[dirichlet_tag]*len(bc["value"]), bc["component"], value_fns]
                dirichlet_bc_temp.append(bc_temp)

            dirichlet_bc_info[fe_key] = dirichlet_bc_temp
        
        return dirichlet_bc_info

    def surface_maps_loader(self, surface_maps_params: dict):
        """Add documentation

        Args:
            degree (_type_): _description_

        Returns:
            _type_: _description_
        """

        # Generic function to check if a point is on a surface
        def surf_tag(tagged_nodes, point):
            return np.isclose(point, tagged_nodes, atol=1e-6).all(axis=1).any()

        def value_fn(value, point):
            return np.array(value)

        location_fns = {}
        surface_kernels = {}
        internal_vars_surfaces = {}
        for fe_key in surface_maps_params:
            location_fns_temp = {}
            surface_kernels_temp = {}
            int_vars_surf_temp = {}
            for bc_key in surface_maps_params[fe_key]:
                bc = surface_maps_params[fe_key][bc_key]
                if bc is None:
                    break

                if bc["type"] == "Neumann":
                    assert len(bc["value"]) == self.fes[fe_key].vec
                    if bc["surface_tag"] is not None:
                        tagged_nodes = self.get_tagged_nodes(bc, fe_key)
                    else:
                        tagged_nodes = self.create_boundary_mask(bc["surface_fn"]["point_comp"], bc["surface_fn"]["where"], fe_key)

                    neumann_tag = fctls.partial(surf_tag, tagged_nodes)

                    location_fns_temp[bc_key] = neumann_tag
                    surface_kernels_temp[bc_key] = {"type": "Neumann", "value": bc["value"], "static": bc["static"]}

                    if not bc["static"]:
                        #TODO: Make easier way to get the shape for surface kernels internal vars
                        normals = self.fes[fe_key].get_surface_normals(neumann_tag)
                        surface_var = np.full_like(normals, np.array(bc["value"]))
                        int_vars_surf_temp[bc_key] = {"t": surface_var}
                    else:
                        int_vars_surf_temp[bc_key] = {}
                    continue

                elif bc["type"] == "Robin":
                    raise ValueError("Robin BC not implemented yet.")
                    continue
                elif bc["type"] == "Spring":
                    raise ValueError("Spring BC not implemented yet.")
                    continue
                elif bc["type"] == "Pressure":
                    assert isinstance(bc["value"], float)

                    tagged_nodes = self.get_tagged_nodes(bc, fe_key)

                    neumann_tag = fctls.partial(surf_tag, tagged_nodes)
                    normals = self.fes[fe_key].get_surface_normals(neumann_tag)
                    surface_var = np.full_like(normals, bc["value"])[:, :, :1]

                    location_fns_temp[bc_key] = neumann_tag
                    surface_kernels_temp[bc_key] = {"type": "Pressure", "value": bc["value"], "static": bc["static"]}

                    if not bc["static"]:
                        #TODO: Make easier way to get the shape for surface kernels internal vars
                        int_vars_surf_temp[bc_key] = {bc["var"][0]: normals, bc["var"][1]: surface_var}
                    else:
                        int_vars_surf_temp[bc_key] = {bc["var"][0]: normals}
                    continue

                else:
                    raise ValueError("BC type not supported. Please choose between Dirichlet, Neumann, Robin, Spring, and Pressure.")

            location_fns[fe_key] = location_fns_temp
            surface_kernels[fe_key] = surface_kernels_temp
            internal_vars_surfaces[fe_key] = int_vars_surf_temp

        return location_fns, surface_kernels, internal_vars_surfaces

    def problem_loader(self, config: ProblemConfig):
        """Add documentation

        Args:
            degree (_type_): _description_

        Returns:
            _type_: _description_
        """

        # Load predefined PDEs
        base_path = Path(cardiax.__file__).parent / "PDEs"
        pde_dirs = {p.parent.name for p in base_path.glob("**/*.py")}

        # Check if custom path given
        if config.custom_path is not None:
            custom_path = config.custom_path
            module = importlib.import_module(custom_path.replace("/", "."))
            Problem_class = getattr(module, "PDE")

            pde_constants = get_dict(config.pde_info, "constants")

        # Check if PDE is in predefined PDEs
        elif config.pde_info["pde_class"] in pde_dirs:
            pde_path = list(base_path.glob(f"**/{config.pde_info['pde_class']}"))[0]
            pdes = {p.name[:-3] for p in pde_path.glob("*.py")}
            assert config.pde_info["pde_type"] in pdes, f"PDE type not supported. Please choose between {pdes}"
            mm = f"{pde_path}/{config.pde_info['pde_type']}"
            cardiax_ind = mm.find("cardiax")
            module = importlib.import_module(mm[cardiax_ind:].replace("/", "."))
            Problem_class = getattr(module, "PDE")

            pde_constants = get_dict(config.pde_info, "constants")

        else:
            raise ValueError("PDE type not supported. Please choose between", pde_dirs)

        # Create dirichlet_bc_info and surface_maps dictionaries
        self.dirichlet_bc_info = self.dirichlet_bc_loader(dirichlet_bc_params=config.dirichlet_bc_info)
        self.location_fns, self.surface_kernels, self.internal_vars_surfaces = self.surface_maps_loader(surface_maps_params=config.surface_maps_info)

        Problem_class = add_surface_kernels(Problem_class, self.surface_kernels)

        problem = Problem_class(self.fes, dirichlet_bc_info=self.dirichlet_bc_info, location_fns=self.location_fns)

        try:
            internal_vars_temp = {}
            ivs = config.pde_info["internal_vars"]
            for fe_key in ivs:
                int_vars_temp = {}
                for iv in ivs[fe_key]:
                    if ivs[fe_key][iv]["value"] == "mesh":
                        data = self.fes[fe_key].mesh.cell_data[iv].astype(np.float32)
                        data = onp.repeat(data[:, None, :], problem.fes[fe_key].num_quads, axis=1)
                        int_vars_temp[iv] = data
                    elif isinstance(ivs[fe_key][iv]["value"], float):
                        data = onp.full((problem.fes[fe_key].cells.shape[0], problem.fes[fe_key].num_quads, ivs[fe_key][iv]["vec"]), ivs[fe_key][iv]["value"]).astype(np.float32)
                        int_vars_temp[iv] = data
                    else:
                        raise ValueError("Internal variable value must be 'mesh' or a float.")
                internal_vars_temp[fe_key] = int_vars_temp
            self.internal_vars = internal_vars_temp
        except: pass #TODO Find error here

        problem.set_params(pde_constants)
        problem.set_internal_vars(self.internal_vars)
        problem.set_internal_vars_surfaces(self.internal_vars_surfaces)
        return problem
    
    def solver_loader(self, config) -> Solver_Base:
        """Add documentation

        Args:
            degree (_type_): _description_

        Returns:
            _type_: _description_
        """

        if config.name == "Newton":
            solver = Newton_Solver(self.problem, np.zeros(self.problem.num_total_dofs_all_vars))
        self.solver_kwargs = config.kwargs
        return solver

    ### May want to do plotting as a separate library
    # def plotting(self, sol):
    #     """Add documentation

    #     Args:
    #         degree (_type_): _description_

    #     Returns:
    #         _type_: _description_
    #     """
        
    #     mesh = self.fes[self.fe_params["var"]].mesh
    #     vec = self.fes[self.fe_params["var"]].vec
    #     mesh.point_data["sol"] = sol.reshape(-1, vec)

    #     if vec == 1:
    #         warped = mesh.warp_by_scalar("sol", factor=1)
    #     else:
    #         warped = mesh.warp_by_vector("sol", factor=1)

    #     pl = pv.Plotter(off_screen=True)
    #     pl.add_mesh(warped, scalars="sol", show_edges=True)
    #     pl.show(screenshot=str(self.parent / self.plot_params["filename"]))

    #     return

    ############ callable functions to solve a problem ##############
    def solve_problem(self, solver_params: dict | None = None) -> tuple[ArrayLike, dict]:
        """ solves problem specified from input file

        Solves a problem to completion (default behavior). Automatically chooses the 
        appropriate load stepping scheme, if load stepping is present.

        Returns
        -------
        sol, info
            solution to problem at nodes/DOFs and information regarding the problem's convergence 

        """
        if solver_params is not None:
            self.solver_kwargs = solver_params
        sol, info = self.solver.solve(**self.solver_kwargs)

        return sol, info

    def get_tagged_nodes(self, bc, var):
        """ gets nodes subjected to a boundary condition

        bc is a dictionary describing the type, value, and location
        of a boundary condition.

        Parameters
        ----------
        bc : dict
            dictionary of boundary condition information.
        """

        tagged_nodes = self.fes[var].mesh.points[self.fes[var].mesh.point_data[bc["surface_tag"]].astype(bool)].astype(np.float64)
        
        return tagged_nodes

    def create_boundary_mask(self, point_ind, where, fe_var):

        def mask_fn(point):
            return np.isclose(point[point_ind], where, atol=1e-6)

        tagged_nodes = jax.vmap(mask_fn)(self.fes[fe_var].mesh.points.astype(np.float64))

        return self.fes[fe_var].mesh.points[tagged_nodes].astype(np.float64)
    
    def dump_config(self, save_dir: Path, filename: str):
        """Dumps this specific manager's configuration to a YAML file."""
        save_dir.mkdir(parents=True, exist_ok=True)
        file_path = save_dir / filename
        
        with open(file_path, "w") as f:
            yaml.safe_dump(asdict(self.config), f, sort_keys=False)
