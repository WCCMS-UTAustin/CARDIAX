
from dataclasses import dataclass
from pathlib import Path
import yaml
from .utils import get_dict, get_Path

@dataclass(frozen=True)
class FEConfig:
    
    var: str
    mesh_path : str
    mesh_generator: dict
    kwargs: dict

    def __post_init__(self):

        return

    @classmethod
    def from_dict(cls, params):

        # Check mesh info
        mesh_generator = params.get("mesh_generator", None)
        mesh_path = params.get("mesh_path", None)

        # Check that a mesh is being used or generated
        if mesh_path is None and mesh_generator["name"] is None:
            raise ValueError("Must provide either a mesh path or a mesh generator.")
        
        # Pull info if generating
        if mesh_path is None:
            #TODO: Add assertions for mesh generator name and kwargs
            mesh_gen = mesh_generator["name"]
            mesh_kwargs = get_dict(params["mesh_generator"], "kwargs")
            mesh_generator = {"name": mesh_gen, "kwargs": mesh_kwargs}
        else:
            mesh_generator = None
            mesh_kwargs = {}

        # Check kwargs for FE
        config_kwargs = params["kwargs"]
        # TODO: Add assertions for these kwargs

        return cls(
            var = params["var"],
            mesh_path = mesh_path,
            mesh_generator = mesh_generator,
            kwargs = config_kwargs
        )

@dataclass(frozen=True)
class PDEConfig:
    
    pde_info : dict
    dirichlet_bc_info : dict
    surface_maps_info: dict
    custom_path: Path = None

    @classmethod
    def from_dict(cls, params):

        # Unpack nested dicts
        #TODO: Add assertions for these dicts
        pde_kwargs = params["pde_info"]
        dirichlet_bc_info = get_dict(params, "dirichlet_bc_info")
        surface_maps_info = get_dict(params, "surface_maps_info")

        return cls(
            pde_info = pde_kwargs,
            dirichlet_bc_info = dirichlet_bc_info,
            surface_maps_info = surface_maps_info
        )

@dataclass(frozen=True)
class SolverConfig:

    name: str
    kwargs: dict

    @classmethod
    def from_dict(cls, params):

        # Unpack nested dicts
        #TODO: Add assertions for these dicts
        solver_name = params["name"]
        solver_kwargs = params["kwargs"]

        return cls(
            name = solver_name,
            kwargs = solver_kwargs
        )

@dataclass(frozen=True)
class ProblemConfig:

    fe_config: FEConfig
    pde_config: PDEConfig
    solver_config: SolverConfig
    directory: Path = None

    @classmethod
    def from_dict(cls, params):

        fe_config = FEConfig.from_dict(params["fe_config"])
        pde_config = PDEConfig.from_dict(params["pde_config"])
        solver_config = SolverConfig.from_dict(params["solver_config"])
        directory = get_Path(params, "directory")

        return cls(fe_config=fe_config, 
                   pde_config=pde_config, 
                   solver_config=solver_config, 
                   directory=directory)

    @classmethod
    def from_yaml(cls, path):
        with open(path) as f:
            params = yaml.safe_load(f)

        return cls.from_dict(params)