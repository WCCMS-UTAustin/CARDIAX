import numpy as onp
import jax.numpy as np
import jax
import meshio
from meshio import Mesh
import pyvista as pv
import os
import gmsh
from typing import Optional
from jax.typing import ArrayLike

from cardiax._basis import get_elements

def gmsh_to_meshio(msh_file: Optional[str] = None, **kwargs) -> pv.UnstructuredGrid:
    """Converts gmsh output to meshio format.
    Currently deletes all mesh data, will change to 
    keep gmsh QoIs.

    Args:
        msh_file (Union[str, None]): The output msh file path.

    Returns:
        pv.UnstructuredGrid: Mesh object
    """

    mesh = meshio.read("temp.msh")
    mesh.cell_data = kwargs.get("cell_data", {})
    mesh.point_data = kwargs.get("point_data", {})
    mesh.cell_sets = {}
    main_type = mesh.cells[-1].type
    temp = []
    [temp.append(c) for c in mesh.cells if c.type == main_type]
    if len(temp) > 1:
        combined_data = onp.concatenate([c.data for c in temp])
        temp = [meshio.CellBlock(cell_type=main_type, data=combined_data)]
    mesh.cells = temp
    if isinstance(msh_file, str):
        mesh.write(msh_file)

    os.remove("temp.msh")
    try:
        pvmesh = pv.from_meshio(mesh)
    except:
        raise RuntimeError("pyvista conversion failed. Please fix gmsh_to_meshio conversion" \
        " for your mesh.")

    return pvmesh

def rectangle_mesh(Nx: int = 10, Ny: int = 10, 
                   Lx: float = 1.0, Ly: float = 1.0, 
                   ele_type: str = "quad", 
                   msh_file: Optional[str] = None, 
                   verbose: bool = False) -> pv.UnstructuredGrid:
    """Generate a rectangle mesh.
    
    Args:
        Nx (int, optional): Number of elements in x direction. Defaults to 10
        Ny (int, optional): Number of elements in y direction. Defaults to 10
        Lx (float, optional): Length in x direction. Defaults to 1.
        Ly (float, optional): Length in y direction. Defaults to 1.
        ele_type (str, optional): Type of mesh elements. Defaults to quad
        msh_file (str, optional): Path to save the generated mesh file. Defaults to None
        verbose (bool, optional): Enable/disable gmsh terminal output. Defaults to False
        
    Returns:
        pyvista.UnstructuredGrid: Mesh object
    """
    
    _, _, _, _, degree, _ = get_elements(ele_type)
    gmsh.initialize()
    if not verbose:
        gmsh.option.setNumber("General.Terminal", 0)
    gmsh.option.setNumber("Mesh.MshFileVersion", 2.2)  # save in old MSH format

    # Need to figure out if quad8 or quad9, below is for quad8
    # Configure to get quad8 (serendipity) instead of quad9 (complete)
    if ele_type == "quad8":
        gmsh.option.setNumber("Mesh.SecondOrderIncomplete", 1)

    # Create rectangle geometry
    p1 = gmsh.model.geo.addPoint(0, 0, 0)
    p2 = gmsh.model.geo.addPoint(Lx, 0, 0)
    p3 = gmsh.model.geo.addPoint(Lx, Ly, 0)
    p4 = gmsh.model.geo.addPoint(0, Ly, 0)
    
    l1 = gmsh.model.geo.addLine(p1, p2)
    l2 = gmsh.model.geo.addLine(p2, p3)
    l3 = gmsh.model.geo.addLine(p3, p4)
    l4 = gmsh.model.geo.addLine(p4, p1)
    
    cl = gmsh.model.geo.addCurveLoop([l1, l2, l3, l4])
    s = gmsh.model.geo.addPlaneSurface([cl])
    
    # Set transfinite lines for structured mesh
    gmsh.model.geo.mesh.setTransfiniteCurve(l1, Nx + 1)
    gmsh.model.geo.mesh.setTransfiniteCurve(l2, Ny + 1)
    gmsh.model.geo.mesh.setTransfiniteCurve(l3, Nx + 1)
    gmsh.model.geo.mesh.setTransfiniteCurve(l4, Ny + 1)
    
    # Set transfinite surface and recombine to get quads
    if "quad" in ele_type:
        gmsh.model.geo.mesh.setTransfiniteSurface(s)
        gmsh.model.geo.mesh.setRecombine(2, s)
    
    gmsh.model.geo.synchronize()
    gmsh.model.mesh.generate(2)
    gmsh.model.mesh.setOrder(degree)
    
    gmsh.write("temp.msh")
    gmsh.finalize()

    mesh = meshio.read("temp.msh")

    top_pts = onp.isclose(mesh.points[:, 1], Ly)
    bottom_pts = onp.isclose(mesh.points[:, 1], 0.)
    left_pts = onp.isclose(mesh.points[:, 0], 0.)
    right_pts = onp.isclose(mesh.points[:, 0], Lx)

    point_data = {"top": top_pts.astype(float),
                  "bottom": bottom_pts.astype(float),
                  "left": left_pts.astype(float),
                  "right": right_pts.astype(float)}

    mesh = gmsh_to_meshio(msh_file, point_data=point_data)
    mesh.points = mesh.points[:, :2]  # Drop z-coordinate for 2D mesh
    return mesh

def box_mesh(Nx: int = 10, Ny: int = 10, Nz: int = 10, 
             Lx: float = 1.0, Ly: float = 1.0, Lz: float = 1.0, 
             ele_type: str = 'hexahedron', 
             msh_file: Optional[str] = None, 
             verbose: bool = False) -> pv.UnstructuredGrid:
    """Generate a box mesh.

    Args:
        Nx (int, optional): Number of elements in x direction. Defaults to 10.
        Ny (int, optional): Number of elements in y direction. Defaults to 10.
        Nz (int, optional): Number of elements in z direction. Defaults to 10.
        Lx (float, optional): Length in x direction. Defaults to 1.0.
        Ly (float, optional): Length in y direction. Defaults to 1.0.
        Lz (float, optional): Length in z direction. Defaults to 1.0.
        ele_type (str, optional): Type of mesh elements. Defaults to 'hexahedron'.
        msh_file (str, optional): Path to save the generated mesh file. Defaults to None.
        verbose (bool, optional): Enable/disable gmsh terminal output. Defaults to False.

    Returns:
        pyvista.UnstructuredGrid: Mesh object
    """
    _, _, _, _, degree, _ = get_elements(ele_type)
    
    gmsh.initialize()
    if not verbose:
        gmsh.option.setNumber("General.Terminal", 0)
    gmsh.option.setNumber("Mesh.MshFileVersion", 2.2)
    if ele_type == 'hexahedron20':
        gmsh.option.setNumber("Mesh.SecondOrderIncomplete", 1)

    # Create 4 corner points for the base rectangle
    p1 = gmsh.model.geo.addPoint(0, 0, 0)
    p2 = gmsh.model.geo.addPoint(Lx, 0, 0)
    p3 = gmsh.model.geo.addPoint(Lx, Ly, 0)
    p4 = gmsh.model.geo.addPoint(0, Ly, 0)

    # Create lines for the rectangle
    l1 = gmsh.model.geo.addLine(p1, p2)
    l2 = gmsh.model.geo.addLine(p2, p3)
    l3 = gmsh.model.geo.addLine(p3, p4)
    l4 = gmsh.model.geo.addLine(p4, p1)

    # Create curve loop and surface
    cl = gmsh.model.geo.addCurveLoop([l1, l2, l3, l4])
    s = gmsh.model.geo.addPlaneSurface([cl])

    # Set transfinite lines for structured mesh
    gmsh.model.geo.mesh.setTransfiniteCurve(l1, Nx + 1)
    gmsh.model.geo.mesh.setTransfiniteCurve(l3, Nx + 1)
    gmsh.model.geo.mesh.setTransfiniteCurve(l2, Ny + 1)
    gmsh.model.geo.mesh.setTransfiniteCurve(l4, Ny + 1)

    # Set transfinite surface and recombine to get quads
    gmsh.model.geo.mesh.setTransfiniteSurface(s)
    gmsh.model.geo.mesh.setRecombine(2, s)

    # Extrude the surface in z to create the volume
    extruded = gmsh.model.geo.extrude([(2, s)], 0, 0, Lz, [Nz], [1], recombine=True)
    # The volume entity is the one with dim==3
    vol = next(tag for dim, tag in extruded if dim == 3)

    gmsh.model.geo.synchronize()
    gmsh.model.mesh.generate(3)
    gmsh.model.mesh.setOrder(degree)
    gmsh.write("temp.msh")
    gmsh.finalize()

    mesh = meshio.read("temp.msh")

    top_pts = mesh.points[:, 2] == Lz
    bottom_pts = mesh.points[:, 2] == 0.
    left_pts = mesh.points[:, 0] == 0.
    right_pts = mesh.points[:, 0] == Lx
    front_pts = mesh.points[:, 1] == 0.
    back_pts = mesh.points[:, 1] == Ly

    point_data = {"top": top_pts.astype(float),
                  "bottom": bottom_pts.astype(float),
                  "left": left_pts.astype(float),
                  "right": right_pts.astype(float),
                  "front": front_pts.astype(float),
                  "back": back_pts.astype(float)}

    mesh = gmsh_to_meshio(msh_file, point_data=point_data)
    return mesh

def sphere_mesh(center: onp.ndarray = onp.array([0., 0., 0.]), 
                radius: float = 1.0, 
                degree: int = 1, 
                msh_file: Optional[str] = None, 
                verbose: bool = False) -> pv.UnstructuredGrid:

    """Generate a sphere mesh.

    Args:
        center (onp.ndarray, optional): Center coordinates of the sphere. Defaults to onp.array([0., 0., 0.]).
        radius (float, optional): Radius of the sphere. Defaults to 1.0.
        degree (int, optional): Mesh element order. Defaults to 1.
        msh_file (Optional[str], optional): Path to save the generated mesh file. Defaults to None.
        verbose (bool, optional): Enable/disable gmsh terminal output. Defaults to False.

    Returns:
        pv.UnstructuredGrid: Mesh object
    """

    # Initialize gmsh
    gmsh.initialize()
    if not verbose:
        gmsh.option.setNumber("General.Terminal", 0)

    # Create a new model
    gmsh.model.add("sphere")

    # Create a sphere
    # Parameters: center coordinates (x, y, z), radius
    center_x, center_y, center_z = center

    # Add a sphere volume
    gmsh.model.occ.addSphere(center_x, center_y, center_z, radius)

    # Synchronize the CAD representation with the gmsh model
    gmsh.model.occ.synchronize()

    # Set mesh size
    gmsh.option.setNumber("Mesh.CharacteristicLengthMin", 0.1)
    gmsh.option.setNumber("Mesh.CharacteristicLengthMax", 0.2)
    gmsh.option.setNumber("Mesh.ElementOrder", degree)

    # Generate 3D mesh
    gmsh.model.mesh.generate(3)

    # Write mesh to file
    gmsh.write("temp.msh")
    gmsh.finalize()

    mesh = meshio.read("temp.msh")

    outer_surface = onp.isclose(onp.linalg.norm(mesh.points - center, axis=1), radius)

    point_data = {"outer_surface": outer_surface.astype(float)}
    
    mesh = gmsh_to_meshio(msh_file, point_data=point_data)
    return mesh

def hollow_sphere_mesh(center: onp.ndarray = onp.array([0., 0., 0.]), 
                       outer_radius: float = 1.0, 
                       inner_radius: float = 0.5, 
                       degree: int = 1, 
                       msh_file: Optional[str] = None, 
                       verbose: bool = False) -> pv.UnstructuredGrid:
    """
    Create a hollow sphere by subtracting an inner sphere from an outer sphere.
    
    Args:
        center (onp.ndarray, optional): Center coordinates of the sphere. Defaults to onp.array([0., 0., 0.]).
        outer_radius (float, optional): Radius of the outer sphere. Defaults to 1.0.
        inner_radius (float, optional): Radius of the inner sphere. Defaults to 0.5.
        degree (int, optional): Mesh element order. Defaults to 1.
        msh_file (Optional[str], optional): Path to save the generated mesh file. Defaults to None.
        verbose (bool, optional): Enable/disable gmsh terminal output. Defaults to False.
    
    Returns:
        pyvista.UnstructuredGrid: Mesh object
    """
    
    # Initialize gmsh
    gmsh.initialize()
    if not verbose:
        gmsh.option.setNumber("General.Terminal", 0)

    # Create a new model
    gmsh.model.add("hollow_sphere")

    # Extract center coordinates
    center_x, center_y, center_z = center

    # Add outer sphere volume
    outer_sphere_tag = gmsh.model.occ.addSphere(center_x, center_y, center_z, outer_radius)
    
    # Add inner sphere volume
    inner_sphere_tag = gmsh.model.occ.addSphere(center_x, center_y, center_z, inner_radius)

    # Subtract inner sphere from outer sphere to create hollow sphere
    gmsh.model.occ.cut([(3, outer_sphere_tag)], [(3, inner_sphere_tag)])

    # Synchronize the CAD representation with the gmsh model
    gmsh.model.occ.synchronize()

    # Set mesh size based on the thickness of the shell
    shell_thickness = outer_radius - inner_radius
    char_length_min = shell_thickness / 10.0
    char_length_max = shell_thickness / 5.0
    
    gmsh.option.setNumber("Mesh.CharacteristicLengthMin", char_length_min)
    gmsh.option.setNumber("Mesh.CharacteristicLengthMax", char_length_max)
    gmsh.option.setNumber("Mesh.ElementOrder", degree)

    # Generate 3D mesh
    gmsh.model.mesh.generate(3)

    # Write mesh to file
    gmsh.write("temp.msh")
    gmsh.finalize()

    mesh = meshio.read("temp.msh")

    outer_surface = onp.isclose(onp.linalg.norm(mesh.points - center, axis=1), outer_radius)
    inner_surface = onp.isclose(onp.linalg.norm(mesh.points - center, axis=1), inner_radius)

    point_data = {"outer_surface": outer_surface.astype(float),
                  "inner_surface": inner_surface.astype(float)}

    mesh = gmsh_to_meshio(msh_file, point_data=point_data)
    return mesh

def ellipsoid_mesh(a: float = 1.0, b: float = 0.8, c: float = 0.6, 
                   mesh_size: float = 0.1, 
                   msh_file: Optional[str] = None, 
                   verbose: bool = False) -> pv.UnstructuredGrid:
    """
    Create an ellipsoid mesh.
    
    Args:
        a (float, optional): Semi-axis length in x direction. Defaults to 1.0.
        b (float, optional): Semi-axis length in y direction. Defaults to 0.8.
        c (float, optional): Semi-axis length in z direction. Defaults to 0.6.
        mesh_size (float, optional): Characteristic length for mesh generation. Defaults to 0.1.
        msh_file (Optional[str], optional): Path to save the generated mesh file. Defaults to None.
        verbose (bool, optional): Enable/disable gmsh terminal output. Defaults to False.
    
    Returns:
        pyvista.UnstructuredGrid: Mesh object
    """

    # Initialize gmsh
    gmsh.initialize()
    if not verbose:
        gmsh.option.setNumber("General.Terminal", 0)
    gmsh.model.add("ellipsoid")
    
    # Create ellipsoid geometry
    # gmsh uses a sphere and then scales it to create an ellipsoid
    sphere_tag = gmsh.model.occ.addSphere(0, 0, 0, 1.0)
    
    # Scale the sphere to create ellipsoid with semi-axes a, b, c
    gmsh.model.occ.dilate([(3, sphere_tag)], 0, 0, 0, a, b, c)
    
    # Synchronize to update the geometry
    gmsh.model.occ.synchronize()
    
    # Set mesh size
    gmsh.model.mesh.setSize(gmsh.model.getEntities(0), mesh_size)
    
    # Generate 3D mesh
    gmsh.model.mesh.generate(3)
    
    # Write mesh file
    gmsh.write("temp.msh")
    gmsh.finalize()

    mesh = meshio.read("temp.msh")

    @jax.vmap
    def outer_ellipse(point):
        x, y, z = point
        return np.isclose((x/a)**2 + (y/b)**2 + (z/c)**2, 1.0)

    outer_surface = outer_ellipse(mesh.points)

    point_data = {"outer_surface": outer_surface.astype(float)}

    mesh = gmsh_to_meshio(msh_file, point_data=point_data)
    return mesh

def hollow_ellipsoid_mesh(center: onp.ndarray = onp.array([0., 0., 0.]), 
                          outer_axes: ArrayLike = (1.0, 0.8, 0.6), 
                          inner_axes: ArrayLike = (0.5, 0.4, 0.3), 
                          degree: int = 1, 
                          cl_min: float = 0.05, 
                          cl_max: float = 0.1,
                          msh_file: Optional[str] = None,
                          verbose=False) -> pv.UnstructuredGrid:
    """
    Create a hollow ellipsoid by subtracting an inner ellipsoid from an outer ellipsoid.

    Args:
        center (onp.ndarray, optional): Center coordinates of the ellipsoid. Defaults to onp.array([0., 0., 0.]).
        outer_axes (ArrayLike, optional): Tuple of (a, b, c) semi-axes for outer ellipsoid. Defaults to (1.0, 0.8, 0.6).
        inner_axes (ArrayLike, optional): Tuple of (a, b, c) semi-axes for inner ellipsoid (must be < outer_axes). Defaults to (0.5, 0.4, 0.3).
        degree (int, optional): Mesh element order. Defaults to 1.
        cl_min (float, optional): Minimum characteristic length for mesh generation. Defaults to 0.05.
        cl_max (float, optional): Maximum characteristic length for mesh generation. Defaults to 0.1.
        msh_file (Optional[str], optional): Path to save the generated mesh file. Defaults to None.
        verbose (bool, optional): Enable/disable gmsh terminal output. Defaults to False.

    Returns:
        pyvista.UnstructuredGrid: Mesh object
    """

    gmsh.initialize()
    if not verbose:
        gmsh.option.setNumber("General.Terminal", 0)
    gmsh.model.add("hollow_ellipsoid")

    center_x, center_y, center_z = center

    # Add outer sphere and scale to ellipsoid
    outer_sphere_tag = gmsh.model.occ.addSphere(center_x, center_y, center_z, 1.0)
    gmsh.model.occ.dilate([(3, outer_sphere_tag)], center_x, center_y, center_z, *outer_axes)

    # Add inner sphere and scale to ellipsoid
    inner_sphere_tag = gmsh.model.occ.addSphere(center_x, center_y, center_z, 1.0)
    gmsh.model.occ.dilate([(3, inner_sphere_tag)], center_x, center_y, center_z, *inner_axes)

    # Subtract inner ellipsoid from outer ellipsoid
    gmsh.model.occ.cut([(3, outer_sphere_tag)], [(3, inner_sphere_tag)])

    gmsh.model.occ.synchronize()

    gmsh.option.setNumber("Mesh.CharacteristicLengthMin", cl_min)
    gmsh.option.setNumber("Mesh.CharacteristicLengthMax", cl_max)
    gmsh.option.setNumber("Mesh.ElementOrder", degree)

    gmsh.model.mesh.generate(3)
    gmsh.write("temp.msh")
    gmsh.finalize()

    mesh = meshio.read("temp.msh")
    A, B, C = outer_axes
    a, b, c = inner_axes

    def ellipse(point, a, b, c):
        x, y, z = point
        return np.isclose((x/a)**2 + (y/b)**2 + (z/c)**2, 1.0)

    outer_surface = jax.vmap(ellipse, in_axes=(0, None, None, None))(mesh.points, A, B, C)
    inner_surface = jax.vmap(ellipse, in_axes=(0, None, None, None))(mesh.points, a, b, c)

    point_data = {"outer_surface": outer_surface.astype(float),
                  "inner_surface": inner_surface.astype(float)}

    mesh = gmsh_to_meshio(msh_file, point_data=point_data)
    return mesh

def cylinder_mesh(height: float = 1.0, 
                  radius: float = 0.25, 
                  element_degree: int = 1,
                  cl_min: float = 0.025, 
                  cl_max: float = 0.05, 
                  msh_file: Optional[str] = None, 
                  verbose: bool = False) -> pv.UnstructuredGrid:
    
    """
    Creates a cylinder mesh.
    
    Args:
        height (float, optional): Height of the cylinder. Defaults to 1.0.
        radius (float, optional): Radius of the cylinder. Defaults to 0.25.
        element_degree (int, optional): Mesh element order. Defaults to 1.
        cl_min (float, optional): Minimum characteristic length for mesh generation. Defaults to 0.025.
        cl_max (float, optional): Maximum characteristic length for mesh generation. Defaults to 0.05.
        msh_file (Optional[str], optional): Path to save the generated mesh file. Defaults to None.
        verbose (bool, optional): Enable/disable gmsh terminal output. Defaults to False.
    
    Returns:
        pyvista.UnstructuredGrid: Mesh object
    """

    gmsh.initialize()
    if not verbose:
        gmsh.option.setNumber("General.Terminal", 0)
    gmsh.model.add("cylinder")

    # Cylinder axis: z direction
    axis = [0, 0, height]
    gmsh.model.occ.addCylinder(0, 0, 0, axis[0], axis[1], axis[2], radius, tag=1)
    gmsh.model.occ.synchronize()

    # Mesh options
    gmsh.option.setNumber("Mesh.CharacteristicLengthMin", cl_min)
    gmsh.option.setNumber("Mesh.CharacteristicLengthMax", cl_max)
    gmsh.option.setNumber("Mesh.ElementOrder", element_degree)
    gmsh.model.mesh.generate(3)

    # Save mesh
    gmsh.write("temp.msh")
    gmsh.finalize()

    mesh = meshio.read("temp.msh")

    bottom = mesh.points[:, 2] == 0.
    top = mesh.points[:, 2] == height
    outer_surface = onp.isclose(np.linalg.norm(mesh.points[:, :2], axis=1), radius)

    point_data = {"top": top.astype(float),
                  "bottom": bottom.astype(float),
                  "outer_surface": outer_surface.astype(float)}

    mesh = gmsh_to_meshio(msh_file, point_data=point_data)
    return mesh

def hollow_cylinder_mesh(height: float = 1.0, 
                         outer_radius: float = 0.25, 
                         inner_radius: float = 0.1, 
                         element_degree: int = 1,
                         cl_min: float = 0.1, 
                         cl_max: float = 0.2, 
                         msh_file: Optional[str] = None, 
                         verbose: bool = False) -> pv.UnstructuredGrid:
    
    """
    Creates a hollow cylinder mesh by subtracting an inner cylinder from the outer.
    
    Args:
        height (float, optional): Height of the cylinder. Defaults to 1.0.
        outer_radius (float, optional): Radius of the outer cylinder. Defaults to 0.25.
        inner_radius (float, optional): Radius of the inner cylinder. Defaults to 0.1.
        element_degree (int, optional): Mesh element order. Defaults to 1.
        cl_min (float, optional): Minimum characteristic length for mesh generation. Defaults to 0.025.
        cl_max (float, optional): Maximum characteristic length for mesh generation. Defaults to 0.05.
        msh_file (Optional[str], optional): Path to save the generated mesh file. Defaults to None.
        verbose (bool, optional): Enable/disable gmsh terminal output. Defaults to False.
    
    Returns:
        pyvista.UnstructuredGrid: Mesh object
    """

    gmsh.initialize()
    if not verbose:
        gmsh.option.setNumber("General.Terminal", 0)
    gmsh.model.add("hollow_cylinder")

    # Outer cylinder
    gmsh.model.occ.addCylinder(0, 0, 0, 0, 0, height, outer_radius, tag=1)
    # Inner cylinder (to subtract)
    gmsh.model.occ.addCylinder(0, 0, 0, 0, 0, height, inner_radius, tag=2)
    gmsh.model.occ.cut([(3, 1)], [(3, 2)], tag=3)
    gmsh.model.occ.synchronize()

    gmsh.option.setNumber("Mesh.CharacteristicLengthMin", cl_min)
    gmsh.option.setNumber("Mesh.CharacteristicLengthMax", cl_max)
    gmsh.option.setNumber("Mesh.ElementOrder", element_degree)
    gmsh.model.mesh.generate(3)

    gmsh.write("temp.msh")
    gmsh.finalize()

    mesh = meshio.read("temp.msh")

    bottom = mesh.points[:, 2] == 0.
    top = mesh.points[:, 2] == height
    outer_surface = onp.isclose(np.linalg.norm(mesh.points[:, :2], axis=1), outer_radius)
    inner_surface = onp.isclose(np.linalg.norm(mesh.points[:, :2], axis=1), inner_radius)

    point_data = {"top": top.astype(float),
                  "bottom": bottom.astype(float),
                  "outer_surface": outer_surface.astype(float),
                  "inner_surface": inner_surface.astype(float)}

    mesh = gmsh_to_meshio(msh_file, point_data=point_data)
    return mesh

def prolate_spheroid_mesh(sigma_min: float = 1.35, sigma_max: float = 1.8, 
                          tau_min: float = -1., tau_max: float = 0.,
                          msh_file: Optional[str] = None, 
                          verbose: bool = False,
                          cl_min: float = 0.25,
                          cl_max: float = 0.5) -> pv.UnstructuredGrid:

    """
    Creates a prolate spheroid mesh through prolate spheroidal coordinate system (sigma, tau, phi).
    Sigma must be >= 1
    Tau must be in [-1, 1]
    
    Args:
        sigma_min (float, optional): Minimum value for sigma coordinate. Defaults to 1.35.
        sigma_max (float, optional): Maximum value for sigma coordinate. Defaults to 1.8.
        tau_min (float, optional): Minimum value for tau coordinate. Defaults to -1.
        tau_max (float, optional): Maximum value for tau coordinate. Defaults to 0.
        msh_file (Optional[str], optional): Path to save the generated mesh file. Defaults to None.
        verbose (bool, optional): Enable/disable gmsh terminal output. Defaults to False.
        cl_min (float, optional): Minimum characteristic length for mesh generation. Defaults to 0.25.
        cl_max (float, optional): Maximum characteristic length for mesh generation. Defaults to 0.5.
        verbose (bool, optional): Enable/disable gmsh terminal output. Defaults to False.
    
    Returns:
        pyvista.UnstructuredGrid: Mesh object
    """

    def compute_foci(tau):
        return onp.sqrt(3 * (1 + 5. * tau**2))

    center_pt = onp.array([0.0, 0.0, 0.0])
    axis_rotation = onp.array([0.0, 0.0, -1.0])

    def PS_coords(x):
        sigma, tau, phi = x
        x0 = compute_foci(tau) * onp.sqrt((sigma**2 - 1) * (1 - tau**2)) * onp.cos(phi)
        x1 = compute_foci(tau) * onp.sqrt((sigma**2 - 1) * (1 - tau**2)) * onp.sin(phi)
        x2 = compute_foci(tau) * sigma * tau
        return onp.array([x0, x1, x2])

    gmsh.initialize()
    if not verbose:
        gmsh.option.setNumber("General.Terminal", 0)
    gmsh.option.setNumber("Geometry.CopyMeshingMethod", 1)
    gmsh.model.add("prolate_spheroid")

    apex_endo_pt = PS_coords(onp.array([sigma_min, tau_min, 0.]))
    apex_epi_pt = PS_coords(onp.array([sigma_max, tau_min, 0.]))
    apex_endo = gmsh.model.geo.addPoint(*apex_endo_pt)
    apex_epi = gmsh.model.geo.addPoint(*apex_epi_pt)
    center = gmsh.model.geo.addPoint(*center_pt)
    apex = gmsh.model.geo.addLine(apex_endo, apex_epi)

    for i in range(2):
        base_endo_pt = PS_coords(onp.array([sigma_min, tau_max, i * onp.pi]))
        base_epi_pt = PS_coords(onp.array([sigma_max, tau_max, i * onp.pi]))

        base_endo = gmsh.model.geo.addPoint(*base_endo_pt)
        base_epi = gmsh.model.geo.addPoint(*base_epi_pt)

        base = gmsh.model.geo.addLine(base_endo, base_epi)
        endo = gmsh.model.geo.addEllipseArc(apex_endo, center, apex_endo, base_endo)
        epi = gmsh.model.geo.addEllipseArc(apex_epi, center, apex_epi, base_epi)

        ll1 = gmsh.model.geo.addCurveLoop([apex, epi, -base, -endo])

        s1 = gmsh.model.geo.addPlaneSurface([ll1])
        out = [(2, s1)]

        rev1 = gmsh.model.geo.revolve([out[0]], center_pt[0], center_pt[1], center_pt[2], 
                            axis_rotation[0], axis_rotation[1], axis_rotation[2], onp.pi/2)
        rev2 = gmsh.model.geo.revolve([out[0]], center_pt[0], center_pt[1], center_pt[2], 
                            axis_rotation[0], axis_rotation[1], axis_rotation[2], -onp.pi/2)
        
    gmsh.model.geo.synchronize()

    # Set mesh size
    gmsh.option.setNumber("Mesh.CharacteristicLengthMin", cl_min)
    gmsh.option.setNumber("Mesh.CharacteristicLengthMax", cl_max)
    gmsh.option.setNumber("Mesh.ElementOrder", 1)

    gmsh.model.mesh.generate(3)

    gmsh.write("temp.msh")
    gmsh.finalize()

    mesh = meshio.read("temp.msh")

    epi_idxs = [33, 37, 42, 45]
    endo_idxs = [35, 39, 44, 47]
    base_idxs = [34, 38, 43, 46]
    epi_pts, endo_pts, base_pts = [], [], []
    for i in range(4):
        epi_idx = epi_idxs[i]
        base_idx = base_idxs[i]
        endo_idx = endo_idxs[i]

        epi_pts.append(onp.unique(mesh.cells[epi_idx].data.flatten()))
        base_pts.append(onp.unique(mesh.cells[base_idx].data.flatten()))
        endo_pts.append(onp.unique(mesh.cells[endo_idx].data.flatten()))

    endo_pts = onp.unique(onp.concatenate(endo_pts))
    epi_pts = onp.unique(onp.concatenate(epi_pts))
    base_pts = onp.unique(onp.concatenate(base_pts))
    endo_mask = onp.zeros(mesh.points.shape[0], dtype=bool) 
    epi_mask = onp.zeros(mesh.points.shape[0], dtype=bool)
    base_mask = onp.zeros(mesh.points.shape[0], dtype=bool)
    apex_mask = (mesh.point_data["gmsh:dim_tags"] == [0, apex_epi]).all(axis=1)

    endo_mask[endo_pts] = True
    epi_mask[epi_pts] = True
    base_mask[base_pts] = True

    point_data = {"endo": endo_mask.astype(float),
                  "epi": epi_mask.astype(float),
                  "base": base_mask.astype(float),
                  "apex": apex_mask.astype(float)}

    mesh = gmsh_to_meshio(msh_file, point_data=point_data)
    return mesh

### Plotting meshes for docs ###
import pyvista as pv
import matplotlib.colors as mcolors
import sys

fns = [
    "rectangle_mesh",
    "box_mesh",
    "sphere_mesh",
    "hollow_sphere_mesh",
    "ellipsoid_mesh",
    "hollow_ellipsoid_mesh",
    "cylinder_mesh",
    "hollow_cylinder_mesh",
    "prolate_spheroid_mesh"
]
for fn in fns:
    mesh = getattr(sys.modules[__name__], fn)()

    colors = ['red', "blue", "green", "magenta", "olive", "orange"]
    pl = pv.Plotter(off_screen=True)
    pl.add_mesh(mesh, show_edges=True, color="grey", opacity=0.25)

    legend_entries = []
    for i, p in enumerate(mesh.point_data):
        start_color = 'white'
        end_color = colors[i]
        custom_cmap = mcolors.LinearSegmentedColormap.from_list(
            'temp',
            [start_color, end_color]
        )
        pl.add_mesh(mesh, opacity=p, scalars=mesh.point_data[p],
                    cmap=custom_cmap, label=p, copy_mesh=True)
        legend_entries.append([p, colors[i]])

    pl.remove_scalar_bar()
    pl.add_axes()
    pl.add_legend(legend_entries, face="^", loc='upper right', bcolor='white')
    pl.screenshot(f'../../../docs/figures/meshes/{fn}.png')