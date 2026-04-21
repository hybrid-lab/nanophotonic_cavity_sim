import tidy3d as td
import numpy as np


def hole_polygon_2d(geometry, params, n_pts=128):
    """Return an (N, 2) array of xy-vertices for a hole centered at the origin.
 
    This is the single source of truth for hole shapes. Both the 3-D
    Tidy3D geometry (``hole_geometry``) and any 2-D rendering / GDS
    export should derive from the same shape definitions here.
 
    Parameters
    ----------
    geometry : str
        Hole type (must match a case below).
    params : array-like
        Shape parameters whose meaning depends on *geometry*.
    n_pts : int
        Vertex count for curved shapes.
 
    Returns
    -------
    np.ndarray, shape (N, 2)
    """
    hp = np.atleast_1d(params)
 
    if geometry == "square":
        hx, hy = hp[0] / 2, hp[1] / 2
        return np.array([
            [-hx, -hy],
            [ hx, -hy],
            [ hx,  hy],
            [-hx,  hy],
            [-hx, -hy],
        ])

 
    if geometry == "ellipse":
        theta = np.linspace(0, 2 * np.pi, n_pts, endpoint=True)
        rx, ry = hp[0] / 2, hp[1] / 2
        return np.column_stack([rx * np.cos(theta), ry * np.sin(theta)])
 
    raise ValueError(f"Unknown geometry: '{geometry}'")
 
 
def hole_geometry(geometry, hole_center, params, thickness):
    """Create a single 3-D Tidy3D hole geometry at the given center.
 
    Uses ``hole_polygon_2d`` for all shapes to keep the geometry
    definition in one place.
    """
    verts = hole_polygon_2d(geometry, params)
    verts_shifted = verts + np.array([hole_center[0], hole_center[1]])
 
    return td.PolySlab(
        vertices=verts_shifted,
        slab_bounds=(
            -thickness / 2 + hole_center[2],
             thickness / 2 + hole_center[2],
        ),
        axis=2,
    )