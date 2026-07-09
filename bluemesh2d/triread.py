from .mesh_file.loadmsh import loadmsh


def triread(name):
    """Read a 2D triangulation from a mesh file.

    Parameters
    ----------
    name : str
        Path to the mesh file.

    Returns
    -------
    vert : ndarray of shape (V, 2) or None
        Vertex coordinates.
    edge : ndarray of shape (E, 2) or None
        Constrained edges.
    tria : ndarray of shape (T, 3) or None
        Triangle connectivity.
    tnum : ndarray of shape (T,) or None
        Part index per triangle.

    Notes
    -----
    Uses :func:`~bluemesh2d.mesh_file.loadmsh.loadmsh` (JIGSAW-style format).

    References
    ----------
    Translation of the MESH2D function ``TRIREAD``.
    Original MATLAB source: https://github.com/dengwirda/mesh2d
    """

    import numpy as np

    vert, edge, tria, tnum = None, None, None, None

    if not isinstance(name, str):
        raise TypeError("triread:incorrectInputClass - Incorrect input class.")

    mesh = loadmsh(name)

    if "point" in mesh and "coord" in mesh["point"]:
        vert = np.array(mesh["point"]["coord"])[:, :2]

    if "edge2" in mesh and "index" in mesh["edge2"]:
        edge = np.array(mesh["edge2"]["index"])[:, :2]

    if "tria3" in mesh and "index" in mesh["tria3"]:
        arr = np.array(mesh["tria3"]["index"])
        tria = arr[:, :3]
        tnum = arr[:, 3]

    return vert, edge, tria, tnum
