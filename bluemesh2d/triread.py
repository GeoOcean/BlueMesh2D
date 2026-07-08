from .mesh_file.loadmsh import loadmsh


def triread(name):
    """
    Read two-dimensional triangulation data from a file.

    This function loads a 2-simplex triangulation {VERT, TRIA} from
    the specified mesh file.

    Parameters
    ----------
    name : str
        Name of the mesh file to read.

    Returns
    -------
    vert : ndarray of shape (V, 2)
        XY coordinates of the triangulation vertices.
    edge : ndarray of shape (E, 2)
        Array of constrained edges.
    tria : ndarray of shape (T, 3)
        Array of triangles (vertex indices).
    tnum : ndarray of shape (T, 1)
        Array of part indices, such that `tnum[ii]` gives the index
        of the part containing the ii-th triangle.

    Notes
    -----
    - Data is returned as non-empty arrays if available in the file.
    - Each row of `tria` and `edge` defines an element:
      `vert[tria[ii, 0], :]`, `vert[tria[ii, 1], :]`, and
      `vert[tria[ii, 2], :]` are the coordinates of the ii-th triangle.
      The edges in `edge` are defined in a similar way.
    - This routine borrows functionality from the JIGSAW package:
      https://github.com/dengwirda/jigsaw-matlab

    References
    ----------
    Translation of the MESH2D function `TRIREAD` by Darren Engwirda.
    Original MATLAB source: https://github.com/dengwirda/mesh2d
    """

    import numpy as np

    vert, edge, tria, tnum = None, None, None, None

    # -------------------------------------------- basic checks
    if not isinstance(name, str):
        raise TypeError("triread:incorrectInputClass - Incorrect input class.")

    # ----------------------------------- borrow JIGSAW I/O func!
    mesh = loadmsh(name)

    # ----------------------------------- extract data if present
    if "point" in mesh and "coord" in mesh["point"]:
        vert = np.array(mesh["point"]["coord"])[:, :2]

    if "edge2" in mesh and "index" in mesh["edge2"]:
        edge = np.array(mesh["edge2"]["index"])[:, :2]

    if "tria3" in mesh and "index" in mesh["tria3"]:
        arr = np.array(mesh["tria3"]["index"])
        tria = arr[:, :3]
        tnum = arr[:, 3]

    return vert, edge, tria, tnum
