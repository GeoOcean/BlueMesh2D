import numpy as np

from ..mesh_cost.triarea import triarea
from ..poly_test.inpoly import inpoly
from .cfmtri import cfmtri

# The optional `triangle` package (Shewchuk's Triangle) provides true
# *constrained* Delaunay triangulation. When it is not installed we fall back
# to the pure-scipy *conforming* Delaunay in `cfmtri` (boundary edges are
# recovered by bisection instead of being constrained directly).
try:
    import triangle as tr
except ImportError:
    tr = None

_warned_fallback = False


def deltri(vert=None, conn=None, node=None, PSLG=None, part=None, kind="constrained"):
    """
    DELTRI : compute a constrained 2-simplex Delaunay triangulation in the 2D plane.

    [vert, conn, tria, tnum] = deltri2(vert, conn, node, pslg, part)
    computes the Delaunay triangulation {vert, tria}, the constraining edges `conn`,
    and the "inside" status vector `tnum`.

    Parameters
    ----------
    vert : ndarray (V, 2)
        Array of XY coordinates to be triangulated.
    conn : ndarray (C, 2)
        Array of constraining edges, where each row defines an edge between vertices.
    node : ndarray (N, 2)
        Array of polygon vertices.
    pslg : ndarray (P, 2)
        Piecewise straight-line graph (PSLG) defining the polygon edges as pairs
        of indices into `node`.
    part : list of ndarray
        List of polygonal parts, where each element `part[k]` contains edge indices
        defining a polygonal region. `pslg[part[k], :]` corresponds to the edges
        of the k-th region.
    tria : ndarray (T, 3)
        Array of vertex indices defining the triangles. Each row corresponds to one
        triangle, such that:
        `vert[tria[ii, 0], :]`, `vert[tria[ii, 1], :]`, and `vert[tria[ii, 2], :]`
        are the coordinates of the ii-th triangle.
    tnum : ndarray (T,)
        Part index for each triangle, where `tnum[ii]` gives the index of the part
        that contains the ii-th triangle.

    See Also
    --------
    delaunayTriangulation : MATLAB equivalent triangulation function.
    delaunaytri : legacy Delaunay triangulation.
    delaunayn : N-dimensional Delaunay triangulation.

    References
    ----------
    Translation of the MESH2D function `DELTRI2`.
    Original MATLAB source: https://github.com/dengwirda/mesh2d
    """

    if vert is None:
        vert = np.empty((0, 2))
    if conn is None:
        conn = np.empty((0, 2), dtype=int)
    if node is None:
        node = np.empty((0, 2))
    if PSLG is None:
        PSLG = np.empty((0, 2), dtype=int)
    if part is None:
        part = []

    vert = np.asarray(vert, float)
    conn = np.asarray(conn, int)
    node = np.asarray(node, float)
    PSLG = np.asarray(PSLG, int)
    kind = kind.lower()

    # -------------------------------- basic checks
    nvrt = vert.shape[0]
    if conn.size and (conn.min() < 0 or conn.max() >= nvrt):
        raise ValueError("deltri:invalidInputs (invalid CONN indices)")

    if node.size:
        nnod = node.shape[0]
        if PSLG.size and (PSLG.min() < 0 or PSLG.max() >= nnod):
            raise ValueError("deltri:invalidInputs (invalid PSLG indices)")
        for p in part:
            if np.min(p) < 0 or np.max(p) >= PSLG.shape[0]:
                raise ValueError("deltri:invalidInputs (invalid PART indices)")

    # -------------------------------- compute Delaunay triangulation
    # Match MATLAB's deltri2 behavior exactly:
    # - 'constrained': use delaunayTriangulation (triangle library in Python)
    # - 'conforming': use cfmtri (bisection algorithm)
    if kind == "constrained" and tr is None:
        # `triangle` not installed -> conforming Delaunay (scipy) fallback.
        global _warned_fallback
        if not _warned_fallback:
            print("deltri: 'triangle' package not installed; falling back to "
                  "conforming Delaunay (scipy). Install 'triangle' for faster, "
                  "truly constrained triangulation.")
            _warned_fallback = True
        kind = "conforming"

    if kind == "constrained":
        tri_input = {
            'vertices': vert,
            'segments': conn
        }
        # 'p' = triangulate PSLG (planar straight line graph)
        tri_output = tr.triangulate(tri_input, 'p')
        vert = tri_output['vertices']
        tria = tri_output['triangles']
    elif kind == "conforming":
        # "conforming" Delaunay - use cfmtri (bisection algorithm)
        vert, conn, tria = cfmtri(vert, conn)
    
    else:
        raise ValueError(f"deltri: invalid KIND selection '{kind}'")

    # -------------------------------- compute "inside" status
    tnum = np.zeros(tria.shape[0], dtype=int)
    if node.size and PSLG.size and part:
        tmid = (vert[tria[:, 0], :] + vert[tria[:, 1], :] + vert[tria[:, 2], :]) / 3.0

        for ppos, pedges in enumerate(part, start=1):
            stat, _ = inpoly(tmid, node, PSLG[pedges, :])
            tnum[stat] = ppos

        # Keep only interior triangles
        mask = tnum > 0
        tria = tria[mask, :]
        tnum = tnum[mask]

    # -------------------------------- flip for correct orientation
    area = triarea(vert, tria)
    neg = area < 0.0
    if np.any(neg):
        tria[neg, :] = tria[neg][:, [0, 2, 1]]

    return vert, conn, tria, tnum
