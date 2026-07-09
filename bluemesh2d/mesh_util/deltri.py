"""Constrained Delaunay triangulation in the plane.

Uses the optional ``triangle`` package (Shewchuk's Triangle) for true
constrained Delaunay triangulation. When it is not installed, falls back to
the pure-scipy conforming Delaunay in :func:`cfmtri` (boundary edges are
recovered by bisection instead of being constrained directly)."""
import numpy as np

from ..mesh_cost.triarea import triarea
from ..poly_test.inpoly import inpoly
from .cfmtri import cfmtri

try:
    import triangle as tr
except ImportError:
    tr = None

_warned_fallback = False


def deltri(vert=None, conn=None, node=None, PSLG=None, part=None, kind="constrained"):
    """A constrained 2-simplex Delaunay triangulation in the plane.

    Parameters
    ----------
    vert : ndarray of shape (V, 2), optional
        Vertex coordinates to triangulate.
    conn : ndarray of shape (C, 2), optional
        Constraining edges as vertex-index pairs.
    node : ndarray of shape (N, 2), optional
        Polygon vertex coordinates for region filtering.
    PSLG : ndarray of shape (P, 2), optional
        Piecewise straight-line graph: edge endpoint indices into ``node``.
    part : list of ndarray, optional
        For each polygon part, indices into ``PSLG`` defining that region's
        boundary edges.
    kind : {'constrained', 'conforming'}, optional
        ``'constrained'`` uses the ``triangle`` package when available;
        otherwise falls back to ``'conforming'``. ``'conforming'`` always
        uses the bisection-based :func:`cfmtri` algorithm.

    Returns
    -------
    vert : ndarray of shape (V, 2)
        Triangulation vertex coordinates (may include Steiner points).
    conn : ndarray of shape (C, 2)
        Constraining edges.
    tria : ndarray of shape (T, 3)
        Triangle connectivity (CCW-oriented).
    tnum : ndarray of shape (T,), dtype int
        Part index for each triangle; 0 for unclassified or exterior triangles.

    Notes
    -----
    When the optional ``triangle`` package is not installed,
    ``kind='constrained'`` is downgraded to ``'conforming'`` after printing a
    one-time warning.

    References
    ----------
    Translation of the MESH2D function ``DELTRI2``.
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

    if kind == "constrained" and tr is None:
        global _warned_fallback
        if not _warned_fallback:
            print(
                "deltri: 'triangle' package not installed; falling back to "
                "conforming Delaunay (scipy). Install 'triangle' for faster, "
                "truly constrained triangulation."
            )
            _warned_fallback = True
        kind = "conforming"

    if kind == "constrained":
        tri_input = {"vertices": vert, "segments": conn}
        tri_output = tr.triangulate(tri_input, "p")
        vert = tri_output["vertices"]
        tria = tri_output["triangles"]
    elif kind == "conforming":
        vert, conn, tria = cfmtri(vert, conn)
    else:
        raise ValueError(f"deltri: invalid KIND selection '{kind}'")

    tnum = np.zeros(tria.shape[0], dtype=int)
    if node.size and PSLG.size and part:
        tmid = (vert[tria[:, 0], :] + vert[tria[:, 1], :] + vert[tria[:, 2], :]) / 3.0

        for ppos, pedges in enumerate(part, start=1):
            stat, _ = inpoly(tmid, node, PSLG[pedges, :])
            tnum[stat] = ppos

        mask = tnum > 0
        tria = tria[mask, :]
        tnum = tnum[mask]

    area = triarea(vert, tria)
    neg = area < 0.0
    if np.any(neg):
        tria[neg, :] = tria[neg][:, [0, 2, 1]]

    return vert, conn, tria, tnum
