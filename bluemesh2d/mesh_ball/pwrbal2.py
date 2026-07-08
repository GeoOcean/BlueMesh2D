import numpy as np

from .inv_2x2 import inv_2x2
from .inv_3x3 import inv_3x3


def pwrbal2(pp, pw, tt):
    """
    Compute the orthoballs (power balls) associated with a 2-simplex triangulation
    embedded in R² or R³.

    This function calculates the *power balls* for the set of 2-simplexes (triangles)
    in a triangulation. Each power ball represents the weighted circumball for the
    triangle, incorporating vertex weights.

    Parameters
    ----------
    PP : ndarray of shape (N, D)
        Coordinates of the vertices in the triangulation, where `D` = 2 or 3.
    PW : ndarray of shape (N,)
        Vector of vertex weights.
    TT : ndarray of shape (T, 3)
        Array of triangle vertex indices defining the 2-simplexes.

    Returns
    -------
    BB : ndarray of shape (T, 3)
        Power balls associated with each triangle, where each row is `[XC, YC, RC²]`
        — the center coordinates and squared radius of the orthoball.

    Notes
    -----
    - The orthoball (power ball) generalizes the circumball to weighted Delaunay
      triangulations (power diagrams).
    - Useful for constructing weighted Delaunay meshes and power diagrams in 2D or 3D.

    References
    ----------
    Translation of the MESH2D function `PWRBAL2`.
    Original MATLAB source: https://github.com/dengwirda/mesh2d
    """

    # ---------------------------------------------- basic checks
    if not (
        isinstance(pp, np.ndarray)
        and isinstance(pw, np.ndarray)
        and isinstance(tt, np.ndarray)
    ):
        raise TypeError("pwrbal2:incorrectInputClass")

    if pp.ndim != 2 or pw.ndim != 2 or tt.ndim != 2:
        raise ValueError("pwrbal2:incorrectDimensions")

    if pp.shape[0] != pw.shape[0] or tt.shape[1] < 3 or pp.shape[1] < 2:
        raise ValueError("pwrbal2:incorrectDimensions")

    dim = pp.shape[1]

    if dim == 2:
        # ------------------------------------------- alloc work
        bb = np.zeros((tt.shape[0], 3))

        # -------------------------------------------- lhs matrix
        ab = pp[tt[:, 1], :] - pp[tt[:, 0], :]
        ac = pp[tt[:, 2], :] - pp[tt[:, 0], :]

        AA = np.zeros((2, 2, tt.shape[0]))
        AA[0, 0, :] = ab[:, 0] * 2.0
        AA[0, 1, :] = ab[:, 1] * 2.0
        AA[1, 0, :] = ac[:, 0] * 2.0
        AA[1, 1, :] = ac[:, 1] * 2.0

        # ------------------------------------------- rhs vector
        Rv = np.zeros((2, 1, tt.shape[0]))
        Rv[0, 0, :] = np.sum(ab * ab, axis=1) - (pw[tt[:, 1], 0] - pw[tt[:, 0], 0])
        Rv[1, 0, :] = np.sum(ac * ac, axis=1) - (pw[tt[:, 2], 0] - pw[tt[:, 0], 0])

        # -------------------------------------------- solve sys.
        II, dd = inv_2x2(AA)

        bb[:, 0] = (II[0, 0, :] * Rv[0, 0, :] + II[0, 1, :] * Rv[1, 0, :]) / dd
        bb[:, 1] = (II[1, 0, :] * Rv[0, 0, :] + II[1, 1, :] * Rv[1, 0, :]) / dd

        bb[:, 0:2] = pp[tt[:, 0], :] + bb[:, 0:2]

        # ------------------------------------------- mean radii
        r1 = np.sum((bb[:, 0:2] - pp[tt[:, 0], :]) ** 2, axis=1)
        r2 = np.sum((bb[:, 0:2] - pp[tt[:, 1], :]) ** 2, axis=1)
        r3 = np.sum((bb[:, 0:2] - pp[tt[:, 2], :]) ** 2, axis=1)

        r1 -= pw[tt[:, 0], 0]
        r2 -= pw[tt[:, 1], 0]
        r3 -= pw[tt[:, 2], 0]

        bb[:, 2] = (r1 + r2 + r3) / 3.0

    elif dim == 3:
        # ------------------------------------------- alloc work
        bb = np.zeros((tt.shape[0], 4))

        # -------------------------------------------- lhs matrix
        ab = pp[tt[:, 1], :] - pp[tt[:, 0], :]
        ac = pp[tt[:, 2], :] - pp[tt[:, 0], :]

        AA = np.zeros((3, 3, tt.shape[0]))
        AA[0, 0, :] = ab[:, 0] * 2.0
        AA[0, 1, :] = ab[:, 1] * 2.0
        AA[0, 2, :] = ab[:, 2] * 2.0
        AA[1, 0, :] = ac[:, 0] * 2.0
        AA[1, 1, :] = ac[:, 1] * 2.0
        AA[1, 2, :] = ac[:, 2] * 2.0

        nv = np.cross(ab, ac)
        AA[2, 0, :] = nv[:, 0]
        AA[2, 1, :] = nv[:, 1]
        AA[2, 2, :] = nv[:, 2]

        # -------------------------------------------- rhs vector
        Rv = np.zeros((3, 1, tt.shape[0]))
        Rv[0, 0, :] = np.sum(ab * ab, axis=1) - (pw[tt[:, 1], 0] - pw[tt[:, 0], 0])
        Rv[1, 0, :] = np.sum(ac * ac, axis=1) - (pw[tt[:, 2], 0] - pw[tt[:, 0], 0])

        # ------------------------------------------- solve sys.
        II, dd = inv_3x3(AA)

        bb[:, 0] = (
            II[0, 0, :] * Rv[0, 0, :]
            + II[0, 1, :] * Rv[1, 0, :]
            + II[0, 2, :] * Rv[2, 0, :]
        ) / dd
        bb[:, 1] = (
            II[1, 0, :] * Rv[0, 0, :]
            + II[1, 1, :] * Rv[1, 0, :]
            + II[1, 2, :] * Rv[2, 0, :]
        ) / dd
        bb[:, 2] = (
            II[2, 0, :] * Rv[0, 0, :]
            + II[2, 1, :] * Rv[1, 0, :]
            + II[2, 2, :] * Rv[2, 0, :]
        ) / dd

        bb[:, 0:3] = pp[tt[:, 0], :] + bb[:, 0:3]

        # -------------------------------------------- mean radii
        r1 = np.sum((bb[:, 0:3] - pp[tt[:, 0], :]) ** 2, axis=1)
        r2 = np.sum((bb[:, 0:3] - pp[tt[:, 1], :]) ** 2, axis=1)
        r3 = np.sum((bb[:, 0:3] - pp[tt[:, 2], :]) ** 2, axis=1)

        r1 -= pw[tt[:, 0], 0]
        r2 -= pw[tt[:, 1], 0]
        r3 -= pw[tt[:, 2], 0]

        bb[:, 3] = (r1 + r2 + r3) / 3.0

    else:
        raise ValueError("pwrbal2:unsupportedDimension")

    return bb
