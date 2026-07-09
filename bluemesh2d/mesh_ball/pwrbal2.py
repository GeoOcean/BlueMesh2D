import numpy as np

from .inv_2x2 import inv_2x2
from .inv_3x3 import inv_3x3


def pwrbal2(pp, pw, tt):
    """Compute power balls (orthoballs) for a weighted triangulation.

    Parameters
    ----------
    pp : ndarray of shape (N, 2) or (N, 3)
        Vertex coordinates.
    pw : ndarray of shape (N, 1)
        Vertex weights.
    tt : ndarray of shape (T, 3)
        Triangle connectivity.

    Returns
    -------
    bb : ndarray of shape (T, 3) or (T, 4)
        Power-ball parameters ``[xc, yc, r²]`` in 2D or ``[xc, yc, zc, r²]`` in 3D.

    References
    ----------
    Translation of the MESH2D function ``PWRBAL2``.
    Original MATLAB source: https://github.com/dengwirda/mesh2d
    """

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
        bb = np.zeros((tt.shape[0], 3))

        ab = pp[tt[:, 1], :] - pp[tt[:, 0], :]
        ac = pp[tt[:, 2], :] - pp[tt[:, 0], :]

        AA = np.zeros((2, 2, tt.shape[0]))
        AA[0, 0, :] = ab[:, 0] * 2.0
        AA[0, 1, :] = ab[:, 1] * 2.0
        AA[1, 0, :] = ac[:, 0] * 2.0
        AA[1, 1, :] = ac[:, 1] * 2.0

        Rv = np.zeros((2, 1, tt.shape[0]))
        Rv[0, 0, :] = np.sum(ab * ab, axis=1) - (pw[tt[:, 1], 0] - pw[tt[:, 0], 0])
        Rv[1, 0, :] = np.sum(ac * ac, axis=1) - (pw[tt[:, 2], 0] - pw[tt[:, 0], 0])

        II, dd = inv_2x2(AA)

        bb[:, 0] = (II[0, 0, :] * Rv[0, 0, :] + II[0, 1, :] * Rv[1, 0, :]) / dd
        bb[:, 1] = (II[1, 0, :] * Rv[0, 0, :] + II[1, 1, :] * Rv[1, 0, :]) / dd

        bb[:, 0:2] = pp[tt[:, 0], :] + bb[:, 0:2]

        r1 = np.sum((bb[:, 0:2] - pp[tt[:, 0], :]) ** 2, axis=1)
        r2 = np.sum((bb[:, 0:2] - pp[tt[:, 1], :]) ** 2, axis=1)
        r3 = np.sum((bb[:, 0:2] - pp[tt[:, 2], :]) ** 2, axis=1)

        r1 -= pw[tt[:, 0], 0]
        r2 -= pw[tt[:, 1], 0]
        r3 -= pw[tt[:, 2], 0]

        bb[:, 2] = (r1 + r2 + r3) / 3.0

    elif dim == 3:
        bb = np.zeros((tt.shape[0], 4))

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

        Rv = np.zeros((3, 1, tt.shape[0]))
        Rv[0, 0, :] = np.sum(ab * ab, axis=1) - (pw[tt[:, 1], 0] - pw[tt[:, 0], 0])
        Rv[1, 0, :] = np.sum(ac * ac, axis=1) - (pw[tt[:, 2], 0] - pw[tt[:, 0], 0])

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
