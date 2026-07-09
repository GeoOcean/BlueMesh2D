import numpy as np


def inv_2x2(AA):
    """Compute determinant-scaled inverses for a block of 2×2 matrices.

    Returns ``det(A) * inv(A)`` for numerical robustness. Divide by ``DA``
    when solving linear systems.

    Parameters
    ----------
    AA : ndarray of shape (2, 2, N)
        Stack of ``N`` individual 2×2 matrices.

    Returns
    -------
    II : ndarray of shape (2, 2, N)
        Determinant-scaled inverse of each matrix.
    DA : ndarray of shape (N,)
        Determinant of each matrix.

    References
    ----------
    Translation of the MESH2D function ``INV_2X2``.
    Original MATLAB source: https://github.com/dengwirda/mesh2d
    """

    if not isinstance(AA, np.ndarray):
        raise TypeError("inv_2x2:incorrectInputClass")

    if AA.ndim > 3:
        raise ValueError("inv_2x2:incorrectDimensions")

    if AA.shape[0] != 2 or AA.shape[1] != 2:
        raise ValueError("inv_2x2:incorrectDimensions")

    II = np.zeros_like(AA)
    DA = det_2x2(AA)

    II[0, 0, :] = AA[1, 1, :]
    II[1, 1, :] = AA[0, 0, :]
    II[0, 1, :] = -AA[0, 1, :]
    II[1, 0, :] = -AA[1, 0, :]

    return II, DA


def det_2x2(AA):
    """Compute determinants for a block of 2×2 matrices.

    Parameters
    ----------
    AA : ndarray of shape (2, 2, N)
        Stack of ``N`` individual 2×2 matrices.

    Returns
    -------
    DA : ndarray of shape (N,)
        Determinant of each matrix.
    """
    return AA[0, 0, :] * AA[1, 1, :] - AA[0, 1, :] * AA[1, 0, :]
