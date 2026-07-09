import numpy as np


def inv_3x3(AA):
    """Compute determinant-scaled inverses for a block of 3×3 matrices.

    Returns ``det(A) * inv(A)`` for numerical robustness. Divide by ``DA``
    when solving linear systems.

    Parameters
    ----------
    AA : ndarray of shape (3, 3, N)
        Stack of ``N`` individual 3×3 matrices.

    Returns
    -------
    II : ndarray of shape (3, 3, N)
        Determinant-scaled inverse of each matrix.
    DA : ndarray of shape (N,)
        Determinant of each matrix.

    References
    ----------
    Translation of the MESH2D function ``INV_3X3``.
    Original MATLAB source: https://github.com/dengwirda/mesh2d
    """

    if not isinstance(AA, np.ndarray):
        raise TypeError("inv_3x3:incorrectInputClass")

    if AA.ndim > 3:
        raise ValueError("inv_3x3:incorrectDimensions")

    if AA.shape[0] != 3 or AA.shape[1] != 3:
        raise ValueError("inv_3x3:incorrectDimensions")

    II = np.zeros_like(AA)
    DA = det_3x3(AA)

    II[0, 0, :] = AA[2, 2, :] * AA[1, 1, :] - AA[2, 1, :] * AA[1, 2, :]
    II[0, 1, :] = AA[2, 1, :] * AA[0, 2, :] - AA[2, 2, :] * AA[0, 1, :]
    II[0, 2, :] = AA[1, 2, :] * AA[0, 1, :] - AA[1, 1, :] * AA[0, 2, :]

    II[1, 0, :] = AA[2, 0, :] * AA[1, 2, :] - AA[2, 2, :] * AA[1, 0, :]
    II[1, 1, :] = AA[2, 2, :] * AA[0, 0, :] - AA[2, 0, :] * AA[0, 2, :]
    II[1, 2, :] = AA[1, 0, :] * AA[0, 2, :] - AA[1, 2, :] * AA[0, 0, :]

    II[2, 0, :] = AA[2, 1, :] * AA[1, 0, :] - AA[2, 0, :] * AA[1, 1, :]
    II[2, 1, :] = AA[2, 0, :] * AA[0, 1, :] - AA[2, 1, :] * AA[0, 0, :]
    II[2, 2, :] = AA[1, 1, :] * AA[0, 0, :] - AA[1, 0, :] * AA[0, 1, :]

    return II, DA


def det_3x3(AA):
    """Compute determinants for a block of 3×3 matrices.

    Parameters
    ----------
    AA : ndarray of shape (3, 3, N)
        Stack of ``N`` individual 3×3 matrices.

    Returns
    -------
    DA : ndarray of shape (N,)
        Determinant of each matrix.
    """
    return (
        AA[0, 0, :] * (AA[1, 1, :] * AA[2, 2, :] - AA[1, 2, :] * AA[2, 1, :])
        - AA[0, 1, :] * (AA[1, 0, :] * AA[2, 2, :] - AA[1, 2, :] * AA[2, 0, :])
        + AA[0, 2, :] * (AA[1, 0, :] * AA[2, 1, :] - AA[1, 1, :] * AA[2, 0, :])
    )
