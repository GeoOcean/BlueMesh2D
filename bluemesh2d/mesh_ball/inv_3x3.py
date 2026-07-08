import numpy as np


def inv_3x3(AA):
    """
    Compute the inverses for a block of 3×3 matrices.

    This function computes the determinants and numerically robust "incomplete inverses"
    for a collection of 3×3 matrices. The inverse matrices are scaled by their determinants
    to improve numerical stability.

    Parameters
    ----------
    AA : ndarray of shape (3, 3, N)
        Array containing N individual 3×3 matrices.

    Returns
    -------
    IA : ndarray of shape (3, 3, N)
        Scaled inverses of each input matrix, such that:
        `IA[:, :, k] = det(A[:, :, k]) * inv(A[:, :, k])`.
    DA : ndarray of shape (N,)
        Determinants of each input matrix.

    Notes
    -----
    - Each returned matrix `IA[:, :, k]` is an *incomplete inverse*,
      scaled by its determinant to enhance numerical robustness.
    - To solve a linear system `A * X = B`, compute `(IA * B) / DA`,
      provided that `DA` is non-zero.

    See Also
    --------
    inv_2x2 : Compute the same for 2×2 matrices.

    References
    ----------
    Translation of the MESH2D function `INV_3X3`.
    Original MATLAB source: https://github.com/dengwirda/mesh2d
    """

    # ---------------------------------------------- basic checks
    if not isinstance(AA, np.ndarray):
        raise TypeError("inv_3x3:incorrectInputClass")

    if AA.ndim > 3:
        raise ValueError("inv_3x3:incorrectDimensions")

    if AA.shape[0] != 3 or AA.shape[1] != 3:
        raise ValueError("inv_3x3:incorrectDimensions")

    # --------------------------------------------- build inv(A)
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
    """
    Determinant for block of 3x3 matrices.

    Parameters
    ----------
    AA : (3,3,N) array

    Returns
    -------
    DA : (N,) array
    """
    return (
        AA[0, 0, :] * (AA[1, 1, :] * AA[2, 2, :] - AA[1, 2, :] * AA[2, 1, :])
        - AA[0, 1, :] * (AA[1, 0, :] * AA[2, 2, :] - AA[1, 2, :] * AA[2, 0, :])
        + AA[0, 2, :] * (AA[1, 0, :] * AA[2, 1, :] - AA[1, 1, :] * AA[2, 0, :])
    )
