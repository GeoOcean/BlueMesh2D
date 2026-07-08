import numpy as np


def inv_2x2(AA):
    """
    Compute the inverses for a block of 2×2 matrices.

    This function computes the determinants and numerically robust "incomplete inverses"
    for a collection of 2×2 matrices. The inverse matrices are scaled by their determinants
    to improve numerical stability.

    Parameters
    ----------
    AA : ndarray of shape (2, 2, N)
        Array containing N individual 2×2 matrices.

    Returns
    -------
    IA : ndarray of shape (2, 2, N)
        Scaled inverses of each input matrix, such that:
        `IA[:, :, k] = det(A[:, :, k]) * inv(A[:, :, k])`.
    DA : ndarray of shape (N,)
        Determinants of each input matrix.

    Notes
    -----
    - Each returned matrix `IA[:, :, k]` is an *incomplete inverse*,
      scaled by its determinant to improve numerical robustness.
    - To solve a linear system `A * X = B`, compute `(IA * B) / DA`,
      provided that `DA` is non-zero.

    See Also
    --------
    inv_3x3 : Compute the same for 3×3 matrices.

    References
    ----------
    Translation of the MESH2D function `INV_2X2`.
    Original MATLAB source: https://github.com/dengwirda/mesh2d
    """

    # ---------------------------------------------- basic checks
    if not isinstance(AA, np.ndarray):
        raise TypeError("inv_2x2:incorrectInputClass")

    if AA.ndim > 3:
        raise ValueError("inv_2x2:incorrectDimensions")

    if AA.shape[0] != 2 or AA.shape[1] != 2:
        raise ValueError("inv_2x2:incorrectDimensions")

    # ---------------------------------------------- build inv(A)
    II = np.zeros_like(AA)
    DA = det_2x2(AA)

    II[0, 0, :] = AA[1, 1, :]
    II[1, 1, :] = AA[0, 0, :]
    II[0, 1, :] = -AA[0, 1, :]
    II[1, 0, :] = -AA[1, 0, :]

    return II, DA


def det_2x2(AA):
    """
    Determinant for block of 2x2 matrices.

    Parameters
    ----------
    AA : (2,2,N) array

    Returns
    -------
    DA : (N,) array
    """
    return AA[0, 0, :] * AA[1, 1, :] - AA[0, 1, :] * AA[1, 0, :]
