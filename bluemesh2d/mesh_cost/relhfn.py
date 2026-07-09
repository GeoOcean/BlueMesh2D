import numpy as np


def relhfn(vert, tria, hvrt):
    """Compute relative edge lengths against a mesh-size function.

    For each unique mesh edge, return ``edge_length / mean(h at endpoints)``.
    Values near 1 indicate good conformance to the sizing field.

    Parameters
    ----------
    vert : ndarray of shape (V, 2)
        Vertex coordinates.
    tria : ndarray of shape (T, 3)
        Triangle connectivity.
    hvrt : ndarray of shape (V,)
        Mesh-size function evaluated at vertices.

    Returns
    -------
    hrel : ndarray of shape (E,)
        Relative edge lengths for the unique edges in the triangulation.

    References
    ----------
    Translation of the MESH2D function ``RELHFN2``.
    Original MATLAB source: https://github.com/dengwirda/mesh2d
    """

    if not (
        isinstance(vert, np.ndarray)
        and isinstance(tria, np.ndarray)
        and isinstance(hvrt, np.ndarray)
    ):
        raise TypeError("relhfn:incorrectInputClass")

    if vert.ndim != 2 or tria.ndim != 2:
        raise ValueError("relhfn:incorrectDimensions")
    if vert.shape[1] != 2 or tria.shape[1] < 3:
        raise ValueError("relhfn:incorrectDimensions")
    if len(hvrt.shape) != 1 or hvrt.shape[0] != vert.shape[0]:
        raise ValueError("relhfn:incorrectDimensions")

    nnod = vert.shape[0]

    if np.min(tria[:, :3]) < 0 or np.max(tria[:, :3]) >= nnod:
        raise ValueError("relhfn:invalidInputs")

    eset = np.vstack([tria[:, [0, 1]], tria[:, [1, 2]], tria[:, [2, 0]]])

    eset = np.sort(eset, axis=1)
    eset = np.unique(eset, axis=0)

    evec = vert[eset[:, 1], :] - vert[eset[:, 0], :]

    elen = np.sqrt(np.sum(evec**2, axis=1))

    hmid = hvrt[eset[:, 1]] + hvrt[eset[:, 0]]
    hmid = 0.5 * hmid
    hrel = elen / hmid

    return hrel
