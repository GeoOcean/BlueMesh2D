import numpy as np

def limgrad(edge, elen, ffun, dfdx, imax):
    """
    Apply gradient limiting to a scalar function defined over an undirected graph.

    This function enforces a maximum gradient constraint on a function `FFUN` defined 
    on the nodes of an undirected graph `{EDGE, ELEN}`. The goal is to ensure that 
    differences in function values across connected nodes do not exceed a prescribed 
    gradient limit, scaled by edge length.

    Parameters
    ----------
    EDGE : ndarray of shape (E, 2)
        Array of undirected edge connections, where each row defines an edge between 
        two node indices `(n1, n2)`.
    ELEN : ndarray of shape (E,)
        Array of edge lengths associated with each edge in `EDGE`.
    FFUN : ndarray of shape (N,)
        Original function values at the graph nodes.
    DFDX : float
        Maximum allowed gradient magnitude. The constraint  
        `abs(FNEW[n2] - FNEW[n1]) <= ELEN[i] * DFDX`  
        is enforced for each edge `(n1, n2)`.
    ITER : int, optional
        Maximum number of iterations allowed for convergence.

    Returns
    -------
    FNEW : ndarray of shape (N,)
        Gradient-limited version of the function `FFUN`.
    FLAG : bool
        Convergence flag. `True` if convergence was achieved within the iteration limit.

    Notes
    -----
    - The algorithm iteratively sweeps through all edges in the graph, 
      adjusting node values until the gradient constraint is satisfied globally.
    - Particularly useful for limiting mesh-size gradients or similar scalar fields 
      defined on unstructured connectivity graphs.

    References
    ----------
    Translation of the MESH2D function `LIMGRAD`.  
    Original MATLAB source: https://github.com/dengwirda/mesh2d  

    See also
    --------
    limhfn : Apply gradient limiting to 2D mesh-size functions.
    """

    # ---------------------------- Basic input checks
    edge = np.asarray(edge, dtype=int)
    elen = np.asarray(elen, dtype=float).reshape(-1)
    ffun = np.asarray(ffun, dtype=float).reshape(-1)

    nnod = ffun.shape[0]

    if edge.ndim != 2 or edge.shape[1] < 2:
        raise ValueError("limgrad:incorrectDimensions - EDGE must be (NE,2)")
    if elen.ndim > 2 or ffun.ndim > 2:
        raise ValueError("limgrad:incorrectDimensions - ELEN/FFUN must be vectors")
    if elen.shape[0] != edge.shape[0]:
        raise ValueError("limgrad:incorrectDimensions - ELEN and EDGE must have same length")
    if dfdx < 0.0 or imax < 0:
        raise ValueError("limgrad:invalidInputArgument - DFDX or IMAX invalid")

    if edge[:, :2].min() < 0 or edge[:, :2].max() >= nnod:
        raise ValueError("limgrad:invalidInputArgument - invalid EDGE indices")

    #IVEC(NPTR(II,1):NPTR(II,2)) are edges adj. to II-TH node
    nvec = np.concatenate([edge[:, 0], edge[:, 1]])
    ivec = np.concatenate([
        np.arange(edge.shape[0], dtype=int),
        np.arange(edge.shape[0], dtype=int)
    ])

    sort_idx = np.argsort(nvec)
    nvec = nvec[sort_idx]
    ivec = ivec[sort_idx]

    mark = np.zeros(nnod, dtype=bool)
    mark[edge[:, 0]] = True
    mark[edge[:, 1]] = True

    idxx = np.where(np.diff(nvec) > 0)[0]

    nptr = np.full((nnod, 2), -1, dtype=int)
    if idxx.size > 0:
        nptr[mark, 0] = np.concatenate(([0], idxx + 1))
        nptr[mark, 1] = np.concatenate((idxx, [len(nvec) - 1]))
    else:
        nptr[mark, 0] = 0
        nptr[mark, 1] = len(nvec) - 1

    # ----------------------------- ASET=ITER if node is "active"
    aset = np.zeros(nnod, dtype=int)
    #----------------------------- exhaustive 'til all satisfied
    ftol = np.min(ffun) * np.sqrt(np.finfo(float).eps)

    for iter in range(1, imax.astype(int) + 1):
        # ------------------------- find "active" nodes this pass
        aidx = np.where(aset == iter - 1)[0]
        if aidx.size == 0:
            break

        # ------------------------ reorder => better convergence
        aidx = aidx[np.argsort(ffun[aidx])]

        # ------------------------- visit adj. edges and set DFDX
        for npos in aidx:
            for jpos in range(nptr[npos, 0], nptr[npos, 1] + 1):
                epos = ivec[jpos]
                nod1, nod2 = edge[epos, :2]

                # ----------------- calc. limits about min.-value
                if ffun[nod1] > ffun[nod2]:
                    fun1 = ffun[nod2] + elen[epos] * dfdx
                    if ffun[nod1] > fun1 + ftol:
                        ffun[nod1] = fun1
                        aset[nod1] = iter
                else:
                    fun2 = ffun[nod1] + elen[epos] * dfdx
                    if ffun[nod2] > fun2 + ftol:
                        ffun[nod2] = fun2
                        aset[nod2] = iter

    flag = (iter < imax)
    return ffun, flag


