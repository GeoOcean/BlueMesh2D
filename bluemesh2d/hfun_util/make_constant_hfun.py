import numpy as np


def make_constant_hfun(h, bounds=None):
    """Constant mesh-size function: ``hfun(xy) = h`` everywhere.

    Parameters
    ----------
    h : float
        Element size (same units as the mesh coordinates, e.g. metres).
    bounds : tuple or None, optional
        Optional ``(xmin, ymin, xmax, ymax)`` attached to the returned
        function as ``.bounds`` (used by ``smooth_precomput_hfun`` to derive
        its domain). Default is ``None`` (no attribute).

    Returns
    -------
    hfun : callable
        ``hfun(test) -> h``, element size at query points ``test`` of shape
        ``(N, 2)``, returned as an ``(N,)`` array.
    """
    h = float(h)
    if not np.isfinite(h) or h <= 0:
        raise ValueError("make_constant_hfun: h must be a positive finite value")

    def hfun(test):
        xy = np.atleast_2d(np.asarray(test, dtype=float))
        return np.full(xy.shape[0], h)

    if bounds is not None:
        hfun.bounds = tuple(bounds)
    return hfun
