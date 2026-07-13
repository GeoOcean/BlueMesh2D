import shapely
import numpy as np


def make_depth_hfun(depth_field, a=0.14, b=28.0, hmin=100.0, hmax=10000.0,
                    detail=None, detail_hmin=None,
                    slope_ncells=None, slope_step=500.0):
    """Depth-based mesh-size function.

    h(xy) = clip(a*d**2 + b*d, hmin, hmax), with d = max(depth_field(xy), 0),
    and an optional finer floor `detail_hmin` inside the `detail` polygon.

    If `slope_ncells` is given (e.g. 15), the size is also limited by the
    bathymetric-slope term  h_slope = 2*pi*d / (slope_ncells * |grad d|),
    which refines the mesh where the bathymetry is steep (shelf break) and
    has no effect where it is flat. Leave it None to disable. `slope_step`
    is the finite-difference step (m) used to estimate the gradient -- use
    roughly the bathymetry raster resolution.

    Carries `.bounds` from `depth_field`, so `smooth_precomput_hfun` needs no domain.
    """
    detail_mask = None
    if detail is not None and detail_hmin is not None:
        def detail_mask(xy):
            return shapely.contains_xy(detail, xy[:, 0], xy[:, 1])

    def grad_mag(xy):
        e = slope_step
        dzdx = (np.asarray(depth_field(xy + [e, 0.0]), dtype=float).reshape(-1)
                - np.asarray(depth_field(xy - [e, 0.0]), dtype=float).reshape(-1)) / (2 * e)
        dzdy = (np.asarray(depth_field(xy + [0.0, e]), dtype=float).reshape(-1)
                - np.asarray(depth_field(xy - [0.0, e]), dtype=float).reshape(-1)) / (2 * e)
        return np.hypot(dzdx, dzdy)

    def hfun(test):
        xy = np.atleast_2d(np.asarray(test, dtype=float))
        d = np.asarray(depth_field(xy), dtype=float).reshape(-1)
        d = np.where(d < 0, 0.0, d)  # depth is non-negative
        values = a * d**2 + b * d
        if slope_ncells is not None:
            g = grad_mag(xy)
            h_slope = 2.0 * np.pi * d / (slope_ncells * np.maximum(g, 1e-12))
            values = np.minimum(values, h_slope)
        lo = np.full(xy.shape[0], hmin, dtype=float)
        if detail_mask is not None:
            lo[detail_mask(xy)] = detail_hmin
        return np.clip(values, lo, hmax)

    if hasattr(depth_field, "bounds"):
        hfun.bounds = depth_field.bounds
    return hfun