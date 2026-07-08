from scipy.interpolate import RegularGridInterpolator
import numpy as np
import matplotlib.pyplot as plt


def carry_bounds(source):
    """Decorator that copies a region onto a size function as ``.bounds``.

    ``smooth_precomput`` reads ``hfun.bounds`` to size its sampling grid without a
    separate domain. Rather than assigning that attribute by hand, decorate the
    size function with the object it derives its extent from (e.g. a
    ``depth_field_from_*`` callable, which exposes ``.bounds``) or a bounds tuple:

        @carry_bounds(depth_field)
        def hfun(xy):
            ...

    Parameters
    ----------
    source : object with a ``.bounds`` attribute, or (xmin, ymin, xmax, ymax)
        Where the extent comes from.
    """
    bounds = source.bounds if hasattr(source, "bounds") else source

    def decorator(func):
        func.bounds = bounds
        return func

    return decorator


def smooth_precomput(hfun, domain=None, max_gradient=0.1, cell_size=None, plot=False):
    """Gradient-limited, pre-computed mesh sizing function.

    Returns a **continuous** callable ``hfuns(xy)`` that behaves like ``hfun``
    but whose spatial gradient is capped: ``|grad(h)| <= max_gradient`` (metres
    of element size per metre travelled). It is the largest ``max_gradient``-
    Lipschitz field that stays below ``hfun`` (the Lipschitz lower envelope,
    ``u(x) = inf_y [ hfun(y) + max_gradient * ||x - y|| ]``), so peaks that grow
    too fast are lowered toward their small neighbours and the fine/coarse
    transition is smooth.

    Gradient limiting is inherently *non-local* (the value at ``x`` depends on
    the field around ``x``), so the black-box ``hfun`` must be sampled over a
    region at some resolution. The region is taken from ``domain`` and the
    resolution / padding are derived from the size field itself, so no tuning
    parameters are needed. The result is cached in a fast linear interpolator,
    making the many repeated calls during meshing cheap.

    Parameters
    ----------
    hfun : callable
        Size function ``hfun(xy) -> h`` with ``xy`` an ``(N, 2)`` array of UTM
        coordinates (metres) and ``h`` the requested element size (metres).
    domain : geometry, (xmin, ymin, xmax, ymax), or None
        Region (in the same UTM coordinates as ``hfun``) to pre-compute over.
        Either any object exposing a ``.bounds`` attribute (e.g. a Shapely
        geometry) or a 4-tuple of bounds. If ``None`` (default), the region is
        taken from ``hfun.bounds`` -- e.g. the raster extent carried by a
        ``depth_field_from_tif`` sizing function -- so no separate domain is
        needed.
    max_gradient : float, optional
        Maximum allowed size gradient in m/m (default 0.1).
    cell_size : float, optional
        Resolution (m) of the internal sampling grid. If ``None`` (default) it is
        auto-derived fine enough to resolve the size transitions (~max extent /
        1200, floored at half the smallest size). Pass a smaller value for
        smoother transitions in fine zones at the cost of memory / build time.
    plot : bool, optional
        If True, show a 3-panel comparison: raw ``hfun``, smoothed field, and
        the reduction (raw - smoothed) applied by the gradient limiting.

    Returns
    -------
    hfuns : callable
        ``hfuns(xy) -> h`` — continuous, gradient-limited, cheap to evaluate.
    """
    g = max_gradient
    if domain is None:
        domain = getattr(hfun, "bounds", None)
    if domain is None:
        raise ValueError(
            "smooth_precomput needs a region: pass `domain` (a geometry or "
            "(xmin, ymin, xmax, ymax)) or attach a `.bounds` attribute to hfun "
            "(depth_field_from_tif provides one)."
        )
    bounds = domain.bounds if hasattr(domain, "bounds") else domain
    xmin, ymin, xmax, ymax = bounds
    w, h = xmax - xmin, ymax - ymin

    # -----------------------probe the size range to auto-scale the sampling
    px = np.linspace(xmin, xmax, 80)
    py = np.linspace(ymin, ymax, 80)
    PX, PY = np.meshgrid(px, py)
    Hp = np.asarray(hfun(np.column_stack([PX.ravel(), PY.ravel()])), float)
    Hp = Hp[np.isfinite(Hp)]
    hmin_est, hmax_est = float(Hp.min()), float(Hp.max())

    # grid fine enough to resolve the size transitions (capped ~1200 cells/side);
    # finer than the element spacing everywhere is infeasible over a large domain,
    # so smooth (pchip) reconstruction below hides the residual grid structure.
    if cell_size is None:
        cell_size = max(max(w, h) / 1200.0, hmin_est / 2.0)
    # pad by the influence radius so the field is valid up to the domain edge
    margin = min((hmax_est - hmin_est) / g, 0.25 * max(w, h))

    # -----------------------build the sampling grid and sample hfun once
    xs = np.arange(xmin - margin, xmax + margin + cell_size, cell_size)
    ys = np.arange(ymin - margin, ymax + margin + cell_size, cell_size)
    X, Y = np.meshgrid(xs, ys)  # (ny, nx)
    H = np.asarray(hfun(np.column_stack([X.ravel(), Y.ravel()])), float)
    H = H.reshape(X.shape)
    # outside the raster hfun may return NaN -> treat unknown areas as coarse
    H[~np.isfinite(H)] = hmax_est
    H_raw = H.copy()  # keep the un-limited field for the optional plot

    # -----------------------limit the gradient (grid "limgrad")
    # Relax every cell against its 8 neighbours until no value can still be
    # lowered to satisfy h[a] - h[b] <= max_gradient * dist(a, b).
    dx = xs[1] - xs[0]
    dy = ys[1] - ys[0]
    dd = np.hypot(dx, dy)
    neighbours = [
        ((slice(1, None), slice(None)), (slice(0, -1), slice(None)), g * dy),
        ((slice(0, -1), slice(None)), (slice(1, None), slice(None)), g * dy),
        ((slice(None), slice(1, None)), (slice(None), slice(0, -1)), g * dx),
        ((slice(None), slice(0, -1)), (slice(None), slice(1, None)), g * dx),
        ((slice(1, None), slice(1, None)), (slice(0, -1), slice(0, -1)), g * dd),
        ((slice(0, -1), slice(0, -1)), (slice(1, None), slice(1, None)), g * dd),
        ((slice(1, None), slice(0, -1)), (slice(0, -1), slice(1, None)), g * dd),
        ((slice(0, -1), slice(1, None)), (slice(1, None), slice(0, -1)), g * dd),
    ]
    for _ in range(H.shape[0] + H.shape[1] + 10):
        changed = False
        for tgt, src, step in neighbours:
            cand = H[src] + step
            lower = cand < H[tgt]
            if lower.any():
                H[tgt][lower] = cand[lower]
                changed = True
        if not changed:
            break

    # -----------------------optional comparison plot
    if plot:
        extent = [xs[0], xs[-1], ys[0], ys[-1]]
        vmin, vmax = H_raw.min(), H_raw.max()
        panels = [
            (H_raw, "hfun (raw)", "viridis", "h (m)", vmin, vmax),
            (H, "hfuns (smoothed)", "viridis", "h (m)", vmin, vmax),
            (H_raw - H, "reduction (raw - smoothed)", "RdYlBu_r", "m", None, None),
        ]
        fig, axes = plt.subplots(1, 3, figsize=(16, 5))
        for ax, (data, title, cmap, lab, lo, hi) in zip(axes, panels):
            im = ax.imshow(data, extent=extent, origin="lower", cmap=cmap,
                           aspect="equal", vmin=lo, vmax=hi)
            ax.set(title=title, xlabel="x (m)", ylabel="y (m)")
            plt.colorbar(im, ax=ax, label=lab)
        fig.tight_layout()
        plt.show()

    # -----------------------cache result in a fast interpolator
    # pchip (shape-preserving, C1) reconstruction: removes the gradient kinks a
    # linear interpolant leaves at grid lines (the "raster cell" imprint) without
    # the ringing/overshoot a plain cubic spline produces, so the size field
    # varies smoothly even where the grid is coarser than the elements.
    interp = RegularGridInterpolator(
        (ys, xs), H, method="pchip", bounds_error=False, fill_value=None
    )
    h_lo, h_hi = float(H.min()), float(H.max())

    def hfuns(xy):
        xy = np.asarray(xy, dtype=float)
        one = xy.ndim == 1
        if one:
            xy = xy.reshape(1, -1)
        # RegularGridInterpolator expects (row=y, col=x) ordering
        h = interp(np.column_stack([xy[:, 1], xy[:, 0]]))
        # clip cubic overshoot back into the field's own range (never below the
        # smallest requested size, so no explosive sub-hmin elements)
        np.clip(h, h_lo, h_hi, out=h)
        return h[0] if one else h

    return hfuns