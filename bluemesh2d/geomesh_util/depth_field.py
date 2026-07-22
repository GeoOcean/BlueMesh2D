import numpy as np
import pyproj
import rasterio
from rasterio.transform import rowcol
from rasterio.warp import transform_bounds
from scipy.interpolate import (
    LinearNDInterpolator,
    NearestNDInterpolator,
    RegularGridInterpolator,
)


def _check_method(method):
    """Validate the interpolation ``method`` shared by all depth-field readers."""
    if method not in ("linear", "nearest"):
        raise ValueError("method must be 'linear' or 'nearest'")
    return method

def depth_field_from_dat(x, y, z, input_crs, output_crs, method="linear"):
    """Build a callable depth field from scattered x-y-z points.

    Parameters
    ----------
    x, y, z : ndarray of shape (N,)
        Point coordinates and depth/elevation values.
    input_crs : str or pyproj.CRS
        CRS of the input coordinates.
    output_crs : str or pyproj.CRS
        CRS in which the returned depth field is queried.
    method : {'linear', 'nearest'}, optional
        Interpolation method. Default is ``'linear'``.

    Returns
    -------
    depth_field : callable
        ``depth_field(xy)`` returns interpolated depth values (m) for ``xy`` of
        shape ``(M, 2)`` in ``output_crs``. Exposes ``.bounds`` in
        ``output_crs``.
    """

    _check_method(method)

    # Clean invalid values
    mask = np.isfinite(x) & np.isfinite(y) & np.isfinite(z)
    x, y, z = x[mask], y[mask], z[mask]

    # Create interpolator
    if method == "linear":
        interp = LinearNDInterpolator(list(zip(x, y)), z, fill_value=np.nan)
    else:  # nearest
        interp = NearestNDInterpolator(list(zip(x, y)), z)

    input_crs = pyproj.CRS.from_user_input(input_crs)
    output_crs = pyproj.CRS.from_user_input(output_crs)
    transformer = pyproj.Transformer.from_crs(output_crs, input_crs, always_xy=True)

    # Closure function
    def depth_field(xy):
        """Evaluate interpolated depth at query coordinates."""
        xs, ys = xy[:, 0], xy[:, 1]
        xs, ys = transformer.transform(xs, ys)
        depth = -interp(xs, ys)
        depth[np.isnan(depth)] = 0.0
        return np.asarray(depth, dtype=float)

    # Expose the data extent in the query (output) CRS, so callers can size a
    # sampling grid without a separate domain (same contract as the other readers).
    depth_field.bounds = transform_bounds(
        input_crs, output_crs, float(x.min()), float(y.min()), float(x.max()), float(y.max())
    )

    return depth_field


def _read_window_decimated(dataset, bbox, max_cells):
    """Read band 1 clipped to ``bbox`` (raster CRS) and decimated to fit.

    Only the requested window is read (never the whole raster), down-sampled
    on read when still larger than ``max_cells`` -- so an oversized bathymetry
    raster is turned into a depth field without materialising it in memory.

    Returns ``(band, transform, step)`` where ``transform`` is the affine of
    the decimated window (maps local col/row -> raster-CRS coordinates).
    """
    from rasterio.transform import Affine
    from rasterio.windows import Window

    W, H = dataset.width, dataset.height
    if bbox is not None:
        xmin, ymin, xmax, ymax = bbox
        inv = ~dataset.transform
        cols, rows = [], []
        for x in (xmin, xmax):
            for y in (ymin, ymax):
                c, r = inv * (x, y)
                cols.append(c)
                rows.append(r)
        col_off = max(0, int(np.floor(min(cols))) - 2)
        row_off = max(0, int(np.floor(min(rows))) - 2)
        col_end = min(W, int(np.ceil(max(cols))) + 2)
        row_end = min(H, int(np.ceil(max(rows))) + 2)
        if col_end <= col_off or row_end <= row_off:
            raise RuntimeError(
                "The domain does not overlap the bathymetry raster.")
    else:
        col_off, row_off, col_end, row_end = 0, 0, W, H

    win = Window(col_off, row_off, col_end - col_off, row_end - row_off)
    win_w, win_h = int(win.width), int(win.height)
    step = 1
    if max_cells and win_w * win_h > max_cells:
        step = int(np.ceil((win_w * win_h / float(max_cells)) ** 0.5))
    out_w = max(1, win_w // step)
    out_h = max(1, win_h // step)

    band = dataset.read(1, window=win, out_shape=(out_h, out_w))
    t = dataset.window_transform(win) * Affine.scale(win_w / out_w, win_h / out_h)
    return band, t, step


def depth_field_from_tif(tiff_path, output_crs, raster_crs=None, method="linear",
                         bbox=None, max_cells=None, invert_z=False,
                         nodata_value=None):
    """Build a callable depth field from a bathymetry GeoTIFF.

    Parameters
    ----------
    tiff_path : str
        Path to the bathymetry GeoTIFF file.
    output_crs : str or pyproj.CRS
        CRS of coordinates passed to the returned depth field.
    raster_crs : str or pyproj.CRS, optional
        CRS of the raster. If ``None``, taken from the GeoTIFF metadata.
    method : {'linear', 'nearest'}, optional
        ``'linear'`` for bilinear interpolation; ``'nearest'`` for nearest pixel.
        Default is ``'linear'``.
    bbox : tuple or None, optional
        ``(xmin, ymin, xmax, ymax)`` in the *raster* CRS. When given, only that
        window of the raster is read (much less memory on large rasters). The
        depth field returns 0 outside it, so the window must cover the full
        query extent. Default ``None`` (whole raster).
    max_cells : int or None, optional
        If the window still exceeds this many cells it is down-sampled on read.
        Default ``None`` (no decimation).
    invert_z : bool, optional
        The raster is assumed to store elevation (positive up), so depth is
        ``-value``. Set ``True`` when the raster already stores depth
        (positive down) or has an inverted Z sign, giving depth ``+value``.
        Default ``False``.
    nodata_value : float or None, optional
        Elevation (positive up) assigned to nodata / non-finite pixels.
        ``None`` (default) uses 0 (sea level -> depth 0).

    Returns
    -------
    depth_field : callable
        ``depth_field(xy)`` returns depth (m) for ``xy`` of shape ``(N, 2)`` in
        ``output_crs``. Exposes ``.bounds`` in ``output_crs``.
    """

    _check_method(method)

    from bluemesh2d.geom_util.proj_util import bundled_raster_data_env
    with bundled_raster_data_env(), rasterio.open(tiff_path) as dataset:
        if raster_crs is None:
            raster_crs = dataset.crs
        nodata = dataset.nodata
        band, transform, _ = _read_window_decimated(dataset, bbox, max_cells)
        # window bounds in raster CRS (from the decimated transform)
        x0, y0 = transform * (0, 0)
        x1, y1 = transform * (band.shape[1], band.shape[0])
        rxmin, rxmax = sorted((x0, x1))
        rymin, rymax = sorted((y0, y1))

    output_crs = pyproj.CRS.from_user_input(output_crs)
    raster_crs = pyproj.CRS.from_user_input(raster_crs) if raster_crs else output_crs
    if raster_crs != output_crs:
        transformer = pyproj.Transformer.from_crs(
            output_crs, raster_crs, always_xy=True
        )
    else:
        transformer = None

    # Depth from elevation (-value), or +value when the Z sign is inverted.
    # nodata / non-finite pixels are replaced by 0 (sea level -> depth 0), so
    # a NaN can never propagate into a nodata hole in the hfun raster.
    sign = 1.0 if invert_z else -1.0
    depth_grid = np.asarray(band, dtype=np.float64) * sign
    bad = ~np.isfinite(depth_grid)
    if nodata is not None:
        bad |= (np.asarray(band) == nodata)
    if bad.any():
        nv = 0.0 if nodata_value is None else float(nodata_value)
        depth_grid[bad] = sign * nv

    if method == "linear":
        # Build interpolator in pixel (row, col) space
        inv_transform = ~transform
        rows = np.arange(band.shape[0])
        cols = np.arange(band.shape[1])
        interp = RegularGridInterpolator(
            (rows, cols),
            depth_grid,
            method="linear",
            bounds_error=False,
            fill_value=np.nan,
        )

    def depth_field(xy):
        xs, ys = xy[:, 0], xy[:, 1]
        if transformer is not None:
            xs, ys = transformer.transform(xs, ys)
        xs, ys = np.asarray(xs), np.asarray(ys)

        if method == "nearest":
            rows, cols = rowcol(transform, xs, ys)
            rows = np.clip(rows, 0, band.shape[0] - 1)
            cols = np.clip(cols, 0, band.shape[1] - 1)
            depth = depth_grid[rows, cols]
        else:
            # method == "linear": continuous (col, row) from inverse transform
            col_row = np.column_stack(inv_transform * (xs, ys))
            # RegularGridInterpolator expects (row, col) for array [rows, cols];
            # subtract 0.5 because the transform is corner-referenced while the
            # grid values sit at cell centres (index k) -- avoids a half-cell
            # shift in the sampled depth.
            row_col = col_row[:, [1, 0]] - 0.5
            depth = interp(row_col)
        # queries outside the read window return NaN; treat as depth 0 so the
        # size function never produces nodata
        depth = np.asarray(depth, dtype=np.float64)
        depth[~np.isfinite(depth)] = 0.0
        return depth

    # Expose the (windowed) raster extent in the query (output) CRS as
    # (xmin, ymin, xmax, ymax), so callers can size a sampling grid.
    if raster_crs != output_crs:
        depth_field.bounds = transform_bounds(
            raster_crs, output_crs, rxmin, rymin, rxmax, rymax)
    else:
        depth_field.bounds = (rxmin, rymin, rxmax, rymax)

    return depth_field


def depth_field_from_xr(ds, input_crs, output_crs, x_name='lon', y_name='lat',
                        z_name="elevation", method="linear"):
    """Build a callable depth field from an xarray bathymetry dataset.

    Parameters
    ----------
    ds : xarray.Dataset
        Bathymetry dataset (e.g. GEBCO subset).
    input_crs : str or pyproj.CRS
        CRS of the dataset coordinates.
    output_crs : str or pyproj.CRS
        CRS in which the depth field is queried.
    x_name, y_name, z_name : str, optional
        Names of longitude, latitude, and elevation variables in ``ds``.
    method : {'linear', 'nearest'}, optional
        Grid interpolation method. Default is ``'linear'``.

    Returns
    -------
    depth_field : callable
        ``depth_field(xy)`` returns depth values (m) for ``xy`` of shape
        ``(N, 2)`` in ``output_crs``. Exposes ``.bounds`` in ``output_crs``.
    """

    _check_method(method)

    # Extract lon/lat grid and data (z assumed [lat, lon])
    lon = np.asarray(ds[x_name].values, dtype=float)
    lat = np.asarray(ds[y_name].values, dtype=float)
    z = np.asarray(ds[z_name].values, dtype=float)
    if z.ndim != 2:
        raise ValueError(f"z must be 2D, got shape {z.shape}")

    # RegularGridInterpolator needs strictly increasing axes; flip if needed
    if lat[0] > lat[-1]:
        lat, z = lat[::-1], z[::-1, :]
    if lon[0] > lon[-1]:
        lon, z = lon[::-1], z[:, ::-1]

    interp = RegularGridInterpolator(
        (lat, lon), z, method=method, bounds_error=False, fill_value=np.nan
    )

    input_crs = pyproj.CRS.from_user_input(input_crs)
    output_crs = pyproj.CRS.from_user_input(output_crs)
    to_ds = pyproj.Transformer.from_crs(output_crs, input_crs, always_xy=True)

    def depth_field(xy):
        """Evaluate interpolated depth at query coordinates."""
        xs, ys = xy[:, 0], xy[:, 1]
        lon_q, lat_q = to_ds.transform(xs, ys)
        # depth = -elevation; interpolator expects (lat, lon) order
        depth = -interp(np.column_stack([np.asarray(lat_q), np.asarray(lon_q)]))
        return np.asarray(depth, dtype=np.float64)

    # Expose the data extent in the query (output) CRS (same contract as the others).
    depth_field.bounds = transform_bounds(
        input_crs, output_crs, float(lon.min()), float(lat.min()),
        float(lon.max()), float(lat.max())
    )

    return depth_field