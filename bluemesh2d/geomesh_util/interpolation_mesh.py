import numpy as np
import pyproj
import rasterio
from scipy.interpolate import LinearNDInterpolator
from scipy.ndimage import map_coordinates, distance_transform_edt
from scipy.spatial import cKDTree


def interpolate_from_xyz(
    x, y, z,
    vert,
    method="rbf",
    rbf_function="cubic",
    epsilon=None,
):
    """Interpolate scattered values at arbitrary target coordinates.

    Parameters
    ----------
    x, y : ndarray of shape (N,)
        Coordinates of scattered data points.
    z : ndarray of shape (N,)
        Scalar values at the scattered points.
    vert : ndarray of shape (M, 2) or (M, 3)
        Target coordinates where interpolation is evaluated.
    method : {'linear', 'nearest', 'rbf'}, optional
        Interpolation method. ``'linear'`` uses
        :class:`scipy.interpolate.LinearNDInterpolator`; ``'nearest'`` uses a
        KD-tree; ``'rbf'`` uses :class:`scipy.interpolate.RBFInterpolator``.
    rbf_function : str, optional
        RBF kernel (e.g. ``'cubic'``, ``'thin_plate_spline'``). Used only when
        ``method='rbf'``.
    epsilon : float, optional
        RBF shape parameter. Auto-estimated when ``None``.

    Returns
    -------
    values_interp : ndarray of shape (M,)
        Interpolated values at ``vert`` (negated; NaN replaced by 0).
    """

    # Remove invalid points
    mask = np.isfinite(x) & np.isfinite(y) & np.isfinite(z)
    x, y, z = x[mask], y[mask], z[mask]
    points = np.column_stack((x, y))

    # Interpolation method
    if method == "linear":
        interp = LinearNDInterpolator(points, z, fill_value=np.nan)
        values_interp = interp(vert[:, 0], vert[:, 1], vert[:, 2])

    elif method == "nearest":
        tree = cKDTree(points)
        _, idx = tree.query(vert)
        values_interp = z[idx]

    elif method == "rbf":
        # lazy import: RBFInterpolator needs scipy >= 1.7, and old QGIS
        # bundles (e.g. macOS LTR with scipy 1.5) must still be able to use
        # the other methods
        try:
            from scipy.interpolate import RBFInterpolator
        except ImportError as exc:
            raise ImportError(
                "method='rbf' needs scipy >= 1.7 (RBFInterpolator); this "
                "environment has an older scipy -- use method='linear' or "
                "'nearest' instead.") from exc
        interp = RBFInterpolator(points, z, kernel=rbf_function, epsilon=epsilon)
        values_interp = interp(vert)

    else:
        raise ValueError("method must be 'linear', 'nearest', or 'rbf'")

    # Handle NaNs
    values_interp = - np.asarray(values_interp, dtype=float)
    values_interp[np.isnan(values_interp)] = 0

    return values_interp


def interpolate_from_tiff(
    tiff_path, vert, input_crs=None, order=3, mode="constant", cval=np.nan,
    invert_z=False, nodata_value=None, max_cells=None
):
    """Interpolate GeoTIFF raster values at mesh node coordinates.

    Only the window covering the mesh nodes is read (decimated when very
    large), so an oversized bathymetry raster is sampled without being read
    in full.

    Parameters
    ----------
    tiff_path : str
        Path to the GeoTIFF file.
    vert : ndarray of shape (N, 2)
        Node coordinates ``(x, y)`` in ``input_crs``.
    input_crs : str or pyproj.CRS, optional
        CRS of ``vert``. If ``None``, assumes the raster CRS.
    order : int, optional
        Interpolation order for :func:`scipy.ndimage.map_coordinates` (0 =
        nearest, 1 = bilinear, 3 = bicubic). Default is 3.
    mode : str, optional
        Boundary handling mode. Default is ``'constant'``.
    cval : float, optional
        Fill value outside the domain when ``mode='constant'``. Default is
        ``np.nan``.
    invert_z : bool, optional
        By default the raster stores elevation (positive up) and node values
        are depth ``-value``. Set ``True`` for a depth-positive-down raster,
        giving ``+value``. Default ``False``.
    nodata_value : float or None, optional
        Elevation (positive up) assigned to nodata / non-finite pixels.
        ``None`` (default) fills them from the nearest valid pixel.
    max_cells : int or None, optional
        Decimate the read window above this many cells. ``None`` sizes it to
        available RAM.

    Returns
    -------
    z : ndarray of shape (N,)
        Node values: depth (``-elevation``), or ``+value`` if ``invert_z``.
    """
    from bluemesh2d.feedback import _available_ram_bytes
    from bluemesh2d.geom_util.proj_util import bundled_raster_data_env
    from bluemesh2d.geomesh_util.depth_field import _read_window_decimated

    if max_cells is None:
        avail = _available_ram_bytes() or 4_000_000_000
        max_cells = int(min(120_000_000, max(4_000_000, 0.15 * avail / 16.0)))
    sign = 1.0 if invert_z else -1.0

    with bundled_raster_data_env(), rasterio.open(tiff_path) as src:
        raster_crs = src.crs
        nodata = src.nodata

        # node coordinates in the raster CRS, and the window that covers them
        if (input_crs is not None
                and pyproj.CRS.from_user_input(input_crs) != raster_crs):
            transformer = pyproj.Transformer.from_crs(
                input_crs, raster_crs, always_xy=True).transform
            xs, ys = transformer(vert[:, 0], vert[:, 1])
        else:
            xs, ys = np.asarray(vert[:, 0]), np.asarray(vert[:, 1])
        bbox = (float(np.min(xs)), float(np.min(ys)),
                float(np.max(xs)), float(np.max(ys)))
        band, transform, _ = _read_window_decimated(src, bbox, max_cells)
        band = band.astype(np.float64)

    if nodata is not None:
        band = np.where(band == nodata, np.nan, band)
    else:
        band = np.where(~np.isfinite(band), np.nan, band)

    if np.isnan(band).any():
        if nodata_value is None:
            mask = np.isnan(band)
            if mask.all():
                band = np.zeros_like(band)  # nothing valid to fill from
            else:
                _, indices = distance_transform_edt(mask, return_indices=True)
                band = band[tuple(indices)]
        else:
            band = np.where(np.isnan(band), float(nodata_value), band)
    band_filled = band

    inv_transform = ~transform
    cols, rows = inv_transform * (xs, ys)

    mask_inside = (
        (cols >= 0) & (cols < band.shape[1]) &
        (rows >= 0) & (rows < band.shape[0])
    )

    z = np.full_like(np.asarray(xs, dtype=float), np.nan, dtype=float)

    if np.any(mask_inside):
        z[mask_inside] = sign * map_coordinates(
            band_filled,
            [rows[mask_inside], cols[mask_inside]],
            order=order,
            mode=mode,
            cval=cval,
            prefilter=False,
        )

    if np.any(~mask_inside):
        rows_clip = np.clip(rows, 0, band.shape[0] - 1)
        cols_clip = np.clip(cols, 0, band.shape[1] - 1)
        z[~mask_inside] = sign * band_filled[
            rows_clip[~mask_inside].astype(int),
            cols_clip[~mask_inside].astype(int)]

    return z


def interpolate_from_xr(
    ds,
    vert,
    order=3,
    mode="constant",
    cval=np.nan,
    x_name='lon',
    y_name='lat',
    z_name="elevation",
    fill_nan=True,
    handle_out_of_bounds="nearest",
):
    """Interpolate bathymetry from an xarray dataset at mesh node coordinates.

    Parameters
    ----------
    ds : xarray.Dataset
        Dataset with coordinate and elevation variables.
    vert : ndarray of shape (N, 2)
        Node coordinates ``(lon, lat)`` in degrees (EPSG:4326).
    order : int, optional
        Interpolation order (0 = nearest, 1 = bilinear, 3 = bicubic). Default is 3.
    mode : str, optional
        Boundary handling mode for :func:`scipy.ndimage.map_coordinates`.
        Default is ``'constant'``.
    cval : float, optional
        Fill value outside the domain when ``mode='constant'``. Default is
        ``np.nan``.
    x_name, y_name : str, optional
        Names of the longitude and latitude coordinate variables in ``ds``.
    z_name : str, optional
        Name of the elevation variable in ``ds``. Default is ``'elevation'``.
    fill_nan : bool, optional
        If ``True``, fill NaN values in the dataset with the nearest valid
        value. Default is ``True``.
    handle_out_of_bounds : {'nearest', 'nan', 'clip'}, optional
        Strategy for points outside the dataset extent. Default is ``'nearest'``.

    Returns
    -------
    z : ndarray of shape (N,)
        Interpolated depth values at mesh nodes (positive for ocean depth).
    """

    lon = ds[x_name].values
    lat = ds[y_name].values
    band = np.asarray(ds[z_name].values).astype(float)
    
    # Handle NaN values in band (e.g., land mask)
    if fill_nan and np.isnan(band).any():
        mask = np.isnan(band)
        _, indices = distance_transform_edt(mask, return_indices=True)
        band = band[tuple(indices)]

    # Extract coordinates (assumed to be lon/lat)
    xs, ys = vert[:, 0], vert[:, 1]

    # np.interp returns indices even for out-of-bounds values (extrapolates)
    lon_idx = np.interp(xs, lon, np.arange(len(lon)))
    lat_idx = np.interp(ys, lat, np.arange(len(lat)))
    
    # Identify points inside/outside domain
    lon_min, lon_max = lon.min(), lon.max()
    lat_min, lat_max = lat.min(), lat.max()
    
    mask_inside = (
        (xs >= lon_min) & (xs <= lon_max) &
        (ys >= lat_min) & (ys <= lat_max)
    )
    
    # Initialize output array
    z = np.full_like(xs, np.nan, dtype=float)
    
    # Interpolate points inside domain
    if np.any(mask_inside):
        # Clip indices to valid range for map_coordinates
        lat_idx_clip = np.clip(lat_idx[mask_inside], 0, len(lat) - 1)
        lon_idx_clip = np.clip(lon_idx[mask_inside], 0, len(lon) - 1)
        
        z[mask_inside] = -map_coordinates(
            band,
            [lat_idx_clip, lon_idx_clip],
            order=order,
            mode=mode,
            cval=cval,
            prefilter=(order > 1)  # Prefilter only for order > 1 (bicubic)
        )
    
    if np.any(~mask_inside):
        if handle_out_of_bounds == "nearest":
            # Use nearest neighbor for out-of-bounds points
            lat_idx_clip = np.clip(lat_idx[~mask_inside], 0, len(lat) - 1).astype(int)
            lon_idx_clip = np.clip(lon_idx[~mask_inside], 0, len(lon) - 1).astype(int)
            z[~mask_inside] = -band[lat_idx_clip, lon_idx_clip]
        elif handle_out_of_bounds == "clip":
            # Clip coordinates to domain and interpolate
            xs_clip = np.clip(xs[~mask_inside], lon_min, lon_max)
            ys_clip = np.clip(ys[~mask_inside], lat_min, lat_max)
            lon_idx_clip = np.interp(xs_clip, lon, np.arange(len(lon)))
            lat_idx_clip = np.interp(ys_clip, lat, np.arange(len(lat)))
            lat_idx_clip = np.clip(lat_idx_clip, 0, len(lat) - 1)
            lon_idx_clip = np.clip(lon_idx_clip, 0, len(lon) - 1)
            z[~mask_inside] = -map_coordinates(
                band,
                [lat_idx_clip, lon_idx_clip],
                order=order,
                mode=mode,
                cval=cval,
                prefilter=(order > 1)
            )
        # else: "nan" - already initialized with NaN

    return z
