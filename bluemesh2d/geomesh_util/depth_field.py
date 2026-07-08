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
    """
    Create a callable depth field from a .dat file containing x y z points.
    No projection handling — assumes all coordinates are in the same system.

    Parameters
    ----------
    x, y, z : ndarray
        1D arrays of point coordinates and depth values.
    input_crs : str or pyproj.CRS
        CRS of the input coordinates (e.g. UTM zone). Default 'EPSG:32630'.
    output_crs : str or pyproj.CRS
        CRS in which the depth field will be queried (e.g. 'EPSG:32630' for UTM zone 30N).
    method : {'linear', 'nearest'}, optional
        Interpolation method to use (default 'linear')
    delimiter : str, optional
        Delimiter used in the .dat file (default: auto-detected by numpy)

    Returns
    -------
    depth_field : function
        Callable: depth_field(xy) -> interpolated depth values (m)
        where xy is an array of shape (N, 2) with [x, y] coordinates.
    """

    _check_method(method)

    # --- Clean invalid values
    mask = np.isfinite(x) & np.isfinite(y) & np.isfinite(z)
    x, y, z = x[mask], y[mask], z[mask]

    # --- Create interpolator
    if method == "linear":
        interp = LinearNDInterpolator(list(zip(x, y)), z, fill_value=np.nan)
    else:  # nearest
        interp = NearestNDInterpolator(list(zip(x, y)), z)

    input_crs = pyproj.CRS.from_user_input(input_crs)
    output_crs = pyproj.CRS.from_user_input(output_crs)
    transformer = pyproj.Transformer.from_crs(output_crs, input_crs, always_xy=True)

    # --- Closure function
    def depth_field(xy):
        """
        Returns interpolated depth (z) for given (x, y) coordinates.
        xy : (N, 2) array
        """
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


def depth_field_from_tif(tiff_path, output_crs,raster_crs=None, method="linear"):
    """
    Create a callable depth field from a GeoTIFF bathymetry file.

    Parameters
    ----------
    tiff_path : str
        Path to the bathymetry GeoTIFF file.
    output_crs : str or pyproj.CRS
        CRS of the coordinates passed to the returned depth field (e.g. UTM).
    method : {'linear', 'nearest'}, optional
        'linear': bilinear interpolation (default, smoother, better for hfun).
        'nearest': nearest pixel (fast, piecewise-constant).

    Returns
    -------
    depth_field : callable
        depth_field(xy) -> depth (m), xy shape (N, 2) in output_crs.
    """

    _check_method(method)

    dataset = rasterio.open(tiff_path)
    band = dataset.read(1)
    nodata = dataset.nodata
    transform = dataset.transform
    if raster_crs is None:
        raster_crs = dataset.crs

    output_crs = pyproj.CRS.from_user_input(output_crs)
    raster_crs = pyproj.CRS.from_user_input(raster_crs) if raster_crs else output_crs
    if raster_crs != output_crs:
        transformer = pyproj.Transformer.from_crs(
            output_crs, raster_crs, always_xy=True
        )
    else:
        transformer = None

    # Depth = -elevation; mask nodata for interpolator
    depth_grid = -np.asarray(band, dtype=np.float64)
    if nodata is not None:
        depth_grid = np.where(band == nodata, np.nan, depth_grid)

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
            return np.asarray(depth, dtype=np.float64)
        else:
            # method == "linear": continuous (col, row) from inverse transform
            col_row = np.column_stack(inv_transform * (xs, ys))
            # RegularGridInterpolator expects (row, col) for array [rows, cols]
            row_col = col_row[:, [1, 0]]
            depth = interp(row_col)
            return np.asarray(depth, dtype=np.float64)

    # Expose the raster extent in the query (output) CRS as (xmin, ymin, xmax,
    # ymax), so callers can size a sampling grid without a separate domain.
    if raster_crs != output_crs:
        depth_field.bounds = transform_bounds(raster_crs, output_crs, *dataset.bounds)
    else:
        depth_field.bounds = tuple(dataset.bounds)

    return depth_field


def depth_field_from_xr(ds, input_crs, output_crs, x_name='lon', y_name='lat',
                        z_name="elevation", method="linear"):
    """
    Create a callable depth field from an xarray.Dataset (bathymetry grid),
    reprojecting coordinates from dataset CRS to the desired output CRS.

    Parameters
    ----------
    ds : xarray.Dataset
        Dataset containing bathymetry (e.g. GEBCO subset) with coordinates (lat, lon).
    input_crs : str or pyproj.CRS
        CRS of the dataset coordinates (e.g. 'EPSG:4326' for lat/lon).
    output_crs : str or pyproj.CRS
        CRS in which the depth field will be queried (e.g. 'EPSG:32630' for UTM zone 30N).
    x_name, y_name, z_name : str, optional
        Names of the longitude, latitude and elevation variables in ``ds``.
    method : {'linear', 'nearest'}, optional
        'linear': bilinear interpolation on the grid (default, smoother).
        'nearest': nearest grid node (fast, piecewise-constant).

    Returns
    -------
    depth_field : function
        Callable: depth_field(xy) -> depth values (m)
        where xy is an array of shape (N, 2) with [x, y] coordinates in `output_crs`.
    """

    _check_method(method)

    # -----------------------extract lon/lat grid and data (z assumed [lat, lon])
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

    # -----------------------prepare transformers
    input_crs = pyproj.CRS.from_user_input(input_crs)
    output_crs = pyproj.CRS.from_user_input(output_crs)
    to_ds = pyproj.Transformer.from_crs(output_crs, input_crs, always_xy=True)

    # -----------------------closure function
    def depth_field(xy):
        """
        Returns interpolated depth at given coordinates.
        xy : (N, 2) array in output_crs (e.g., UTM)
        """
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