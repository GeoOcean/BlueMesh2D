import numpy as np
import pyproj
import rasterio
from scipy.interpolate import LinearNDInterpolator, RBFInterpolator
from scipy.ndimage import map_coordinates, distance_transform_edt
from scipy.spatial import cKDTree


def interpolate_from_xyz(
    x, y, z,
    vert,
    method="rbf",
    rbf_function="cubic",
    epsilon=None,
):
    """
    Interpolation of values from scattered (x, y, z, value) points at arbitrary 3D nodes.

    Parameters
    ----------
    x, y, z : (N,) ndarray
        Coordinates of the scattered data points.
    vert : (N, 3) ndarray
        Target coordinates (x, y, z) where interpolation is evaluated.
    method : {'linear', 'nearest', 'rbf'}, optional
        Interpolation method:
        - 'linear' : piecewise linear interpolation using tetrahedra (scipy.LinearNDInterpolator)
        - 'nearest': nearest-neighbor interpolation using KDTree
        - 'rbf'    : smooth radial basis interpolation (scipy.RBFInterpolator)
    delimiter : str, optional
        Column delimiter in the file (default auto).
    rbf_function : str, optional
        RBF kernel function ('multiquadric', 'inverse', 'gaussian', 'thin_plate_spline', etc.)
        Used only if method='rbf'.
    epsilon : float, optional
        Shape parameter for RBF interpolation. Auto-estimated if None.

    Returns
    -------
    values_interp : (N,) ndarray
        Interpolated values at the target 3D points.
    """

    # --- Remove invalid points
    mask = np.isfinite(x) & np.isfinite(y) & np.isfinite(z)
    x, y, z = x[mask], y[mask], z[mask]
    points = np.column_stack((x, y))

    # --- Interpolation method
    if method == "linear":
        interp = LinearNDInterpolator(points, z, fill_value=np.nan)
        values_interp = interp(vert[:, 0], vert[:, 1], vert[:, 2])

    elif method == "nearest":
        tree = cKDTree(points)
        _, idx = tree.query(vert)
        values_interp = z[idx]

    elif method == "rbf":
        interp = RBFInterpolator(points, z, kernel=rbf_function, epsilon=epsilon)
        values_interp = interp(vert)

    else:
        raise ValueError("method must be 'linear', 'nearest', or 'rbf'")

    # --- Handle NaNs
    values_interp = - np.asarray(values_interp, dtype=float)
    values_interp[np.isnan(values_interp)] = 0

    return values_interp


def interpolate_from_tiff(
    tiff_path, vert, input_crs=None, order=3, mode="constant", cval=np.nan
):
    """
    Fast interpolation of GeoTIFF values at mesh nodes (bicubic/bilinear).

    Parameters
    ----------
    tiff_path : str
        Path to GeoTIFF file.
    vert : (N, 2) array
        Node coordinates (x, y) in input CRS.
    input_crs : str or pyproj.CRS, optional
        CRS of input mesh. If None, assumes same as raster.
    order : int
        Interpolation order (0=nearest, 1=bilinear, 3=bicubic).
    mode : str
        Boundary handling mode for map_coordinates.
    cval : float
        Constant value outside domain if mode="constant".

    Returns
    -------
    z : (N,) array
        Interpolated raster values at mesh nodes.
    """
    with rasterio.open(tiff_path) as src:
        band = src.read(1).astype(np.float64)
        transform = src.transform
        raster_crs = src.crs
        nodata = src.nodata

        if nodata is not None:
            band = np.where(band == nodata, np.nan, band)
        else:
            band = np.where(~np.isfinite(band), np.nan, band)

        if np.isnan(band).any():
            mask = np.isnan(band)
            _, indices = distance_transform_edt(mask, return_indices=True)
            band_filled = band[tuple(indices)]
        else:
            band_filled = band

        if (
            input_crs is not None
            and pyproj.CRS.from_user_input(input_crs) != raster_crs
        ):
            transformer = pyproj.Transformer.from_crs(
                input_crs, raster_crs, always_xy=True
            ).transform
            xs, ys = transformer(vert[:, 0], vert[:, 1])
        else:
            xs, ys = vert[:, 0], vert[:, 1]

        inv_transform = ~transform
        cols, rows = inv_transform * (xs, ys)

        mask_inside = (
            (cols >= 0) & (cols < band.shape[1]) &
            (rows >= 0) & (rows < band.shape[0])
        )

        z = np.full_like(xs, np.nan, dtype=float)

        if np.any(mask_inside):
            z[mask_inside] = -map_coordinates(
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
            z[~mask_inside] = -band_filled[rows_clip[~mask_inside].astype(int),
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
    """
    Fast interpolation of bathymetry values from an xarray.Dataset (e.g. GEBCO)
    at given mesh nodes (bicubic/bilinear).

    Parameters
    ----------
    ds : xarray.Dataset
        Dataset with coordinates 'lat' and 'lon' and variable `var_name`.
    vert : (N, 2) array
        Node coordinates (lon, lat) in degrees (EPSG:4326).
    order : int, optional
        Interpolation order (0=nearest, 1=bilinear, 3=bicubic). Default 3.
    mode : str, optional
        Boundary handling mode for map_coordinates. Default "constant".
    cval : float, optional
        Constant value outside domain if mode="constant". Default np.nan.
    var_name : str, optional
        Name of the variable in `ds` containing elevation values.
    fill_nan : bool, optional
        If True, fill NaN values in dataset with nearest valid value. Default True.
    handle_out_of_bounds : str, optional
        How to handle out-of-bounds points: "nearest" (use nearest neighbor),
        "nan" (return NaN), or "clip" (clip to domain bounds). Default "nearest".

    Returns
    -------
    z : (N,) array
        Interpolated depth values at mesh nodes (positive for ocean depth).
    """

    # -----------------------extract coordinates and data
    lon = ds[x_name].values
    lat = ds[y_name].values
    band = np.asarray(ds[z_name].values).astype(float)
    
    # Handle NaN values in band (e.g., land mask)
    if fill_nan and np.isnan(band).any():
        mask = np.isnan(band)
        _, indices = distance_transform_edt(mask, return_indices=True)
        band = band[tuple(indices)]

    # -----------------------extract coordinates (assumed to be lon/lat)
    xs, ys = vert[:, 0], vert[:, 1]

    # -----------------------compute pixel indices
    # np.interp returns indices even for out-of-bounds values (extrapolates)
    lon_idx = np.interp(xs, lon, np.arange(len(lon)))
    lat_idx = np.interp(ys, lat, np.arange(len(lat)))
    
    # -----------------------identify points inside/outside domain
    lon_min, lon_max = lon.min(), lon.max()
    lat_min, lat_max = lat.min(), lat.max()
    
    mask_inside = (
        (xs >= lon_min) & (xs <= lon_max) &
        (ys >= lat_min) & (ys <= lat_max)
    )
    
    # Initialize output array
    z = np.full_like(xs, np.nan, dtype=float)
    
    # -----------------------interpolate points inside domain
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
    
    # -----------------------handle out-of-bounds points
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
