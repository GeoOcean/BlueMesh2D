import numpy as np
import pyproj
from shapely.ops import transform


def get_utm_crs_from_crs(crs):
    """Return a UTM CRS suited to the given geographic CRS."""
    if crs.is_projected:
        return crs
    transformer = pyproj.Transformer.from_crs(crs, "EPSG:4326", always_xy=True)
    lon0, lat0 = transformer.transform(0, 0)
    utm_zone = int((lon0 + 180) / 6) + 1
    epsg_code = 326 if lat0 >= 0 else 327
    return pyproj.CRS.from_epsg(epsg_code * 100 + utm_zone)

def get_local_utm_crs(crs, x=None, y=None, bbox=None):
    """
    Return a local UTM CRS (Transverse Mercator centered on the data).

    If the CRS is already projected, it is returned unchanged.
    Otherwise, the data is converted to WGS84 to compute a center point,
    and a Transverse Mercator projection is built with that central
    meridian and origin latitude (units in meters).

    Parameters
    ----------
    crs : pyproj.CRS
        CRS of the input data (geographic or projected).
    x, y : array-like, optional
        Point coordinates (same size). Ignored if bbox is provided.
    bbox : tuple, optional
        (xmin, ymin, xmax, ymax) in the input CRS.
        Used if (x, y) are not provided.

    Returns
    -------
    pyproj.CRS
        Projected CRS in meters, centered on the data's area.
    """
    crs = pyproj.CRS.from_user_input(crs)
    if crs.is_projected:
        return crs

    # Compute the center in the source CRS
    if bbox is not None:
        xmin, ymin, xmax, ymax = bbox
        x_center = (xmin + xmax) / 2.0
        y_center = (ymin + ymax) / 2.0
    elif x is not None and y is not None:
        x_center = np.nanmean(np.asarray(x))
        y_center = np.nanmean(np.asarray(y))
    else:
        raise ValueError("Provide either (x, y) or bbox.")

    # Convert the center to WGS84
    transformer = pyproj.Transformer.from_crs(crs, "EPSG:4326", always_xy=True)
    lon_center, lat_center = transformer.transform(x_center, y_center)

    # Local Transverse Mercator: central meridian = lon_center, origin latitude = lat_center
    # k=1 at the central meridian, units in meters
    wkt = (
        f'PROJCS["UTM local",'
        f'GEOGCS["WGS 84",DATUM["WGS_1984",SPHEROID["WGS 84",6378137,298.257223563]],'
        f'PRIMEM["Greenwich",0],UNIT["degree",0.0174532925199433]],'
        f'PROJECTION["Transverse_Mercator"],'
        f'PARAMETER["latitude_of_origin",{lat_center}],'
        f'PARAMETER["central_meridian",{lon_center}],'
        f'PARAMETER["scale_factor",1],'
        f'PARAMETER["false_easting",0],'
        f'PARAMETER["false_northing",0],'
        f'UNIT["metre",1]]'
    )
    return pyproj.CRS.from_wkt(wkt)

def get_proj_crs_from_ll(lon0, lat0):
    """
    Create a local Transverse Mercator projection centered on given coordinates.
    
    This projection is designed to match Delft3D-FM's behavior when jsferic=0
    (cartesian/plane coordinates). Delft3D uses a local plane projection for
    distance calculations in flow_geominit.f90, where distances are computed
    as simple Euclidean distances in the projected coordinate system.
    
    The projection uses:
    - Transverse Mercator (tmerc) for minimal distortion near the center
    - Scale factor k=1.0 at the central meridian (better for local accuracy)
    - Centered on the provided lon0, lat0 coordinates
    - Units in meters (matching Delft3D's internal calculations)

    Parameters
    ----------
    lon0 : float
        Longitude of the projection center (degrees).
    lat0 : float
        Latitude of the projection center (degrees).

    Returns
    -------
    pyproj.CRS
        Local Transverse Mercator projection CRS in meters, centered on (lon0, lat0).
        
    Notes
    -----
    This matches Delft3D-FM's approach in flow_geominit.f90 where:
    - When jsferic=0, distances are computed as sqrt(dx² + dy²) in projected coordinates
    - The projection should minimize distortion for the local area of interest
    - Using k=1.0 provides better local accuracy than k=0.9996 (UTM standard)
    
    References
    ----------
    Delft3D-FM: flow_geominit.f90 (line 407)
    geometry_module.f90: dbdistance function (line 370-399)
    """

    # Use k=1.0 for better local accuracy (vs k=0.9996 for UTM)
    # This matches Delft3D's local plane projection behavior
    proj_local = pyproj.CRS.from_proj4(
        f"+proj=tmerc +lat_0={lat0} +lon_0={lon0} +k=1.0 +x_0=0 +y_0=0 +datum=WGS84 +units=m +no_defs"
    )
    return proj_local


def reproject_node(node, crs_from, crs_to):
    """
    Reproject 2D geometry from one CRS to another.

    Parameters
    ----------
    node : ndarray of shape (N, 2)
        Array of vertex coordinates to be reprojected.
    crs_from : pyproj.CRS
        Source coordinate reference system.
    crs_to : pyproj.CRS
        Target coordinate reference system.

    Returns
    -------
    ndarray of shape (N, 2)
        Reprojected vertex coordinates.
    """

    node = np.asarray(node)
    transformer = pyproj.Transformer.from_crs(crs_from, crs_to, always_xy=True)
    x2, y2 = transformer.transform(node[:, 0], node[:, 1])
    return np.column_stack((x2, y2))


def reproject_geometry(geom, crs_from, crs_to):
    """
    Reproject a geometry or coordinate array from one CRS to another.

    Parameters
    ----------
    geom : ndarray of shape (N, 2) or shapely geometry
        Coordinates or geometry to be reprojected.
    crs_from : pyproj.CRS or str
        Source coordinate reference system.
    crs_to : pyproj.CRS or str
        Target coordinate reference system.

    Returns
    -------
    same type as input
        Reprojected geometry or coordinate array.
    """

    transformer = pyproj.Transformer.from_crs(crs_from, crs_to, always_xy=True)
    return transform(transformer.transform, geom).buffer(0)
