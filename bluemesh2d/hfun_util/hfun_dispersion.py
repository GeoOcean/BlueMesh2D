import numpy as np


def hfun_wavenumhunt(test, depth_field, T, N, zmin, hmin):
    """Build a mesh-size function from local wavelength.

    Parameters
    ----------
    test : ndarray of shape (N, 2)
        Coordinates where the mesh size is evaluated.
    depth_field : callable or array_like
        Water depth at each query point; a scalar or length-``N`` array.
    T : float
        Wave period in seconds.
    N : int
        Target number of cells per wavelength.
    zmin : float
        Minimum water depth used in the dispersion relation.
    hmin : float
        Minimum allowed mesh size.

    Returns
    -------
    hfun : ndarray of shape (N,)
        Target cell size at each query point.
    """
    # Determine local depth
    if callable(depth_field):
        h = depth_field(test)
    else:
        h = np.asarray(depth_field)
        if h.size == 1:
            h = np.full(test.shape[0], h)
        elif h.shape[0] != test.shape[0]:
            raise ValueError("depth_field must match number of test points")

    # Compute wavelength
    h = np.maximum(h, zmin)
    L, _ = wavenumhunt(T, h)

    # Mesh size proportional to wavelength
    hfun = L / N

    # Enforce minimum mesh size
    hfun = np.maximum(hfun, hmin)

    return hfun


def wavenumhunt(T, h):
    """Compute wavelength and wavenumber via the Hunt (1979) approximation.

    Parameters
    ----------
    T : float or array_like
        Wave period in seconds.
    h : float or array_like
        Water depth in metres.

    Returns
    -------
    L : float or ndarray
        Wavelength in metres.
    k : float or ndarray
        Wavenumber in 1/m.
    """
    D = np.array(
        [
            0.6666666666,
            0.3555555555,
            0.1608465608,
            0.0632098765,
            0.0217540484,
            0.0065407983,
        ]
    )

    L0 = (9.81 * T**2) / (2 * np.pi)
    k0 = 2 * np.pi / L0
    k0h = k0 * h

    # Approximation of kh following Hunt (1979)
    poly = (
        D[0] * k0h**1
        + D[1] * k0h**2
        + D[2] * k0h**3
        + D[3] * k0h**4
        + D[4] * k0h**5
        + D[5] * k0h**6
    )
    kh = k0h * np.sqrt(1 + (k0h * (1 + poly)) ** -1)

    k = kh / h
    L = 2 * np.pi / k

    return L, k
