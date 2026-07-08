"""
Named constants shared across the ``ortho_merge`` orthogonalization/merge pipeline.
"""

import numpy as np

# Earth radius (WGS84 spherical approximation), aligned with Delft's
# `physicalconsts`. Used for lon/lat <-> local-metric distance conversions.
EARTH_RADIUS = 6378137.0
DEG2RAD = np.pi / 180.0
RAD2DEG = 180.0 / np.pi
EARTH_RADIUS_DEG2RAD = EARTH_RADIUS * DEG2RAD
EARTH_RADIUS_SQ = EARTH_RADIUS * EARTH_RADIUS

# Distance-to-pole tolerance (degrees) below which pole-specific handling
# kicks in for spherical distance/circumcenter computations.
DTOL_POLE = 1.0e-6

# Default "small flow link" threshold (Delft3D-FM convention): an internal
# edge is flagged as a small flow link when the distance between the two
# adjacent triangle circumcenters is below
# `0.9 * threshold * 0.5 * (sqrt(area1) + sqrt(area2))`.
DEFAULT_SMALLLINK_THRESHOLD = 0.11

# Base amplitude of the per-edge orthogonalization displacement inside
# `apply_combined_ortho_smoother_to_zone` (conservative, to avoid overshoot).
DEFAULT_ORTHO_ALPHA = 0.025
