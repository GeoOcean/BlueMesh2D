from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import colors as mcolors
from shapely.geometry import Polygon

from bluemesh2d.geom_util.poly_util import polygon_to_node_edge
from bluemesh2d.refine import refine
from bluemesh2d.smooth import smooth

# --- Geometry parameters -----------------------------------------------------

SIDE = 15.0  # half-width of the triangle base
HEIGHT_RATIO = np.sqrt(2.5)  # summit height, as a multiple of SIDE

WAVE_AMPLITUDE = 0.15 * SIDE
WAVE_PERIOD = SIDE / (2 * np.pi)
WAVE_PHASE = np.pi / 2

WATER_TOP_OFFSET = 0.005 * SIDE  # gap between the triangle base and the water
WATER_THICKNESS = 0.206 * SIDE  # total depth of the water band

# --- Mesh parameters ---------------------------------------------------------

H_FINE = 1.0  # element size at the base of the triangle (and in the water)
H_COARSE = 5.0  # element size at the summit

REFINE_OPTS_WATER = {"kind": "delaunay"}
REFINE_OPTS_TRIANGLE = {"kind": "delfront"}
SMOOTH_OPTS = {"vtol": 1.0e-6, "iter": 128}

# --- Style -------------------------------------------------------------------

COLOR_TRIANGLE = "#33b4f2"
COLOR_WATER = "#efc978"
COLOR_EDGE = "#ffffff"
DPI = 300


def wave_line(n_points: int) -> np.ndarray:
    """Return the ``(n_points, 2)`` polyline of the wave, from left to right."""
    x = np.linspace(-SIDE, SIDE, n_points)
    t = np.linspace(0.0, WAVE_PERIOD, n_points) + WAVE_PHASE
    y = -WAVE_AMPLITUDE * np.sin(t) ** 2 + WAVE_AMPLITUDE / 2
    return np.column_stack((x, y))


def graded_segment(p0, p1, step_start: float, step_end: float) -> np.ndarray:
    """Discretise the segment ``p0 -> p1`` with a linearly graded step size.

    Spacing grows (or shrinks) linearly from ``step_start`` at ``p0`` to
    ``step_end`` at ``p1``. Both end points are included.
    """
    p0 = np.asarray(p0, dtype=float)
    p1 = np.asarray(p1, dtype=float)

    length = np.linalg.norm(p1 - p0)
    if length == 0:
        return p0[None, :]

    n_points = max(2, int(np.ceil(2 * length / (step_start + step_end))) + 1)
    weights = np.linspace(step_start, step_end, n_points - 1)
    params = np.concatenate(([0.0], np.cumsum(weights / weights.sum())))
    return p0 + (p1 - p0) * params[:, None]


def make_depth_hfun(y_ref: np.ndarray, h_ref: np.ndarray):
    """Build a mesh-size function that only depends on the ``y`` coordinate.

    ``y_ref`` must be increasing; sizes are interpolated from ``h_ref`` and
    clamped outside the reference range.
    """
    y_ref = np.asarray(y_ref, dtype=float)
    h_ref = np.asarray(h_ref, dtype=float)

    def hfun(test):
        test = np.asarray(test, dtype=float)
        if test.ndim > 0 and test.shape[-1] == 2:
            test = test[..., 1]
        return np.interp(test, y_ref, h_ref, left=h_ref[0], right=h_ref[-1])

    return hfun


def build_geometry():
    """Return ``(poly_triangle, poly_water, hfun)`` for the logo."""
    wave = wave_line(int(2 * SIDE / H_FINE) + 1)

    # Sides of the triangle, graded from fine at the base to coarse at the top.
    left = graded_segment([-SIDE, 0.0], [0.0, SIDE * HEIGHT_RATIO], H_FINE, H_COARSE)
    right = graded_segment([0.0, SIDE * HEIGHT_RATIO], [SIDE, 0.0], H_COARSE, H_FINE)
    outline = np.vstack((left, right[1:]))

    # Reuse the left-side grading as the vertical mesh-size profile.
    hfun = make_depth_hfun(left[:, 1], np.linspace(H_FINE, H_COARSE, len(left)))

    poly_triangle = Polygon(np.vstack((outline, wave[::-1][1:])))
    poly_water = Polygon(
        np.vstack(
            (
                wave - [0.0, WATER_TOP_OFFSET],
                np.flip(wave - [0.0, WATER_THICKNESS], axis=0),
            )
        )
    )
    return poly_triangle, poly_water, hfun


def mesh_polygon(polygon: Polygon, hfun, refine_opts: dict):
    """Mesh ``polygon`` with the given size function, then smooth it.

    Returns ``(vert, tria)``: the node coordinates and the triangle table.
    """
    node, edge = polygon_to_node_edge(polygon)
    vert, etri, tria, tnum = refine(node, edge, [], refine_opts, hfun)
    vert, etri, tria, tnum = smooth(vert, etri, tria, tnum, SMOOTH_OPTS, hfun)
    return vert, tria


def plot_logo(mesh_triangle, mesh_water):
    """Draw both meshes on a transparent figure and return it."""
    vert_triangle, tria_triangle = mesh_triangle
    vert_water, tria_water = mesh_water

    fig, ax = plt.subplots()
    for (vert, tria), color, linewidth in (
        ((vert_triangle, tria_triangle), COLOR_TRIANGLE, 2),
        ((vert_water, tria_water), COLOR_WATER, 1),
    ):
        ax.tripcolor(
            vert[:, 0],
            vert[:, 1],
            tria,
            np.ones(len(tria)),
            cmap=mcolors.ListedColormap([color]),
            shading="flat",
            edgecolors=COLOR_EDGE,
            linewidth=linewidth,
        )
        ax.triplot(vert[:, 0], vert[:, 1], tria, color=COLOR_EDGE, linewidth=0.4)

    fig.patch.set_alpha(0)
    ax.set_axis_off()
    ax.set_aspect("equal")
    return fig


def main(output_dir: Path = Path(__file__).parent) -> None:
    poly_triangle, poly_water, hfun = build_geometry()

    mesh_triangle = mesh_polygon(poly_triangle, hfun, REFINE_OPTS_TRIANGLE)
    mesh_water = mesh_polygon(poly_water, H_FINE, REFINE_OPTS_WATER)

    fig = plot_logo(mesh_triangle, mesh_water)

    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"mesh_geocean.webp"
    fig.savefig(path, dpi=DPI, bbox_inches="tight", pad_inches=0, transparent=True)
    plt.close(fig)


if __name__ == "__main__":
    main(Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).parent)
