"""Meshing cases used by the tests, mirroring :mod:`bluemesh2d.tridemo`.

Each case reproduces the *geometry and options* of a ``tridemo`` demo but
without any plotting, so it can be run head-less and its numerical output
compared against a stored reference.  Keeping the cases here (rather than
calling ``tridemo`` directly, which draws figures and blocks on
``plt.show``) lets the same builders feed both the reference generator
(:mod:`tests.regenerate_references`) and the assertions in the tests.

A case is a callable returning a ``dict`` with, at minimum, ``vert`` and
``tria``; smoothing cases add the smoothed arrays too.  The dict is what gets
saved to / compared against the ``.npz`` golden files.
"""
import os

import numpy as np

from bluemesh2d.refine import refine
from bluemesh2d.smooth import smooth
from bluemesh2d.smood import smood
from bluemesh2d.triread import triread

POLY_DATA = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "bluemesh2d",
    "poly_data",
)

# Quiet options for the flow-orthogonalisation pass (no progress printing).
_SMOOD_OPTS = {"disp": np.inf}


def _square_with_hole():
    """PSLG of :func:`bluemesh2d.tridemo.demo0` -- square with a square hole."""
    node = np.array(
        [
            [0, 0], [9, 0], [9, 9], [0, 9],   # outer square
            [4, 4], [5, 4], [5, 5], [4, 5],   # inner square (hole)
        ],
        dtype=float,
    )
    edge = (
        np.array(
            [
                [1, 2], [2, 3], [3, 4], [4, 1],   # outer
                [5, 6], [6, 7], [7, 8], [8, 5],   # inner
            ]
        )
        - 1
    )
    return node, edge


def case_demo0_basic():
    """demo0, panel 1: quality-only refinement (no size constraint)."""
    node, edge = _square_with_hole()
    vert, etri, tria, tnum = refine(node, edge, [], {})
    return {"vert": vert, "tria": tria, "tnum": tnum}


def case_demo0_hfun():
    """demo0, panel 2: uniform target edge-length ``hfun = 0.5``."""
    node, edge = _square_with_hole()
    vert, etri, tria, tnum = refine(node, edge, [], {}, 0.5)
    return {"vert": vert, "tria": tria, "tnum": tnum}


def case_demo6_internal():
    """demo6: a square with internal edge constraints, refined then smoothed."""
    node = np.array(
        [
            [-1.0, -1.0], [1.0, -1.0], [1.0, 1.0], [-1.0, 1.0],
            [0.0, 0.0], [0.2, 0.7], [0.6, 0.2], [0.4, 0.8], [0.0, 0.5],
            [-0.7, 0.3], [-0.1, 0.1], [-0.6, 0.5], [-0.9, -0.8],
            [-0.6, -0.7], [-0.3, -0.6], [0.0, -0.5], [0.3, -0.4],
            [-0.3, 0.4], [-0.1, 0.3],
        ]
    )
    edge = np.array(
        [
            [0, 1], [1, 2], [2, 3], [3, 0],
            [4, 5], [4, 6], [4, 7], [4, 8], [4, 9], [4, 10], [4, 11],
            [4, 12], [4, 13], [4, 14], [4, 15], [4, 16], [4, 17], [4, 18],
        ]
    )
    part = [np.array([0, 1, 2, 3])]
    vert, etri, tria, tnum = refine(node, edge, part, {"kind": "delaunay"}, 0.175)
    vnew, enew, tnew, tnum2 = smooth(vert, etri, tria, tnum)
    return {
        "vert": vert, "tria": tria, "tnum": tnum,
        "vert_smooth": vnew, "tria_smooth": tnew, "tnum_smooth": tnum2,
    }


def case_demo9_islands():
    """demo9: real coastal geometry (islands.msh), refined with ``lfshfn``."""
    from bluemesh2d.hfun_util.lfshfn import lfshfn
    from bluemesh2d.hfun_util.trihfn import trihfn
    from bluemesh2d.mesh_util.idxtri import idxtri

    node, edge, _, _ = triread(os.path.join(POLY_DATA, "islands.msh"))
    vlfs, tlfs, hlfs = lfshfn(node, edge)
    slfs = idxtri(vlfs, tlfs)
    vert, etri, tria, tnum = refine(
        node, edge, [], {}, trihfn, vlfs, tlfs, slfs, hlfs
    )
    vnew, enew, tnew, tnum2 = smooth(vert, etri, tria, tnum)
    return {
        "vert": vert, "tria": tria, "tnum": tnum,
        "vert_smooth": vnew, "tria_smooth": tnew, "tnum_smooth": tnum2,
    }


def case_smood_square():
    """Flow-orthogonalisation (smood) of the demo0 ``hfun=0.5`` mesh.

    The four outer-square corners are held fixed, exercising the ``fixed``
    argument used by the QGIS pipeline.
    """
    node, edge = _square_with_hole()
    vert, etri, tria, tnum = refine(node, edge, [], {}, 0.5)
    fixed = np.array([0, 1, 2, 3])  # outer-square corners
    vsm, csm, tsm, nsm = smood(
        vert.copy(), None, tria.copy(), tnum.copy(), _SMOOD_OPTS, fixed=fixed
    )
    return {
        "vert": vert, "tria": tria, "tnum": tnum, "fixed": fixed,
        "vert_smood": vsm, "tria_smood": tsm, "tnum_smood": nsm,
    }


# Cases with a stored golden ``.npz`` (small, exact-regression checks).
CASES = {
    "demo0_basic": case_demo0_basic,
    "demo0_hfun": case_demo0_hfun,
    "demo6_internal": case_demo6_internal,
    "smood_square": case_smood_square,
}

# Large real-geometry case checked for validity only (no golden stored -- the
# mesh has >100k triangles, too heavy to keep in the repo).
SMOKE_CASES = {
    "demo9_islands": case_demo9_islands,
}
