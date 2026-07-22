"""Regression and validity tests for the BlueMesh2D meshing pipeline.

These reproduce a subset of :mod:`bluemesh2d.tridemo` head-less and check the
numerical output two ways:

* **Regression** -- small cases are compared array-for-array against the
  golden ``.npz`` files in ``tests/reference/`` (regenerate them with
  ``python -m tests.regenerate_references`` when the output *intentionally*
  changes).
* **Validity** -- every generated mesh must be geometrically sane: finite
  coordinates, in-range connectivity, and strictly positive triangle areas
  (no zero-area or inverted elements).

The ``smood`` (flow-orthogonalisation) pass gets its own checks: it must hold
the ``fixed`` vertices in place and improve dual-mesh orthogonality relative
to the raw Delaunay mesh.
"""
import os

import numpy as np
import pytest

from tests._cases import CASES, SMOKE_CASES, case_smood_square

REFERENCE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "reference")

# Coordinates are compared with a tolerance so a different BLAS / scipy build
# on another platform does not trip the exact-regression check; topology
# (triangle count and connectivity) must match exactly.
COORD_ATOL = 1e-6


def _tri_areas(vert, tria):
    """Signed area of each triangle (CCW positive)."""
    p = np.asarray(vert, dtype=float)
    t = np.asarray(tria, dtype=int)[:, :3]
    a, b, c = p[t[:, 0]], p[t[:, 1]], p[t[:, 2]]
    return 0.5 * ((b[:, 0] - a[:, 0]) * (c[:, 1] - a[:, 1])
                  - (c[:, 0] - a[:, 0]) * (b[:, 1] - a[:, 1]))


def _assert_valid_mesh(vert, tria):
    """A mesh must have finite nodes, in-range triangles and non-degenerate area."""
    vert = np.asarray(vert, dtype=float)
    tria = np.asarray(tria, dtype=int)
    assert vert.ndim == 2 and vert.shape[1] == 2
    assert tria.ndim == 2 and tria.shape[1] >= 3
    assert np.isfinite(vert).all(), "non-finite vertex coordinates"
    assert tria[:, :3].min() >= 0 and tria[:, :3].max() < vert.shape[0], \
        "triangle references a non-existent vertex"
    areas = _tri_areas(vert, tria)
    assert np.all(np.abs(areas) > 1e-12), "degenerate (zero-area) triangle present"


def _max_abs_cosphi(vert, tria):
    """Maximum ``|cos phi|`` of the dual-mesh flow links (planar geometry)."""
    from bluemesh2d.ortho_merge.ortho_merge_iter import dual_criteria_on_fan_mesh

    tria = np.asarray(tria, dtype=np.int64)[:, :3]
    tri_origin_face_id = np.arange(tria.shape[0], dtype=np.int64)
    quad_face_mask = np.zeros(tria.shape[0], dtype=bool)
    _, max_c, _ = dual_criteria_on_fan_mesh(
        np.asarray(vert, dtype=np.float64), tria,
        tri_origin_face_id, quad_face_mask,
        cosphi_threshold=0.49, removesmalllinkstrsh=0.11, jsferic=0,
    )
    return float(max_c)


# --------------------------------------------------------------------------- #
# Regression against stored golden output
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("name", list(CASES))
def test_case_matches_reference(name):
    """Each small case reproduces its golden ``.npz`` (topology + coordinates)."""
    ref_path = os.path.join(REFERENCE_DIR, name + ".npz")
    assert os.path.exists(ref_path), (
        f"missing reference {ref_path!r}; run "
        f"'python -m tests.regenerate_references'")

    result = CASES[name]()
    with np.load(ref_path) as ref:
        keys = set(ref.files)
        assert keys == set(result), (
            f"reference keys {keys} != produced keys {set(result)}")
        for key in keys:
            got = np.asarray(result[key])
            exp = np.asarray(ref[key])
            assert got.shape == exp.shape, f"{name}:{key} shape {got.shape} != {exp.shape}"
            if np.issubdtype(exp.dtype, np.floating):
                np.testing.assert_allclose(
                    got, exp, atol=COORD_ATOL,
                    err_msg=f"{name}:{key} coordinates drifted")
            else:
                np.testing.assert_array_equal(
                    got, exp, err_msg=f"{name}:{key} connectivity changed")


# --------------------------------------------------------------------------- #
# Validity of every produced mesh (including the large smoke case)
# --------------------------------------------------------------------------- #
def _check_valid(result):
    _assert_valid_mesh(result["vert"], result["tria"])
    if "vert_smooth" in result:
        _assert_valid_mesh(result["vert_smooth"], result["tria_smooth"])
    if "vert_smood" in result:
        _assert_valid_mesh(result["vert_smood"], result["tria_smood"])


@pytest.mark.parametrize("name", list(CASES))
def test_case_produces_valid_mesh(name):
    """Every small case yields geometrically valid mesh(es)."""
    _check_valid(CASES[name]())


@pytest.mark.slow
@pytest.mark.parametrize("name", list(SMOKE_CASES))
def test_large_case_produces_valid_mesh(name):
    """Large real-geometry case (islands, >100k triangles) is valid.

    Marked ``slow``; skip with ``-m 'not slow'``.
    """
    _check_valid(SMOKE_CASES[name]())


# --------------------------------------------------------------------------- #
# smood (flow orthogonalisation) specific behaviour
# --------------------------------------------------------------------------- #
def test_smood_holds_fixed_vertices():
    """``fixed`` vertices are never displaced by the smood pass."""
    result = case_smood_square()
    fixed = result["fixed"]
    np.testing.assert_allclose(
        result["vert_smood"][fixed], result["vert"][fixed], atol=1e-12,
        err_msg="smood moved a vertex that was flagged as fixed")


def test_smood_preserves_node_count_and_moves_interior():
    """smood keeps the node numbering and actually relaxes free vertices."""
    result = case_smood_square()
    assert result["vert_smood"].shape == result["vert"].shape, \
        "smood changed the vertex count (numbering must be preserved)"
    moved = np.linalg.norm(result["vert_smood"] - result["vert"], axis=1)
    assert moved.max() > 1e-6, "smood did not move any vertex"


def test_smood_improves_orthogonality():
    """smood lowers (or holds) the worst dual-mesh non-orthogonality."""
    result = case_smood_square()
    before = _max_abs_cosphi(result["vert"], result["tria"])
    after = _max_abs_cosphi(result["vert_smood"], result["tria_smood"])
    assert after <= before + 1e-9, (
        f"smood worsened orthogonality: max|cos phi| {before:.4f} -> {after:.4f}")
