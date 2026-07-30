"""Recovery-merge option: merge what triangles-only recovery cannot fix.

Triangles-only smood clears small flow links by moving nodes and flipping
edges. When the last few small links sit where it may not act, recovery
stagnates and ``require_both_criteria`` makes the whole run fail. The
``recovery_merge_small_links`` option lets the merge step clean up exactly
those elements.

The stall is provoked with a deliberately large ``smalllink_threshold``
(0.6 instead of the 0.11 default), which makes many ordinary links count as
"small" -- the same dead end as a real mesh with a stubborn one, but small
and quick enough to test.
"""
import numpy as np
import pytest

from bluemesh2d.refine import refine
from bluemesh2d.smood import smood

from tests._cases import _square_with_hole

# Provokes the stall; recovery capped low so the failure comes quickly.
_STALL_OPTS = {
    "disp": np.inf,
    "smalllink_threshold": 0.6,
    "require_both_criteria": True,
    "max_recovery_iterations": 5,
    "recovery_stagnation_break": 3,
}


@pytest.fixture(scope="module")
def stalling_mesh():
    node, edge = _square_with_hole()
    vert, _etri, tria, tnum = refine(node, edge, [], {}, 1.2)
    return vert, tria, tnum


def _run(mesh, **extra):
    vert, tria, tnum = mesh
    opts = dict(_STALL_OPTS)
    opts.update(extra)
    return smood(vert.copy(), None, tria.copy(), tnum.copy(), opts,
                 fixed=np.array([0, 1, 2, 3]))


@pytest.mark.slow
def test_without_recovery_merge_the_run_fails(stalling_mesh):
    """Baseline: the documented failure the option (on by default) prevents."""
    with pytest.raises(RuntimeError, match="still violates dual criteria"):
        _run(stalling_mesh, recovery_merge_small_links=False)


@pytest.mark.slow
@pytest.mark.parametrize("from_iter", [0, 1, 2, None])
def test_recovery_merge_rescues_the_run(stalling_mesh, from_iter):
    """Same mesh, same thresholds: merging the leftovers completes the run.

    Every switch-on point is exercised, plus ``None`` = touch no option at all,
    which must succeed on the defaults (enabled, from recovery cycle 2).
    """
    vert_in, tria_in, _ = stalling_mesh
    extra = {} if from_iter is None else {
        "recovery_merge_small_links": True,
        "recovery_merge_from_iter": from_iter,
    }
    vert, _conn, tria, _tnum = _run(stalling_mesh, **extra)

    # node count is preserved (merging removes faces, never nodes)
    assert len(vert) == len(vert_in)
    # output stays triangle-only: merged quads are re-split on their other
    # diagonal, which is what removes the small link
    assert tria.ndim == 2 and tria.shape[1] == 3
    assert len(tria) == len(tria_in)
    assert tria.min() >= 0 and tria.max() < len(vert)


def test_option_defaults():
    """Enabled by default, from recovery cycle 2 (cheap check)."""
    from bluemesh2d.smood import makeopt_smood

    opts = makeopt_smood({})
    assert opts["recovery_merge_small_links"] is True
    assert opts["recovery_merge_from_iter"] == 2
