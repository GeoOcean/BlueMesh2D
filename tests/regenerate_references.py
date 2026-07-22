"""Regenerate the golden reference ``.npz`` files under ``tests/reference/``.

Run this whenever the meshing algorithms *intentionally* change output::

    python -m tests.regenerate_references          # from the repo root
    python tests/regenerate_references.py

Each case in :data:`tests._cases.CASES` is executed and its result dict is
saved as ``tests/reference/<name>.npz``.  The regression tests
(:mod:`tests.test_meshing`) then re-run the same cases and compare against
these files.  Review the diff (e.g. triangle counts) before committing new
references.
"""
import os

import numpy as np

# Support both "python -m tests.regenerate_references" and direct execution.
try:
    from tests._cases import CASES
except ImportError:  # run as a script from inside tests/
    import sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from tests._cases import CASES

REFERENCE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "reference")


def main():
    os.makedirs(REFERENCE_DIR, exist_ok=True)
    for name, builder in CASES.items():
        result = builder()
        path = os.path.join(REFERENCE_DIR, name + ".npz")
        np.savez_compressed(path, **result)
        ntri = np.asarray(result["tria"]).shape[0]
        print(f"  wrote {name}.npz  (|TRIA|={ntri})")
    print(f"Done. {len(CASES)} reference file(s) in {REFERENCE_DIR}")


if __name__ == "__main__":
    main()
