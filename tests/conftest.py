"""Shared pytest configuration for the BlueMesh2D test-suite.

Forces a non-interactive matplotlib backend so the tridemo-derived cases can
be exercised head-less (no windows, no ``plt.show`` blocking), and makes the
repository root importable when the tests are run from a checkout without an
editable install.
"""
import os
import sys

os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib
matplotlib.use("Agg", force=True)

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
