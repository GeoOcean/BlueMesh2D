"""Feedback plumbing shared by the staged meshing pipeline (QGIS-free)."""
from __future__ import annotations

import io
import sys



class MeshCanceled(Exception):
    """Exception raised when a run is cancelled.

    Raised when ``feedback.isCanceled()`` becomes ``True`` mid-run.
    """


class _NullFeedback:
    """No-op feedback so the facade runs without QGIS.

    Writes to ``sys.__stdout__`` (the interpreter's real stdout) rather than
    via ``print``, so it stays safe while ``refine``/``smooth`` output is
    captured by ``contextlib.redirect_stdout`` (printing to the redirected
    stdout would recurse).
    """

    def isCanceled(self):
        return False

    def pushInfo(self, msg):
        sys.__stdout__.write(str(msg) + "\n")

    def pushWarning(self, msg):
        sys.__stdout__.write("WARNING: " + str(msg) + "\n")

    def setProgress(self, pct):
        pass


class _LogWriter(io.TextIOBase):
    """File-like object that forwards captured stdout lines to feedback.

    Parameters
    ----------
    feedback : object
        Feedback sink exposing ``pushInfo(str)``, e.g. a
        ``QgsProcessingFeedback`` or :class:`_NullFeedback`.
    """

    def __init__(self, feedback):
        self._fb = feedback
        self._buf = ""

    def write(self, s):
        self._buf += s
        while "\n" in self._buf:
            line, self._buf = self._buf.split("\n", 1)
            if line.strip():
                self._fb.pushInfo(line)
        return len(s)

    def flush(self):
        if self._buf.strip():
            self._fb.pushInfo(self._buf)
        self._buf = ""


def _check(feedback):
    if feedback.isCanceled():
        raise MeshCanceled()


class _SubProgress:
    """Feedback proxy mapping a sub-task's 0-100 progress into [lo, hi].

    Lets a stage function report absolute progress while a multi-stage
    caller (``generate_mesh``) keeps a monotonic overall bar.
    """

    def __init__(self, feedback, lo, hi):
        self._fb = feedback
        self._lo, self._hi = float(lo), float(hi)

    def isCanceled(self):
        return self._fb.isCanceled()

    def pushInfo(self, msg):
        self._fb.pushInfo(msg)

    def pushWarning(self, msg):
        self._fb.pushWarning(msg)

    def setProgress(self, pct):
        self._fb.setProgress(
            self._lo + (self._hi - self._lo) * float(pct) / 100.0)


def _available_ram_bytes():
    """Available system RAM in bytes, or ``None`` when it cannot be found."""
    try:
        import psutil
        return int(psutil.virtual_memory().available)
    except Exception:
        pass
    try:  # Linux fallback, no psutil needed
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1]) * 1024
    except Exception:
        pass
    return None


def _warn_if_ram_risk(feedback, nbytes, what, hint=None):
    """Warn when an estimated allocation may exhaust the available RAM.

    Parameters
    ----------
    feedback : object
        Feedback sink exposing ``pushWarning``.
    nbytes : float
        Estimated memory need (bytes).
    what : str
        Short description of the allocation, used in the message.
    hint : str or None, optional
        Extra sentence suggesting how to reduce the memory need.

    Returns
    -------
    risky : bool
        ``True`` when a warning was emitted.
    """
    avail = _available_ram_bytes()
    if avail is None or nbytes < 0.5 * avail:
        return False
    msg = (f"{what} needs roughly {nbytes / 1e9:.1f} GB "
           f"(~{avail / 1e9:.1f} GB of RAM available). QGIS may become "
           "unresponsive or crash.")
    if hint:
        msg += " " + hint
    feedback.pushWarning(msg)
    return True

