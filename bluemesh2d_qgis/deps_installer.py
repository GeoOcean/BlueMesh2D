"""Dependency check + guided pip installation for the BlueMesh2D plugin.

This module must stay importable in *any* state of the environment: it only
imports the standard library and ``qgis.PyQt`` at module level (no numpy, no
shapely, no bundled ``bluemesh2d``), so the plugin can always load far enough
to explain what is missing and offer to install it.

The package lists are duplicated from ``bluemesh2d/dependencies.py`` on
purpose (the installer must work when the bundled package itself is broken);
keep the two in sync.
"""

from __future__ import annotations

import contextlib
import io
import os
import platform
import site
import subprocess
import sys
import sysconfig

# Required at runtime by the meshing pipeline (import name == pip name for
# every entry). Keep in sync with bluemesh2d/dependencies.py.
REQUIRED = ("numpy", "scipy", "shapely", "rasterio", "pyproj",
            "matplotlib", "contourpy", "netCDF4")

# Optional packages, offered as checkboxes in the dialog.
OPTIONAL = {
    "xarray": "needed only by the smood (orthogonalization) option",
    "triangle": "faster, truly constrained Delaunay triangulation",
}
# checkbox state the dialog opens with
OPTIONAL_DEFAULT = {"xarray": True, "triangle": False}


def find_missing(names):
    """Return the sub-list of ``names`` that is not installed.

    Uses ``importlib.util.find_spec`` (module lookup WITHOUT executing the
    module): actually importing freshly installed binary wheels (rasterio,
    shapely, ...) into a running QGIS can hard-crash it, because the wheels
    bundle their own GDAL/GEOS that clash with the libraries QGIS already
    loaded. Presence is what we need here; the real imports happen on the
    next QGIS start.
    """
    import importlib.util

    missing = []
    for name in names:
        try:
            if importlib.util.find_spec(name) is None:
                missing.append(name)
        except Exception:
            missing.append(name)
    return missing


def _venv_dir():
    """Plugin-managed venv location, in the profile's ``python`` dir.

    ``.../profiles/<profile>/python/bluemesh2d_deps`` -- deliberately OUTSIDE
    ``python/plugins/``: the QGIS Plugin Manager treats every folder in
    ``plugins/`` as a plugin and would list the venv as a broken (red) entry.
    Being outside the plugin folder also means it survives plugin upgrades.
    """
    plugins_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(os.path.dirname(plugins_dir), "bluemesh2d_deps")


def _migrate_old_venv():
    """Move a venv created by older plugin versions out of ``plugins/``."""
    import shutil

    plugins_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    old = os.path.join(plugins_dir, "bluemesh2d_deps")
    new = _venv_dir()
    if os.path.isdir(old) and not os.path.isdir(new):
        try:
            shutil.move(old, new)
        except Exception:
            pass  # fall through: a fresh venv will be created at `new`


def _venv_site_packages(venv=None):
    venv = venv or _venv_dir()
    if platform.system() == "Windows":
        return os.path.join(venv, "Lib", "site-packages")
    v = f"python{sys.version_info.major}.{sys.version_info.minor}"
    return os.path.join(venv, "lib", v, "site-packages")


def _activate_user_site():
    """Add the pip ``--user`` site-packages to sys.path (no-op if absent).

    Python only auto-adds the user site at interpreter startup when the
    directory already exists; when pip just created it (first ``--user``
    install ever, common on Windows/OSGeo4W), the freshly installed packages
    are invisible until the next start unless the path is added by hand.
    """
    try:
        us = site.getusersitepackages()
    except Exception:
        return False
    if us and os.path.isdir(us) and us not in sys.path:
        site.addsitedir(us)
        return True
    return False


def activate_venv():
    """Make plugin-installed packages importable (no-op when absent).

    Called at plugin load: adds the plugin venv's site-packages (PEP 668
    systems) and the pip ``--user`` site directory to sys.path.
    """
    _migrate_old_venv()
    _activate_user_site()
    sp = _venv_site_packages()
    if os.path.isdir(sp) and sp not in sys.path:
        site.addsitedir(sp)
        return True
    return False


def _venv_python(venv):
    return os.path.join(venv, "Scripts" if platform.system() == "Windows"
                        else "bin", "python")


def _venv_has_pip(venv):
    py = _venv_python(venv)
    if not os.path.exists(py):
        return False
    try:
        return subprocess.run([py, "-m", "pip", "--version"],
                              capture_output=True, timeout=120
                              ).returncode == 0
    except Exception:
        return False


def _bootstrap_pip(venv, log):
    """Install pip into the venv with the official get-pip.py bootstrap.

    Needed on Debian/Ubuntu when neither ``python3-pip`` nor ``python3-venv``
    (which provides ensurepip) is installed: the venv can still be created
    without pip, and get-pip.py adds pip to it -- no root, no apt.
    """
    import tempfile
    import urllib.request

    url = "https://bootstrap.pypa.io/get-pip.py"
    log.append(f"pip is missing: bootstrapping it from {url} ...")
    try:
        with urllib.request.urlopen(url, timeout=60) as resp:
            script = resp.read()
        with tempfile.NamedTemporaryFile("wb", suffix="_get-pip.py",
                                         delete=False) as f:
            f.write(script)
            path = f.name
        proc = subprocess.run([_venv_python(venv), path],
                              capture_output=True, text=True, timeout=600)
        os.unlink(path)
        log.append(proc.stdout[-2000:])
        if proc.returncode != 0:
            log.append(proc.stderr[-2000:])
            return False
        return True
    except Exception as exc:
        log.append(f"pip bootstrap failed: {exc}")
        return False


def _create_venv(log):
    """Create the plugin venv (with pip) seeing QGIS's system packages.

    The venv is ALWAYS created without pip first: Debian's patched ``venv``
    module calls ``sys.exit(1)`` when ensurepip is unavailable, and running
    ``with_pip=True`` in-process would terminate QGIS itself. pip is then
    added from a subprocess (isolated, cannot kill QGIS): ``ensurepip`` when
    available, the official get-pip.py otherwise.
    """
    import venv as venv_mod

    _migrate_old_venv()
    venv = _venv_dir()
    if _venv_has_pip(venv):
        return venv
    log.append(f"Creating a dedicated environment: {venv}")
    try:
        venv_mod.EnvBuilder(system_site_packages=True,
                            with_pip=False).create(venv)
    except BaseException as exc:  # incl. SystemExit from patched builders
        log.append(f"venv creation failed: {exc}")
        return None
    py = _venv_python(venv)
    try:
        proc = subprocess.run(
            [py, "-Im", "ensurepip", "--upgrade", "--default-pip"],
            capture_output=True, text=True, timeout=600)
        if proc.returncode == 0 and _venv_has_pip(venv):
            return venv
        log.append("ensurepip unavailable; using get-pip.py instead ...")
    except Exception as exc:
        log.append(f"ensurepip failed ({exc}); using get-pip.py instead ...")
    if _bootstrap_pip(venv, log):
        return venv
    return None


def _venv_pip_install(packages, log):
    """Install into the plugin venv with its own interpreter (Linux path)."""
    venv = _create_venv(log)
    if venv is None:
        return False
    py = _venv_python(venv)
    try:
        proc = subprocess.run(
            [py, "-m", "pip", "install", *packages],
            capture_output=True, text=True, timeout=1800)
        log.append(proc.stdout)
        if proc.stderr:
            log.append(proc.stderr)
        if proc.returncode != 0:
            return False
    except Exception as exc:
        log.append(f"venv pip failed: {exc}")
        return False
    activate_venv()
    return True


def _is_externally_managed():
    """True on PEP 668 interpreters (Debian/Ubuntu system Python)."""
    for key in ("stdlib", "platstdlib"):
        try:
            marker = os.path.join(sysconfig.get_path(key),
                                  "EXTERNALLY-MANAGED")
            if os.path.exists(marker):
                return True
        except Exception:
            pass
    return False


def _is_conda():
    """True when QGIS's Python runs inside a real conda environment.

    Only ``conda-meta/`` inside ``sys.prefix`` counts: the official macOS
    QGIS bundles are *built from* conda-forge packages (``sys.version`` says
    "packaged by conda-forge") without being a managed conda environment --
    there pip is the right installer. ``CONDA_PREFIX`` alone is also not
    trusted: it leaks into QGIS when the app is launched from a shell that
    happens to have a conda env activated.
    """
    return os.path.isdir(os.path.join(sys.prefix, "conda-meta"))


def pip_args(packages, force_break=False):
    """Build the in-process ``pip.main`` argument list for this platform.

    Returns ``None`` for conda-based QGIS installs, where pip would fight the
    environment: :func:`manual_command` then shows the conda command instead.
    On PEP 668 systems the venv route (:func:`_venv_pip_install`) is tried
    first; ``force_break`` builds the last-resort ``--break-system-packages``
    variant used only when venv creation fails.
    """
    if _is_conda():
        return None
    args = ["install", "--user"]
    if force_break and _is_externally_managed():
        args.append("--break-system-packages")
    return args + list(packages)


def manual_command(packages):
    """The per-OS command a user can run by hand if the dialog's pip fails."""
    pkgs = " ".join(packages)
    if _is_conda():
        return f"conda install -c conda-forge {pkgs.lower()}"
    system = platform.system()
    if system == "Windows":
        return f"python -m pip install {pkgs}   (in the OSGeo4W Shell)"
    if system == "Darwin":
        arglist = ", ".join(f"'{p}'" for p in packages)
        return (f"import pip; pip.main(['install', '--user', {arglist}])"
                "   (in the QGIS Python console)")
    if _is_externally_managed():
        apt = " ".join("python3-" + p.lower() for p in packages)
        return (f"{_venv_dir()}/bin/python -m pip install {pkgs}\n"
                f"  (or system packages: sudo apt install {apt})")
    return f"python3 -m pip install --user {pkgs}"


def _numpy_constraint():
    """Pin numpy to the environment's current major version, if any.

    QGIS bundles C extensions (matplotlib, ...) compiled against its own
    numpy; letting pip pull a numpy with a different MAJOR version into the
    user site shadows the bundled one and breaks every compiled module with
    '_ARRAY_API not found'. Reads the version from package metadata only --
    importing numpy here could itself trigger the clash.
    """
    try:
        from importlib.metadata import version
        major = int(version("numpy").split(".")[0])
        return f"numpy<{major + 1}"
    except Exception:
        return None


def run_pip(packages, log=None):
    """Install ``packages`` with the strategy fitting this platform.

    - conda QGIS: refuse and show the conda command.
    - PEP 668 (Debian/Ubuntu system Python): install into a plugin-managed
      venv created with ``--system-site-packages`` (no
      ``--break-system-packages`` needed); fall back to in-process pip with
      that flag only if the venv cannot be created.
    - everywhere else: in-process ``pip.main --user`` (the only invocation
      that works on every QGIS bundle -- macOS vcpkg builds ship a python
      executable that cannot run standalone).

    Every install carries a numpy major-version pin matching the packages
    QGIS already ships (see :func:`_numpy_constraint`).

    Returns ``(ok, output)``.
    """
    constraint = _numpy_constraint()
    if constraint:
        packages = list(packages) + [constraint]

    if _is_conda():
        return False, ("This QGIS runs on a conda Python; install the "
                       "packages with conda instead:\n  "
                       + manual_command(packages))

    if _is_externally_managed():
        lines = []
        if _venv_pip_install(packages, lines):
            lines.append("\nInstalled into the plugin environment "
                         f"({_venv_dir()}) -- the system Python was not "
                         "touched.")
            return True, "\n".join(lines)
        lines.append("\nFalling back to --break-system-packages ...")
        ok, out = _run_pip_inprocess(pip_args(packages, force_break=True))
        if not ok:
            out += ("\n\nIf pip/venv are unavailable on this system, install "
                    "them once with:\n  sudo apt install python3-pip "
                    "python3-venv\nthen click 'Install now' again.")
        return ok, "\n".join(lines) + "\n" + out

    ok, out = _run_pip_inprocess(pip_args(packages))
    if ok:
        # a first-ever --user install creates the user site dir mid-session;
        # put it on sys.path so the post-install check can see the packages
        _activate_user_site()
    return ok, out


def _run_pip_inprocess(args):
    """Run pip in this interpreter, capturing its output."""
    buf = io.StringIO()
    try:
        import pip  # noqa: F401
        from pip._internal.cli.main import main as pip_main
    except Exception:
        try:
            from pip import main as pip_main  # very old pip
        except Exception as exc:
            return False, f"pip is not available in this QGIS Python: {exc}"
    try:
        with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
            rc = pip_main(args)
    except SystemExit as exc:  # some pip versions sys.exit()
        rc = int(exc.code or 0)
    except Exception as exc:
        return False, buf.getvalue() + f"\npip crashed: {exc}"
    return rc == 0, buf.getvalue()


class DepsDialog:
    """Qt dialog listing missing packages with an 'Install now' button.

    Wrapped in a plain class (not a QDialog subclass at module level) so this
    module can be imported and unit-tested without Qt.
    """

    def __init__(self, parent=None):
        from qgis.PyQt.QtWidgets import (
            QCheckBox, QDialog, QHBoxLayout, QLabel, QPlainTextEdit,
            QPushButton, QVBoxLayout,
        )

        self._missing_required = find_missing(REQUIRED)
        self._missing_optional = find_missing(list(OPTIONAL))

        self.dialog = QDialog(parent)
        self.dialog.setWindowTitle("BlueMesh2D — Python dependencies")
        self.dialog.setMinimumWidth(560)
        lay = QVBoxLayout(self.dialog)

        if self._missing_required:
            lay.addWidget(QLabel(
                "<b>BlueMesh2D needs these Python packages "
                "(not found in this QGIS):</b><br>"
                + ", ".join(self._missing_required)))
        else:
            lay.addWidget(QLabel(
                "<b>All required packages are installed.</b>"))

        self._boxes = {}
        if self._missing_optional:
            lay.addWidget(QLabel("Optional packages:"))
            for name in self._missing_optional:
                box = QCheckBox(f"{name} — {OPTIONAL[name]}")
                box.setChecked(OPTIONAL_DEFAULT.get(name, False))
                self._boxes[name] = box
                lay.addWidget(box)

        self.log = QPlainTextEdit()
        self.log.setReadOnly(True)
        self.log.setMinimumHeight(160)
        self.log.setPlaceholderText(
            "pip output will appear here.\n\nManual fallback command:\n  "
            + manual_command(self._missing_required or list(REQUIRED)))
        lay.addWidget(self.log)

        btns = QHBoxLayout()
        self.install_btn = QPushButton("Install now")
        self.close_btn = QPushButton("Close")
        btns.addStretch(1)
        btns.addWidget(self.install_btn)
        btns.addWidget(self.close_btn)
        lay.addLayout(btns)

        self.install_btn.clicked.connect(self._install)
        self.close_btn.clicked.connect(self.dialog.reject)
        if not self._missing_required and not self._missing_optional:
            self.install_btn.setEnabled(False)

    def _selected_packages(self):
        pkgs = list(self._missing_required)
        pkgs += [n for n, b in self._boxes.items() if b.isChecked()]
        return pkgs

    def _install(self):
        from qgis.PyQt.QtCore import Qt
        from qgis.PyQt.QtWidgets import QApplication

        pkgs = self._selected_packages()
        if not pkgs:
            self.log.setPlainText("Nothing selected to install.")
            return
        self.log.setPlainText(f"Installing: {', '.join(pkgs)} ...\n")
        QApplication.setOverrideCursor(Qt.WaitCursor)
        QApplication.processEvents()
        try:
            ok, out = run_pip(pkgs)
        finally:
            QApplication.restoreOverrideCursor()
        self.log.appendPlainText(out)
        still = find_missing(pkgs)
        if ok and not still:
            self.log.appendPlainText(
                "\nDone. Please RESTART QGIS to activate BlueMesh2D.")
            self.install_btn.setEnabled(False)
        else:
            self.log.appendPlainText(
                "\nInstallation did not complete"
                + (f" (still missing: {', '.join(still)})." if still else ".")
                + "\nManual fallback:\n  " + manual_command(still or pkgs))

    def exec(self):
        # QGIS shows an app-wide busy cursor while plugins load; without an
        # explicit arrow override the dialog inherits the spinning cursor
        # for its whole lifetime.
        from qgis.PyQt.QtCore import Qt
        from qgis.PyQt.QtWidgets import QApplication

        QApplication.setOverrideCursor(Qt.ArrowCursor)
        try:
            return self.dialog.exec()
        finally:
            QApplication.restoreOverrideCursor()


def ensure_dependencies(parent=None):
    """Return True when all required packages import; else show the dialog."""
    if not find_missing(REQUIRED):
        return True
    DepsDialog(parent).exec()
    return not find_missing(REQUIRED)
