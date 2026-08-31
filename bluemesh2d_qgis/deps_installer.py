"""Dependency check + guided pip installation for the BlueMesh2D plugin.

This module must stay importable in *any* state of the environment: it only
imports the standard library and ``qgis.PyQt`` at module level (no numpy, no
shapely, no ``bluemesh2d``), so the plugin can always load far enough to
explain what is missing and offer to install it.

The plugin ships no copy of the library: it installs the published
``bluemesh2d`` distribution from PyPI, which pulls numpy, scipy, shapely,
rasterio, matplotlib, netCDF4, xarray and triangle in through its own
metadata. Developers working from a checkout can install that checkout
editable instead -- see :func:`source_checkout`.
"""

from __future__ import annotations

import contextlib
import io
import os
import platform
import site
import subprocess  # nosec B404 -- fixed argv, shell=False; see _venv_has_pip
import sys
import sysconfig

# Oldest bluemesh2d release this plugin version works against.
MIN_VERSION = "0.1.4"

# Queried to tell the user whether their bluemesh2d is the latest release.
# Only ever contacted from the dependency dialog (never on plugin load), and
# every failure is swallowed: the check is informational, and QGIS must keep
# working offline, behind a proxy or when PyPI is down.
PYPI_JSON_URL = "https://pypi.org/pypi/{dist}/json"
PYPI_TIMEOUT = 3.0

# What pip installs. Everything the pipeline needs comes from bluemesh2d's own
# dependency metadata, EXCEPT pyproj: the library imports it but does not
# declare it, and nothing else in the tree pulls it in, so it stays explicit.
PIP_REQUIRED = (f"bluemesh2d>={MIN_VERSION}", "pyproj")

# Conda-forge names of the dependency stack. Only used to spell out the
# manual command on conda-based QGIS installs (see manual_command): there is
# no conda package for bluemesh2d itself, and pip must not drag binary wheels
# into a conda environment.
CONDA_PACKAGES = ("numpy", "scipy", "shapely", "rasterio", "pyproj",
                  "matplotlib", "netcdf4", "xarray")

# What is checked for presence, as import names. `bluemesh2d` is a PEP 420
# namespace package, so a bare `bluemesh2d` lookup succeeds for any stray
# directory of that name on sys.path -- check a real submodule instead.
REQUIRED = ("bluemesh2d.pipeline", "pyproj")


def find_missing(names):
    """Return the sub-list of ``names`` that is not installed.

    Uses ``importlib.util.find_spec`` (module lookup WITHOUT executing the
    module): actually importing freshly installed binary wheels (rasterio,
    shapely, ...) into a running QGIS can hard-crash it, because the wheels
    bundle their own GDAL/GEOS that clash with the libraries QGIS already
    loaded. Presence is what we need here; the real imports happen on the
    next QGIS start.

    Dotted names work too: ``find_spec`` imports the *parent* package only
    (a namespace package, so no code runs) and then merely locates the
    submodule.
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


def _version_tuple(text):
    """Numeric release part of a version string, for ordering ('1.2.3rc1')."""
    parts = []
    for chunk in text.split(".")[:3]:
        digits = ""
        for ch in chunk:
            if not ch.isdigit():
                break
            digits += ch
        parts.append(int(digits) if digits else 0)
    return tuple(parts)


def installed_version(dist="bluemesh2d"):
    """Version of an installed distribution, or None -- metadata only.

    Never imports the package: reading metadata is safe in a live QGIS,
    importing freshly installed binary wheels is not.
    """
    try:
        from importlib.metadata import version
        return version(dist)
    except Exception:
        return None


def needs_upgrade():
    """True when bluemesh2d is installed but older than :data:`MIN_VERSION`.

    Such an install satisfies :func:`find_missing` yet is too old for this
    plugin, so the dialog must still offer to install.
    """
    current = installed_version()
    if current is None:
        return False
    return _version_tuple(current) < _version_tuple(MIN_VERSION)


def pip_required(min_version=None):
    """The pip specs to install, floored at ``min_version`` (default
    :data:`MIN_VERSION`).

    Passing the newest PyPI release is how the dialog performs an *upgrade*:
    pip only touches an already-satisfied requirement when the spec demands a
    version it does not have, so raising the floor upgrades through every
    install route (venv, ``--user``, ``--break-system-packages``) without an
    extra ``--upgrade`` flag.
    """
    return (f"bluemesh2d>={min_version or MIN_VERSION}", "pyproj")


def latest_version(dist="bluemesh2d", timeout=PYPI_TIMEOUT, url=None):
    """Newest release of ``dist`` on PyPI, or None if it cannot be determined.

    Standard library only (this module must stay importable in a broken
    environment) and never raises: no network, a proxy, a timeout, an HTTP
    error or unexpected JSON all return None, which callers treat as "cannot
    tell" and skip the check.
    """
    import json
    import urllib.request

    try:
        request = urllib.request.Request(
            (url or PYPI_JSON_URL).format(dist=dist),
            headers={"User-Agent": "BlueMesh2D-QGIS-plugin"},
        )
        # nosec B310 -- constant https URL, not user input
        with contextlib.closing(
                urllib.request.urlopen(request, timeout=timeout)) as response:
            version = json.load(response)["info"]["version"]
        return version.strip() or None
    except Exception:
        return None


def update_available(timeout=PYPI_TIMEOUT, url=None):
    """Newest PyPI version when it is newer than the installed one, else None.

    Returns None when bluemesh2d is absent (there is nothing to compare -- the
    install path already fetches the newest release) and when PyPI cannot be
    reached, so a failed query never looks like "you are up to date".
    """
    current = installed_version()
    if current is None:
        return None
    latest = latest_version(timeout=timeout, url=url)
    if latest is None:
        return None
    return latest if _version_tuple(current) < _version_tuple(latest) else None


def source_checkout():
    """Repo root when this plugin folder sits inside a bluemesh2d checkout.

    Undocumented developer hook: returns a path only for someone running the
    plugin straight from the repository (typically a symlink from the QGIS
    profile's ``plugins/`` into the checkout -- hence the ``realpath``), or
    when ``BLUEMESH2D_DEV_PATH`` points at a checkout. A plugin installed
    from a zip always gets ``None``, so the dialog's development section
    never shows up for end users.
    """
    env = os.environ.get("BLUEMESH2D_DEV_PATH")
    candidates = [env] if env else []
    candidates.append(
        os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
    for root in candidates:
        cfg = os.path.join(root, "pyproject.toml")
        try:
            with open(cfg, encoding="utf-8") as fh:
                if 'name = "bluemesh2d"' in fh.read():
                    return root
        except OSError:
            continue
    return None


def dev_install(path):
    """Install a source checkout editable (``pip install -e <path>``).

    Goes through :func:`run_pip`, so it lands in the same place as a normal
    install (plugin venv on PEP 668 systems, user site elsewhere) and picks
    up the same numpy pin. Dependencies resolve from the checkout's own
    pyproject.toml; pyproj is added for the same reason as in
    :data:`PIP_REQUIRED`.
    """
    return run_pip(["-e", path, "pyproj"])


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
        # on failure just fall through: a fresh venv is created at `new`
        with contextlib.suppress(Exception):
            shutil.move(old, new)


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
    """True when the plugin venv already has a working pip.

    Every ``subprocess.run`` in this module (here and below) runs a fixed
    argument list with ``shell=False``: the executable is the interpreter of
    the venv this plugin created itself and the arguments are literals or
    paths this module computed. No user-supplied string reaches the command
    line, hence the ``nosec`` markers.
    """
    py = _venv_python(venv)
    if not os.path.exists(py):
        return False
    try:
        return subprocess.run([py, "-m", "pip", "--version"],  # nosec B603
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
    # constant, hard-coded https URL -- assert it anyway so no future edit can
    # turn this into a file:/ or custom-scheme fetch
    if not url.startswith("https://"):
        raise ValueError("get-pip.py must be fetched over https")
    log.append(f"pip is missing: bootstrapping it from {url} ...")
    try:
        with urllib.request.urlopen(url, timeout=60) as resp:  # nosec B310
            script = resp.read()
        with tempfile.NamedTemporaryFile("wb", suffix="_get-pip.py",
                                         delete=False) as f:
            f.write(script)
            path = f.name
        proc = subprocess.run([_venv_python(venv), path],  # nosec B603
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
        proc = subprocess.run(  # nosec B603
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
        proc = subprocess.run(  # nosec B603
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
        with contextlib.suppress(Exception):
            marker = os.path.join(sysconfig.get_path(key),
                                  "EXTERNALLY-MANAGED")
            if os.path.exists(marker):
                return True
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


def manual_command(packages=PIP_REQUIRED):
    """The per-OS command a user can run by hand if the dialog's pip fails.

    ``packages`` are pip requirement specs; conda gets the same specs, since
    bluemesh2d is on conda-forge only via pip anyway -- there is no apt/conda
    system package for it, which is why the old distro-package hint is gone.
    """
    pkgs = " ".join(packages)
    if _is_conda():
        # bluemesh2d has no conda package, but its dependencies do -- and
        # letting pip pull binary wheels into a conda env is what breaks it.
        # So: conda for the stack, pip --no-deps for the library itself.
        return (f"conda install -c conda-forge {' '.join(CONDA_PACKAGES)}\n"
                f"  python -m pip install --no-deps bluemesh2d>={MIN_VERSION}")
    system = platform.system()
    if system == "Windows":
        return f"python -m pip install {pkgs}   (in the OSGeo4W Shell)"
    if system == "Darwin":
        arglist = ", ".join(f"'{p}'" for p in packages)
        return (f"import pip; pip.main(['install', '--user', {arglist}])"
                "   (in the QGIS Python console)")
    if _is_externally_managed():
        return f"{_venv_dir()}/bin/python -m pip install {pkgs}"
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
        return False, ("This QGIS runs on a conda Python; pip wheels would "
                       "fight the conda stack. Install by hand instead:\n  "
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

        self._missing = find_missing(REQUIRED)
        self._outdated = needs_upgrade()
        self._checkout = source_checkout()
        # "is it the latest?" only matters for an install that already works;
        # when something is missing or too old, installing fetches the newest
        # release anyway, so the query is skipped and the dialog stays instant
        self._update = (None if (self._missing or self._outdated)
                        else update_available())

        self.dialog = QDialog(parent)
        self.dialog.setWindowTitle("BlueMesh2D — Python dependencies")
        self.dialog.setMinimumWidth(560)
        lay = QVBoxLayout(self.dialog)

        if self._missing:
            lay.addWidget(QLabel(
                "<b>BlueMesh2D is not installed in this QGIS.</b><br>"
                "Installing it also brings in numpy, scipy, shapely, "
                "rasterio, matplotlib, netCDF4, xarray and triangle."))
        elif self._outdated:
            lay.addWidget(QLabel(
                f"<b>BlueMesh2D {installed_version()} is installed, but this "
                f"plugin needs {MIN_VERSION} or newer.</b>"))
        elif self._update:
            lay.addWidget(QLabel(
                f"<b>BlueMesh2D {installed_version()} is installed; "
                f"{self._update} is available on PyPI.</b><br>"
                "The plugin works with the installed version; upgrading is "
                "optional."))
        else:
            lay.addWidget(QLabel(
                f"<b>BlueMesh2D {installed_version()} is installed.</b>"))

        self.dev_box = None
        if self._checkout is not None:
            # only ever visible when running from a source checkout
            self.dev_box = QCheckBox(
                "Development: install this source checkout editable\n"
                f"({self._checkout})")
            self.dev_box.setChecked(False)
            lay.addWidget(self.dev_box)

        self.log = QPlainTextEdit()
        self.log.setReadOnly(True)
        self.log.setMinimumHeight(160)
        self.log.setPlaceholderText(
            "pip output will appear here.\n\nManual fallback command:\n  "
            + manual_command())
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
        # a developer may want to switch to an editable install at any time,
        # so the button stays live whenever the dev checkbox is available
        if (not self._missing and not self._outdated and not self._update
                and self.dev_box is None):
            self.install_btn.setEnabled(False)
        if self._update and not self._missing:
            self.install_btn.setText(f"Upgrade to {self._update}")

    def _dev_selected(self):
        return self.dev_box is not None and self.dev_box.isChecked()

    def _install(self):
        from qgis.PyQt.QtCore import Qt
        from qgis.PyQt.QtWidgets import QApplication

        dev = self._dev_selected()
        # raising the floor to the newest release is what makes pip upgrade
        packages = pip_required(self._update) if self._update else PIP_REQUIRED
        if dev:
            self.log.setPlainText(
                f"Installing editable: {self._checkout} ...\n")
        else:
            self.log.setPlainText(
                f"Installing: {', '.join(packages)} ...\n")
        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        QApplication.processEvents()
        try:
            ok, out = (dev_install(self._checkout) if dev
                       else run_pip(list(packages)))
        finally:
            QApplication.restoreOverrideCursor()
        self.log.appendPlainText(out)
        # the post-install check must see what pip just wrote: put the venv /
        # user site on sys.path and drop the finders' cached directory listings
        import importlib

        activate_venv()
        importlib.invalidate_caches()
        still = find_missing(REQUIRED)
        if ok and not still:
            self.log.appendPlainText(
                "\nDone. Please RESTART QGIS to activate BlueMesh2D.")
            self.install_btn.setEnabled(False)
        else:
            self.log.appendPlainText(
                "\nInstallation did not complete"
                + (f" (still missing: {', '.join(still)})." if still else ".")
                + "\nManual fallback:\n  " + manual_command())

    def exec(self):
        # QGIS shows an app-wide busy cursor while plugins load; without an
        # explicit arrow override the dialog inherits the spinning cursor
        # for its whole lifetime.
        from qgis.PyQt.QtCore import Qt
        from qgis.PyQt.QtWidgets import QApplication

        QApplication.setOverrideCursor(Qt.CursorShape.ArrowCursor)
        try:
            return self.dialog.exec()
        finally:
            QApplication.restoreOverrideCursor()


def ensure_dependencies(parent=None):
    """Return True when bluemesh2d is present and recent; else show the dialog."""
    if not find_missing(REQUIRED) and not needs_upgrade():
        return True
    DepsDialog(parent).exec()
    return not find_missing(REQUIRED)
