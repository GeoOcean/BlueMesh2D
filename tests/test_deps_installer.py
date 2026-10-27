"""Tests for the QGIS plugin's dependency installer.

``deps_installer`` is deliberately free of third-party imports at module
level (see its docstring), which is exactly what makes it testable outside
QGIS: it is imported here as a top-level module from the plugin folder, with
no ``qgis`` package available. Only the non-Qt half is covered -- ``DepsDialog``
builds widgets and is left to manual testing in QGIS.
"""
import importlib.util
import os
import shlex
import subprocess
import sys
import textwrap

import pytest

_PLUGIN_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "bluemesh2d_qgis",
)


@pytest.fixture(scope="module")
def deps():
    spec = importlib.util.spec_from_file_location(
        "bluemesh2d_deps_installer",
        os.path.join(_PLUGIN_DIR, "deps_installer.py"),
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    yield module
    del sys.modules[spec.name]


def test_module_imports_without_third_party(deps):
    # the whole point of the module: it must load in a broken environment
    assert deps.MIN_VERSION
    assert deps.PIP_REQUIRED[0].startswith("bluemesh2d>=")
    assert "pyproj" in deps.PIP_REQUIRED


def test_find_missing_handles_dotted_and_absent_names(deps):
    # os.path exists, so a dotted name must resolve; the parent import that
    # find_spec performs is what this guards against regressing
    assert deps.find_missing(["os.path"]) == []
    assert deps.find_missing(["definitely_not_a_module"]) == \
        ["definitely_not_a_module"]
    assert deps.find_missing(["os.definitely_not_a_submodule"]) == \
        ["os.definitely_not_a_submodule"]


def test_version_tuple_orders_releases(deps):
    assert deps._version_tuple("0.1.1") == (0, 1, 1)
    assert deps._version_tuple("0.1.2.dev0") < deps._version_tuple("0.2.0")
    assert deps._version_tuple("1.0") < deps._version_tuple("1.0.1")
    # non-numeric suffixes must not raise
    assert deps._version_tuple("0.1.1rc1") == (0, 1, 1)


def _fake_versions(**versions):
    return lambda dist="bluemesh2d": versions.get(dist)


def test_needs_upgrade(deps, monkeypatch):
    monkeypatch.setattr(deps, "installed_version", _fake_versions())
    assert deps.needs_upgrade() is False  # not installed != outdated

    monkeypatch.setattr(deps, "installed_version",
                        _fake_versions(bluemesh2d="0.0.9"))
    assert deps.needs_upgrade() is True

    monkeypatch.setattr(deps, "installed_version",
                        _fake_versions(bluemesh2d=deps.MIN_VERSION))
    assert deps.needs_upgrade() is False


def test_needs_upgrade_old_packages(deps, monkeypatch):
    # shapely 1.x has no shapely.prepare; netCDF4 1.5 uses the removed np.bool
    ok = {"bluemesh2d": deps.MIN_VERSION, **deps.MIN_PACKAGES}
    for name, old in (("shapely", "1.8.5"), ("netCDF4", "1.5.4"),
                      ("scipy", "1.5.1")):
        monkeypatch.setattr(deps, "installed_version",
                            _fake_versions(**{**ok, name: old}))
        assert deps.needs_upgrade() is True
        assert list(deps.outdated_packages()) == [name]

    monkeypatch.setattr(deps, "installed_version", _fake_versions(**ok))
    assert deps.needs_upgrade() is False
    assert deps.outdated_packages() == {}


def test_source_checkout_finds_this_repo(deps):
    # the tests run from a checkout, so the plugin folder's parent is one
    root = deps.source_checkout()
    assert root is not None
    assert os.path.isfile(os.path.join(root, "pyproject.toml"))


def test_source_checkout_none_outside_a_checkout(deps, monkeypatch, tmp_path):
    # simulate a plugin installed from a zip: no pyproject.toml above it
    fake = tmp_path / "plugins" / "bluemesh2d_qgis" / "deps_installer.py"
    fake.parent.mkdir(parents=True)
    fake.write_text("")
    monkeypatch.setattr(deps.os.path, "realpath", lambda _p: str(fake))
    monkeypatch.delenv("BLUEMESH2D_DEV_PATH", raising=False)
    assert deps.source_checkout() is None


def test_source_checkout_honours_env_override(deps, monkeypatch, tmp_path):
    (tmp_path / "pyproject.toml").write_text('name = "bluemesh2d"\n')
    monkeypatch.setenv("BLUEMESH2D_DEV_PATH", str(tmp_path))
    assert deps.source_checkout() == str(tmp_path)


def test_metadata_external_deps_matches_pip_required(deps):
    """metadata.txt advertises to plugins.qgis.org what we actually install."""
    import configparser

    cfg = configparser.ConfigParser()
    cfg.read(os.path.join(_PLUGIN_DIR, "metadata.txt"), encoding="utf-8")
    advertised = [s.strip() for s in
                  cfg["general"]["external_deps"].split(",") if s.strip()]
    assert advertised == list(deps.PIP_REQUIRED)


def test_manual_command_mentions_both_packages(deps, monkeypatch):
    monkeypatch.setattr(deps, "_is_conda", lambda: False)
    for system in ("Linux", "Windows", "Darwin"):
        monkeypatch.setattr(deps.platform, "system", lambda s=system: s)
        cmd = deps.manual_command()
        assert "bluemesh2d" in cmd and "pyproj" in cmd


def test_manual_command_on_conda_uses_conda_for_the_stack(deps, monkeypatch):
    monkeypatch.setattr(deps, "_is_conda", lambda: True)
    cmd = deps.manual_command()
    assert "conda install -c conda-forge" in cmd
    # bluemesh2d has no conda package: it must come from pip, without deps
    assert f'pip install --no-deps "bluemesh2d>={deps.MIN_VERSION}"' in cmd


# --------------------------------------------------------------- PyPI check
# `latest_version` is the only part of the module that touches the network.
# Every test here feeds it a local file:// URL or a failing opener, so the
# suite never depends on PyPI being reachable.

def _pypi_json_file(tmp_path, version, monkeypatch=None):
    """A file:// URL serving a minimal PyPI JSON payload.

    ``latest_version`` only opens https, so when ``monkeypatch`` is given an
    https URL is returned instead, with ``urlopen`` patched to serve the file.
    """
    import json
    import urllib.request
    path = tmp_path / "pypi.json"
    path.write_text(json.dumps({"info": {"version": version}}))
    if monkeypatch is None:
        return path.as_uri()
    monkeypatch.setattr(
        urllib.request, "urlopen",
        lambda request, timeout=None: open(path, "rb"))
    return "https://pypi.invalid/pypi/bluemesh2d/json"


def test_latest_version_reads_the_release_from_the_payload(deps, tmp_path, monkeypatch):
    url = _pypi_json_file(tmp_path, "9.9.9", monkeypatch)
    assert deps.latest_version(url=url) == "9.9.9"


def test_latest_version_returns_none_when_pypi_is_unreachable(deps, tmp_path):
    # offline, proxy, timeout, 404: all must be swallowed, never raised
    missing = (tmp_path / "nope.json").as_uri()
    assert deps.latest_version(url=missing) is None


def test_latest_version_returns_none_on_unexpected_payload(deps, tmp_path):
    path = tmp_path / "bad.json"
    path.write_text('{"unexpected": true}')
    assert deps.latest_version(url=path.as_uri()) is None


def test_update_available_flags_a_newer_release(deps, tmp_path, monkeypatch):
    monkeypatch.setattr(deps, "installed_version", lambda dist="bluemesh2d": "0.1.4")
    assert deps.update_available(url=_pypi_json_file(tmp_path, "0.1.6", monkeypatch)) == "0.1.6"


def test_update_available_is_none_when_already_latest(deps, tmp_path, monkeypatch):
    monkeypatch.setattr(deps, "installed_version", lambda dist="bluemesh2d": "0.1.6")
    assert deps.update_available(url=_pypi_json_file(tmp_path, "0.1.6", monkeypatch)) is None
    # a local build ahead of PyPI must not be reported as outdated either
    monkeypatch.setattr(deps, "installed_version", lambda dist="bluemesh2d": "0.2.0")
    assert deps.update_available(url=_pypi_json_file(tmp_path, "0.1.6", monkeypatch)) is None


def test_update_available_is_none_when_not_installed(deps, tmp_path, monkeypatch):
    # nothing to compare: the install path fetches the newest release anyway
    monkeypatch.setattr(deps, "installed_version", lambda dist="bluemesh2d": None)
    assert deps.update_available(url=_pypi_json_file(tmp_path, "0.1.6", monkeypatch)) is None


def test_update_available_is_none_when_the_query_fails(deps, tmp_path, monkeypatch):
    # a failed query must not read as "you are up to date"
    monkeypatch.setattr(deps, "installed_version", lambda dist="bluemesh2d": "0.1.4")
    assert deps.update_available(url=(tmp_path / "nope.json").as_uri()) is None


def test_pip_required_raises_the_floor_for_an_upgrade(deps):
    assert deps.pip_required() == deps.PIP_REQUIRED
    specs = deps.pip_required("0.1.6")
    assert specs[0] == "bluemesh2d>=0.1.6"
    assert "pyproj" in specs


# ------------------------------------------------------ cross-platform paths
# Windows and macOS cannot be run from a Linux box, so what differs between
# them is simulated by patching `platform.system`; the CI matrix
# (.github/workflows/test-qgis-plugin.yml) runs this same file on all three.

def _as(deps, monkeypatch, system):
    monkeypatch.setattr(deps.platform, "system", lambda: system)


def test_venv_python_is_python_exe_under_scripts_on_windows(deps, monkeypatch):
    # `Scripts\python` (no .exe) never exists, so a bare os.path.exists on it
    # would report "no venv" for a perfectly good one
    _as(deps, monkeypatch, "Windows")
    assert deps._venv_python("v") == os.path.join("v", "Scripts", "python.exe")
    assert deps._venv_site_packages("v") == \
        os.path.join("v", "Lib", "site-packages")
    for system in ("Linux", "Darwin"):
        _as(deps, monkeypatch, system)
        assert deps._venv_python("v") == os.path.join("v", "bin", "python")
        assert deps._venv_site_packages("v").startswith(
            os.path.join("v", "lib", "python"))


def test_no_console_flag_only_on_windows(deps, monkeypatch):
    _as(deps, monkeypatch, "Windows")
    assert "creationflags" in deps._no_console()
    for system in ("Linux", "Darwin"):
        _as(deps, monkeypatch, system)
        assert deps._no_console() == {}


@pytest.mark.parametrize("system", ["Linux", "Windows", "Darwin"])
@pytest.mark.parametrize("managed", [False, True])
def test_manual_command_never_leaves_a_spec_unquoted(
        deps, monkeypatch, system, managed):
    # an unquoted `bluemesh2d>=0.1.8` is a shell redirection in cmd,
    # PowerShell, bash and zsh: it installs the unpinned package and drops a
    # file named `=0.1.8` in the current directory
    _as(deps, monkeypatch, system)
    monkeypatch.setattr(deps, "_is_conda", lambda: False)
    monkeypatch.setattr(deps, "_is_externally_managed", lambda: managed)
    specs = (*deps.PIP_REQUIRED, "numpy<2")
    cmd = deps.manual_command(specs)
    quote = "'" if system == "Darwin" else '"'  # macOS: a Python string
    for spec in specs:
        if any(c in spec for c in "<>"):
            assert f"{quote}{spec}{quote}" in cmd, spec


def test_manual_command_survives_shell_parsing(deps, monkeypatch, tmp_path):
    monkeypatch.setattr(deps, "_is_conda", lambda: False)
    monkeypatch.setattr(deps, "_is_externally_managed", lambda: False)
    _as(deps, monkeypatch, "Linux")
    line = deps.manual_command().splitlines()[0]
    tokens = shlex.split(line)
    assert tokens[:5] == ["python3", "-m", "pip", "install", "--user"]
    assert f"bluemesh2d>={deps.MIN_VERSION}" in tokens
    assert not any(t.startswith(("=", ">")) for t in tokens)

    # a venv under a path with a space must stay one argument
    monkeypatch.setattr(deps, "_is_externally_managed", lambda: True)
    monkeypatch.setattr(deps, "_venv_dir", lambda: str(tmp_path / "with space"))
    first = shlex.split(deps.manual_command().splitlines()[0])[0]
    assert first == deps._venv_python(str(tmp_path / "with space"))


def test_manual_command_names_where_to_run_it(deps, monkeypatch):
    monkeypatch.setattr(deps, "_is_conda", lambda: False)
    _as(deps, monkeypatch, "Windows")
    assert "OSGeo4W" in deps.manual_command()
    _as(deps, monkeypatch, "Darwin")
    assert "QGIS Python console" in deps.manual_command()


def test_pip_missing_hint_is_distro_neutral_and_os_aware(deps, monkeypatch):
    _as(deps, monkeypatch, "Linux")
    hint = deps._pip_missing_hint()
    assert "apt" in hint and "dnf" in hint and "pacman" in hint
    _as(deps, monkeypatch, "Windows")
    assert "OSGeo4W" in deps._pip_missing_hint()
    _as(deps, monkeypatch, "Darwin")
    assert deps._pip_missing_hint() == ""


# ------------------------------------------------------ sys.path precedence

@pytest.fixture
def isolated_sys_path(monkeypatch):
    """A sys.path the test may reshape, with the import caches reset."""
    monkeypatch.setattr(sys, "path", list(sys.path))
    yield
    importlib.invalidate_caches()


def _write_pkg(root, name, marker):
    pkg = root / name
    pkg.mkdir(parents=True)
    (pkg / "__init__.py").write_text(f"WHERE = {marker!r}\n")


def test_installed_site_dir_goes_ahead_of_the_system_one(
        deps, monkeypatch, tmp_path, isolated_sys_path):
    # `site.addsitedir` appends, so an older system copy (Ubuntu's apt
    # shapely 1.8, the macOS bundle's Shapely 1.x) would keep shadowing the
    # newer one pip just installed
    system, ours = tmp_path / "system", tmp_path / "ours"
    _write_pkg(system, "zz_shadow_pkg", "system")
    _write_pkg(ours, "zz_shadow_pkg", "ours")
    sys.path.append(str(system))
    monkeypatch.setattr(deps.site, "getsitepackages", lambda: [str(system)])
    tail = sys.path[-1]

    assert deps._add_site_first(str(ours)) is True

    assert sys.path.index(str(ours)) < sys.path.index(str(system))
    assert sys.path[-1] == tail  # nothing else was disturbed
    importlib.invalidate_caches()
    sys.modules.pop("zz_shadow_pkg", None)
    try:
        import zz_shadow_pkg
        assert zz_shadow_pkg.WHERE == "ours"
    finally:
        sys.modules.pop("zz_shadow_pkg", None)


def test_site_dir_is_appended_when_there_is_no_system_entry_to_anchor_on(
        deps, monkeypatch, tmp_path, isolated_sys_path):
    monkeypatch.setattr(deps.site, "getsitepackages", lambda: ["/nonexistent"])
    (tmp_path / "ours").mkdir()
    assert deps._add_site_first(str(tmp_path / "ours")) is True
    assert sys.path[-1] == str(tmp_path / "ours")  # the historical behaviour
    monkeypatch.setattr(deps.site, "getsitepackages",
                        lambda: (_ for _ in ()).throw(AttributeError()))
    (tmp_path / "more").mkdir()
    assert deps._add_site_first(str(tmp_path / "more")) is True


def test_activate_venv_prefers_the_plugin_venv(
        deps, monkeypatch, tmp_path, isolated_sys_path):
    venv = tmp_path / "bluemesh2d_deps"
    sp = deps._venv_site_packages(str(venv))
    _write_pkg(__import__("pathlib").Path(sp), "zz_venv_pkg", "venv")
    system = tmp_path / "system"
    _write_pkg(system, "zz_venv_pkg", "system")
    sys.path.append(str(system))
    monkeypatch.setattr(deps.site, "getsitepackages", lambda: [str(system)])
    monkeypatch.setattr(deps, "_venv_dir", lambda: str(venv))
    monkeypatch.setattr(deps, "_migrate_old_venv", lambda: None)
    monkeypatch.setattr(deps, "_activate_user_site", lambda: False)

    assert deps.activate_venv() is True
    assert deps.activate_venv() is False  # idempotent

    importlib.invalidate_caches()
    sys.modules.pop("zz_venv_pkg", None)
    try:
        import zz_venv_pkg
        assert zz_venv_pkg.WHERE == "venv"
    finally:
        sys.modules.pop("zz_venv_pkg", None)


# ------------------------------------------------- standalone interpreter
# Embedded in QGIS, sys.executable is the application (qgis-bin.exe, the
# QGIS binary in QGIS.app), not a Python. Running it with `-m venv` / `-m pip`
# starts another QGIS.

def _spy_on_subprocess(deps, monkeypatch):
    ran = []
    real_run = subprocess.run

    def run(argv, *args, **kwargs):
        ran.append(list(argv))
        return real_run(argv, *args, **kwargs)

    monkeypatch.setattr(deps.subprocess, "run", run)
    return ran


def test_python_executable_skips_a_host_application(deps, monkeypatch, tmp_path):
    host = tmp_path / ("qgis-bin.exe" if os.name == "nt" else "QGIS")
    host.write_text("not a python")
    monkeypatch.setattr(sys, "executable", str(host))
    monkeypatch.setattr(sys, "_base_executable", str(host), raising=False)
    ran = _spy_on_subprocess(deps, monkeypatch)

    found = deps._python_executable()

    # the host is never executed, whatever else is (or is not) found
    assert all(str(host) not in argv for argv in ran)
    assert all(deps._PYTHON_NAME.match(os.path.basename(argv[0]))
               for argv in ran)
    if found is not None:
        assert deps._PYTHON_NAME.match(os.path.basename(found))


def test_python_executable_none_when_nothing_usable(deps, monkeypatch, tmp_path):
    monkeypatch.setattr(sys, "executable", str(tmp_path / "QGIS"))
    monkeypatch.setattr(sys, "_base_executable", "", raising=False)
    monkeypatch.setattr(sys, "base_exec_prefix", str(tmp_path / "empty"))
    monkeypatch.setattr(deps.sysconfig, "get_config_var", lambda _n: None)
    assert deps._python_executable() is None


@pytest.mark.skipif(os.name == "nt", reason="needs a #! script as the fake")
@pytest.mark.parametrize("reported, expected_found", [
    ("{ver}", True),    # right interpreter
    ("2.7", False),     # a python of another version is no use
])
def test_python_executable_checks_the_version(
        deps, monkeypatch, tmp_path, reported, expected_found):
    ver = f"{sys.version_info.major}.{sys.version_info.minor}"
    exe = tmp_path / "bin" / f"python{ver}"
    exe.parent.mkdir()
    exe.write_text(f"#!/bin/sh\necho {reported.format(ver=ver)}\n")
    exe.chmod(0o755)
    monkeypatch.setattr(sys, "executable", str(tmp_path / "QGIS"))
    monkeypatch.setattr(sys, "_base_executable", "", raising=False)
    monkeypatch.setattr(sys, "base_exec_prefix", str(tmp_path))
    monkeypatch.setattr(deps.sysconfig, "get_config_var", lambda _n: None)
    assert deps._python_executable() == (str(exe) if expected_found else None)


@pytest.mark.skipif(os.name == "nt", reason="needs a #! script as the fake")
def test_python_executable_rejects_one_that_cannot_run(deps, monkeypatch, tmp_path):
    # the vcpkg macOS bundle ships a python3.x that aborts outside the app
    ver = f"{sys.version_info.major}.{sys.version_info.minor}"
    exe = tmp_path / "bin" / f"python{ver}"
    exe.parent.mkdir()
    exe.write_text("#!/bin/sh\necho \"No module named 'encodings'\" >&2\nexit 1\n")
    exe.chmod(0o755)
    monkeypatch.setattr(sys, "executable", str(tmp_path / "QGIS"))
    monkeypatch.setattr(sys, "_base_executable", "", raising=False)
    monkeypatch.setattr(sys, "base_exec_prefix", str(tmp_path))
    monkeypatch.setattr(deps.sysconfig, "get_config_var", lambda _n: None)
    assert deps._python_executable() is None


def test_create_venv_gives_up_cleanly_without_an_interpreter(
        deps, monkeypatch, tmp_path):
    venv = tmp_path / "bluemesh2d_deps"
    monkeypatch.setattr(deps, "_venv_dir", lambda: str(venv))
    monkeypatch.setattr(deps, "_migrate_old_venv", lambda: None)
    monkeypatch.setattr(deps, "_python_executable", lambda: None)
    log = []
    assert deps._create_venv(log) is None
    assert not venv.exists()  # nothing half-built is left behind
    assert "standalone Python" in "\n".join(log)


def test_pep668_without_an_interpreter_falls_back_to_user_install(
        deps, monkeypatch, tmp_path):
    # Homebrew's QGIS: externally managed, yet sys.executable is the app
    monkeypatch.setattr(deps, "_is_conda", lambda: False)
    monkeypatch.setattr(deps, "_is_externally_managed", lambda: True)
    monkeypatch.setattr(deps, "_numpy_constraint", lambda: None)
    monkeypatch.setattr(deps, "_venv_dir", lambda: str(tmp_path / "v"))
    monkeypatch.setattr(deps, "_migrate_old_venv", lambda: None)
    monkeypatch.setattr(deps, "_python_executable", lambda: None)
    seen = []
    monkeypatch.setattr(deps, "_run_pip_inprocess",
                        lambda args: seen.append(args) or (True, "ok"))
    ok, out = deps.run_pip(["bluemesh2d"])
    assert ok
    assert seen == [["install", "--user", "--break-system-packages",
                     "bluemesh2d"]]
    assert "--break-system-packages" in out


def test_create_venv_builds_a_working_venv(deps, monkeypatch, tmp_path):
    """The real thing, on whichever OS runs the suite.

    On Windows this is what proves ``Scripts\\python.exe`` is where the venv
    interpreter really lives.
    """
    if importlib.util.find_spec("ensurepip") is None:
        pytest.skip("this Python has no ensurepip")
    venv = tmp_path / "bluemesh2d_deps"
    monkeypatch.setattr(deps, "_venv_dir", lambda: str(venv))
    monkeypatch.setattr(deps, "_migrate_old_venv", lambda: None)
    log = []
    made = deps._create_venv(log)
    assert made == str(venv), "\n".join(log)
    assert os.path.isfile(deps._venv_python(made))
    assert os.path.isdir(deps._venv_site_packages(made))
    assert deps._venv_has_pip(made)
    # a second call finds the venv it built instead of building again
    assert deps._create_venv([]) == str(venv)
