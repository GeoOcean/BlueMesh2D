"""Tests for the QGIS plugin's dependency installer.

``deps_installer`` is deliberately free of third-party imports at module
level (see its docstring), which is exactly what makes it testable outside
QGIS: it is imported here as a top-level module from the plugin folder, with
no ``qgis`` package available. Only the non-Qt half is covered -- ``DepsDialog``
builds widgets and is left to manual testing in QGIS.
"""
import importlib.util
import os
import sys

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


def test_needs_upgrade(deps, monkeypatch):
    monkeypatch.setattr(deps, "installed_version", lambda: None)
    assert deps.needs_upgrade() is False  # not installed != outdated

    monkeypatch.setattr(deps, "installed_version", lambda: "0.0.9")
    assert deps.needs_upgrade() is True

    monkeypatch.setattr(deps, "installed_version", lambda: deps.MIN_VERSION)
    assert deps.needs_upgrade() is False


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
    assert "pip install --no-deps bluemesh2d" in cmd
