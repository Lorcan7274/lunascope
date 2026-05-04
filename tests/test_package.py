"""Smoke tests for the top-level ``lunascope`` package."""

import importlib
import re

import pytest


def test_package_imports():
    mod = importlib.import_module("lunascope")
    assert mod is not None


def test_version_string_format():
    import lunascope

    assert hasattr(lunascope, "__version__")
    assert isinstance(lunascope.__version__, str)
    # Versions should look like "1.5.5" or "1.5.0a1" — three numeric components
    assert re.match(r"^\d+\.\d+\.\d+", lunascope.__version__)


@pytest.mark.parametrize(
    "submodule",
    [
        "lunascope.helpers",
        "lunascope.runtime_paths",
        "lunascope.tls",
        "lunascope.session_state",
        "lunascope.lwf",
        "lunascope.updater",
        "lunascope.components.topo_clocs",
        "lunascope.components.topo_core",
        "lunascope.components.harmonizer_funcs",
    ],
)
def test_submodules_importable(submodule):
    importlib.import_module(submodule)
