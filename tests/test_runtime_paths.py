"""Tests for :mod:`lunascope.runtime_paths`."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

from lunascope import runtime_paths


def test_user_cache_root_returns_path():
    p = runtime_paths.user_cache_root()
    assert isinstance(p, Path)


def test_app_cache_root_creates_directory(tmp_path, monkeypatch):
    """``app_cache_root`` must create the cache dir and return it."""
    monkeypatch.setattr(runtime_paths, "user_cache_root", lambda: tmp_path)
    cache = runtime_paths.app_cache_root()
    assert cache.exists()
    assert cache.is_dir()
    assert cache.name == "lunascope"
    assert cache.parent == tmp_path


def test_app_state_file_joins_parts(tmp_path, monkeypatch):
    monkeypatch.setattr(runtime_paths, "user_cache_root", lambda: tmp_path)
    p = runtime_paths.app_state_file("subdir", "settings.json")
    assert p.parts[-2:] == ("subdir", "settings.json")
    assert p.parent.parent.name == "lunascope"


def test_app_cache_root_falls_back_when_unwritable(tmp_path, monkeypatch):
    """If the preferred path can't be created, fall back to a tempdir."""
    bad = tmp_path / "blocked"
    # simulate "preferred" location whose mkdir raises
    monkeypatch.setattr(runtime_paths, "user_cache_root", lambda: bad)

    real_mkdir = Path.mkdir

    def fake_mkdir(self, *args, **kwargs):
        if str(self).startswith(str(bad)):
            raise OSError("simulated permission denied")
        return real_mkdir(self, *args, **kwargs)

    monkeypatch.setattr(Path, "mkdir", fake_mkdir)
    out = runtime_paths.app_cache_root()
    assert out.exists()
    assert "lunascope-cache" in str(out)


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX-only XDG behaviour")
def test_xdg_cache_home_honored_on_linux(tmp_path, monkeypatch):
    if sys.platform == "darwin":
        pytest.skip("macOS uses ~/Library/Caches, not XDG")
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
    assert runtime_paths.user_cache_root() == tmp_path


@pytest.mark.skipif(sys.platform != "win32", reason="Windows-only env var")
def test_windows_uses_localappdata(tmp_path, monkeypatch):
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path))
    assert runtime_paths.user_cache_root() == tmp_path
