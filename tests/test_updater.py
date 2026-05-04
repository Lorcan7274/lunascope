"""Tests for :mod:`lunascope.updater`.

Network access is mocked; we exercise version comparison logic and the
PyPI/GitHub fetcher functions against in-memory fake responses.
"""

from __future__ import annotations

import io
import json

import pytest

from lunascope import updater


class _FakeResp:
    def __init__(self, payload: bytes):
        self._payload = payload

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def read(self):
        return self._payload


def _patch_urlopen(monkeypatch, payload: dict):
    body = json.dumps(payload).encode("utf-8")

    def fake_urlopen(req, timeout=None, context=None):
        return _FakeResp(body)

    monkeypatch.setattr(updater.urllib.request, "urlopen", fake_urlopen)


def test_fetch_latest_version_pypi_parses_info_block(monkeypatch):
    _patch_urlopen(monkeypatch, {"info": {"version": "1.6.0"}})
    assert updater._fetch_latest_version_pypi() == "1.6.0"


def test_fetch_latest_version_github_strips_v_prefix(monkeypatch):
    _patch_urlopen(monkeypatch, {"tag_name": "v2.3.4"})
    assert updater._fetch_latest_version_github() == "2.3.4"


def test_fetch_latest_version_github_unprefixed(monkeypatch):
    _patch_urlopen(monkeypatch, {"tag_name": "0.9.1"})
    assert updater._fetch_latest_version_github() == "0.9.1"


@pytest.mark.parametrize(
    "current,latest,expected_newer",
    [
        ("1.5.0", "1.5.1", True),
        ("1.5.5", "1.5.5", False),
        ("1.5.5", "1.5.4", False),
        ("0.9.9", "1.0.0", True),
        ("1.10.0", "1.9.0", False),
    ],
)
def test_version_tuple_comparison(current, latest, expected_newer):
    """Mimic the comparison used in :func:`updater.check_and_prompt`."""
    cur = tuple(int(x) for x in current.split("."))
    lat = tuple(int(x) for x in latest.split("."))
    assert (lat > cur) is expected_newer


def test_fetch_pypi_propagates_errors(monkeypatch):
    def boom(req, timeout=None, context=None):
        raise RuntimeError("network down")

    monkeypatch.setattr(updater.urllib.request, "urlopen", boom)
    with pytest.raises(RuntimeError, match="network down"):
        updater._fetch_latest_version_pypi()


def test_local_test_index_picks_highest_wheel(monkeypatch, tmp_path):
    """When ``_LOCAL_TEST_INDEX`` is set, scan wheels in that directory."""
    for ver in ("1.0.0", "1.5.5", "1.10.0", "0.9.0"):
        (tmp_path / f"lunascope-{ver}-py3-none-any.whl").write_bytes(b"")
    monkeypatch.setattr(updater, "_LOCAL_TEST_INDEX", str(tmp_path))
    assert updater._fetch_latest_version_pypi() == "1.10.0"


def test_local_test_index_no_wheels_raises(monkeypatch, tmp_path):
    monkeypatch.setattr(updater, "_LOCAL_TEST_INDEX", str(tmp_path))
    with pytest.raises(RuntimeError, match="No wheels"):
        updater._fetch_latest_version_pypi()
