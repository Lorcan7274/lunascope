"""Tests for :mod:`lunascope.tls`."""

import ssl
import sys

import pytest

from lunascope import tls


def test_create_default_context_returns_sslcontext():
    ctx = tls.create_default_context()
    assert isinstance(ctx, ssl.SSLContext)
    # Should be a TLS client context with default protocol
    assert ctx.protocol in (ssl.PROTOCOL_TLS_CLIENT, ssl.PROTOCOL_TLS)


def test_configure_tls_is_noop_off_macos(monkeypatch):
    """``configure_tls`` must not touch the global SSL state on non-macOS."""
    if sys.platform == "darwin":
        pytest.skip("This test asserts non-macOS behaviour")
    # Reset the module-level flag so we can observe the no-op path
    monkeypatch.setattr(tls, "_TRUSTSTORE_INSTALLED", False)
    tls.configure_tls()
    assert tls._TRUSTSTORE_INSTALLED is False


def test_configure_tls_idempotent(monkeypatch):
    monkeypatch.setattr(tls, "_TRUSTSTORE_INSTALLED", True)
    # Should be a no-op when already installed; no exception expected.
    tls.configure_tls()
    assert tls._TRUSTSTORE_INSTALLED is True
