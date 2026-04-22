import ssl
import sys


_TRUSTSTORE_INSTALLED = False


def configure_tls() -> None:
    """Use the native macOS trust store when available.

    The frozen app can otherwise end up validating HTTPS only against the
    bundled CA file, which misses locally trusted enterprise/intercept roots.
    """
    global _TRUSTSTORE_INSTALLED
    if _TRUSTSTORE_INSTALLED or sys.platform != "darwin":
        return
    try:
        import truststore
        truststore.inject_into_ssl()
        _TRUSTSTORE_INSTALLED = True
    except Exception:
        # Fall back to Python's default TLS behavior if truststore is missing
        # or fails to initialize in a given runtime.
        pass


def create_default_context() -> ssl.SSLContext:
    """Create an SSL context aligned with the app's TLS configuration."""
    configure_tls()
    if sys.platform == "darwin" and _TRUSTSTORE_INSTALLED:
        return ssl.create_default_context()
    try:
        import certifi
        return ssl.create_default_context(cafile=certifi.where())
    except Exception:
        return ssl.create_default_context()
