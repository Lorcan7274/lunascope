# Lunascope test suite

Pytest-based unit/integration tests covering the Qt-free analytic core
and a Qt round-trip suite for session save/restore and widget helpers.

## Running

Install the test extras and run pytest:

```bash
pip install -e ".[test]"
pytest
```

The Qt tests use the offscreen platform plugin, which is set automatically in
`tests/conftest.py`. No display server is required.

To run a single file or marker:

```bash
pytest tests/test_topo_clocs.py
pytest -m "not qt"            # skip Qt-dependent tests
pytest -m qt                  # only Qt tests
```

## Coverage

| Test file                     | Module under test                                  |
|-------------------------------|----------------------------------------------------|
| `test_package.py`             | top-level imports, version string                  |
| `test_runtime_paths.py`       | `lunascope.runtime_paths`                          |
| `test_tls.py`                 | `lunascope.tls`                                    |
| `test_helpers_logic.py`       | `lunascope.helpers` (pure logic)                   |
| `test_helpers_qt.py`          | `lunascope.helpers` (Qt widgets)                   |
| `test_topo_clocs.py`          | `lunascope.components.topo_clocs`                  |
| `test_topo_core.py`           | `lunascope.components.topo_core`                   |
| `test_harmonizer_funcs.py`    | `lunascope.components.harmonizer_funcs`            |
| `test_lwf.py`                 | `lunascope.lwf` (binary format reader)             |
| `test_session_state.py`       | `lunascope.session_state` (save/restore)           |
| `test_updater.py`             | `lunascope.updater` (PyPI / GitHub fetchers)       |
| `test_results_io.py`          | `lunascope.components.results_io` (pkl/zip)        |

CI is configured in `.github/workflows/test.yml` to run on Ubuntu, macOS,
and Windows against Python 3.10 and 3.12.
