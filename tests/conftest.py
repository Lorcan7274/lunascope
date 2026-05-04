"""Pytest fixtures and global configuration for the Lunascope test suite.

The Qt-based tests are run against PySide6 with the ``offscreen`` platform
plugin so that no display server is required.  ``QApplication`` is created
lazily via the :func:`qapp` fixture; tests that only exercise pure-logic
modules do not need it.
"""

from __future__ import annotations

import os
import sys

import matplotlib

# Force a non-interactive matplotlib backend before any test imports it.
matplotlib.use("Agg", force=True)

# Force Qt to use the offscreen platform plugin.  This must be set before
# QApplication is constructed and before PySide6 is imported anywhere.
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest


@pytest.fixture(scope="session")
def qapp():
    """Session-scoped QApplication for Qt tests.

    Tests that need Qt widgets should depend on this fixture.  Reusing one
    application across the session avoids repeated start-up cost.
    """
    from PySide6.QtWidgets import QApplication

    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv)
    yield app
    # Do not call app.quit() here — re-creating QApplication later in the
    # same process is fragile across PySide6 versions.


@pytest.fixture
def tiny_channels_df():
    """Small synthetic channels dataframe shared by harmonizer tests."""
    import pandas as pd

    return pd.DataFrame(
        [
            {"ID": "S1", "CH": "EEG_F3", "SR": "256", "TRANS": "AC", "PDIM": "uV"},
            {"ID": "S1", "CH": "EEG_C3", "SR": "256", "TRANS": "AC", "PDIM": "uV"},
            {"ID": "S1", "CH": "ECG",    "SR": "128", "TRANS": "AC", "PDIM": "mV"},
            {"ID": "S2", "CH": "EEG_F3", "SR": "256", "TRANS": "AC", "PDIM": "uV"},
            {"ID": "S2", "CH": "EEG_O1", "SR": "256", "TRANS": "AC", "PDIM": "uV"},
            {"ID": "S2", "CH": "EOG_L",  "SR": "256", "TRANS": "AC", "PDIM": "uV"},
            {"ID": "S3", "CH": "EEG_F3", "SR": "200", "TRANS": "AC", "PDIM": "uV"},
            {"ID": "S3", "CH": "EEG_C3", "SR": "256", "TRANS": "AC", "PDIM": "uV"},
        ]
    )


@pytest.fixture
def tiny_annots_df():
    """Small synthetic annotations dataframe shared by harmonizer tests."""
    import pandas as pd

    return pd.DataFrame(
        [
            {"ID": "S1", "ANNOT": "arousal"},
            {"ID": "S1", "ANNOT": "spindle"},
            {"ID": "S2", "ANNOT": "arousal"},
            {"ID": "S2", "ANNOT": "Arousal"},
            {"ID": "S3", "ANNOT": "spindle"},
        ]
    )
