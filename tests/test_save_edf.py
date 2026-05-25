from __future__ import annotations

import pandas as pd

from lunascope.components.save_edf import SaveEDFMixin


class _FakeInstance:
    def __init__(self):
        self.commands = []
        self.edf = self

    def stat(self):
        return {"edf_file": "/tmp/sample.edf"}

    def annots(self):
        return pd.DataFrame({"Annotations": ["SPINDLE", "SO"]})

    def id(self):
        return "S1"

    def eval_lunascope(self, command):
        self.commands.append(command)


class _FakeDialog:
    Accepted = 1

    def __init__(self, *_args, **_kwargs):
        pass

    def exec(self):
        return self.Accepted

    def get_script(self):
        return "WRITE edf-dir=/tmp/out\nWRITE-ANNOTS file=/tmp/out/S1.annot"


class _FakeMessageBox:
    last_info = None
    last_critical = None

    @classmethod
    def information(cls, _parent, title, text):
        cls.last_info = (title, text)

    @classmethod
    def critical(cls, _parent, title, text):
        cls.last_critical = (title, text)


class _Controller(SaveEDFMixin):
    def __init__(self):
        self.ui = object()
        self.p = _FakeInstance()


def test_save_edf_annots_runs_export_commands_on_attached_instance(monkeypatch):
    import lunascope.components.save_edf as mod

    ctrl = _Controller()
    monkeypatch.setattr(mod, "SaveEDFDialog", _FakeDialog)
    monkeypatch.setattr(mod, "QDialog", type("_Dialog", (), {"Accepted": 1}))
    monkeypatch.setattr(mod, "QMessageBox", _FakeMessageBox)

    ctrl._save_edf_annots()

    assert ctrl.p.commands == [
        "WRITE edf-dir=/tmp/out",
        "WRITE-ANNOTS file=/tmp/out/S1.annot",
    ]
    assert _FakeMessageBox.last_critical is None
    assert _FakeMessageBox.last_info == (
        "Export complete",
        "EDF and/or annotation file(s) written successfully.",
    )
