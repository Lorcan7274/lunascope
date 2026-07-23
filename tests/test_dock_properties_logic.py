from lunascope.components.cmaps import parse_cmap
from lunascope.components.dock_properties import _update_config_entries


def test_update_config_entries_writes_filter_and_preserves_order():
    text = """[sig]
A col=#111111
B

[ann]
X col=#222222
"""

    updated = _update_config_entries(
        text,
        "sig",
        ["B"],
        {"col": "#abcdef", "f": (1.0, 4.0)},
        order=["B", "A"],
    )
    parsed = parse_cmap(updated)

    assert parsed["sig_order"] == ["B", "A"]
    assert parsed["sig"]["B"]["col"] == "#abcdef"
    assert parsed["sig"]["B"]["f"] == (1.0, 4.0)
    assert "[ann]" in updated
    assert "X col=#222222" in updated


def test_update_config_entries_can_clear_signal_fields():
    text = "[sig]\nA col=#111111 f=1,4 ylim=-2,2 y=0,1\n"

    updated = _update_config_entries(
        text,
        "sig",
        ["A"],
        {"col": None, "f": None, "ylim": None, "y": None},
    )

    record = parse_cmap(updated)["sig"]["A"]
    assert record["col"] is None
    assert record["f"] is None
    assert record["ylim"] is None
    assert record["y"] is None
