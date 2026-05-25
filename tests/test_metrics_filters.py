from lunascope.components.metrics import _restore_signal_filter_defaults


def test_restore_signal_filter_defaults_keeps_current_channels_and_prunes_stale_user_filters():
    fmap = {
        "C3": "Delta",
        "C4": "User",
        "OLD": "Theta",
        "BAD": "Missing",
    }
    fmap_frqs = {
        "Delta": [1, 4],
        "Theta": [4, 8],
        "User": [],
    }
    user_fmap_frqs = {
        "C4": [12, 15],
        "OLD": [8, 10],
    }

    restored = _restore_signal_filter_defaults(
        ["C3", "C4", "O1"],
        fmap,
        fmap_frqs,
        user_fmap_frqs,
    )

    assert restored == {
        "C3": "Delta",
        "C4": "User",
    }
    assert fmap == restored
    assert user_fmap_frqs == {
        "C4": [12, 15],
    }


def test_restore_signal_filter_defaults_drops_unknown_filter_codes():
    fmap = {
        "C3": "NotAFilter",
    }
    fmap_frqs = {
        "Delta": [1, 4],
        "User": [],
    }
    user_fmap_frqs = {}

    restored = _restore_signal_filter_defaults(
        ["C3"],
        fmap,
        fmap_frqs,
        user_fmap_frqs,
    )

    assert restored == {}
    assert fmap == {}
