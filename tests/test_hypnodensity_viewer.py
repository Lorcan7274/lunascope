import pandas as pd
import pytest
import numpy as np

from lunascope.components.hypnodensity_viewer import (
    apply_pc_orientation,
    display_hypnodensity,
    flatten_hdstats_tables,
    hdstats_feature_group,
    hdstats_feature_label,
    hdstats_analysis_metadata,
    hdstats_model_loadings,
    matching_hdstats_feature,
    parse_hdstats_model,
    thumbnail_time_window,
    thumbnail_duration_seconds,
)


def test_display_modes_preserve_probability_mass_and_use_purple_sleep():
    trace = np.array([[.1, .2, .3, .15, .25]], dtype=float)
    nrem, labels, colors = display_hypnodensity(trace, "nrem_rem_wake")
    assert labels == ("NR", "R", "W")
    assert np.allclose(nrem, [[.6, .15, .25]])
    wake_sleep, labels, colors = display_hypnodensity(trace, "wake_sleep")
    assert labels == ("W", "S")
    assert colors == ("#31AD52", "#9333ea")
    assert np.allclose(wake_sleep, [[.25, .75]])


def test_thumbnail_spt_window_uses_persisted_analysis_flanks_without_recalculation():
    trace = np.zeros((100, 5))
    window = thumbnail_time_window(trace, np.arange(100) * 5.0,
                                   {"HD_ANALYSIS_START_SEC": 100.0,
                                    "HD_ANALYSIS_STOP_SEC": 300.0}, "spt_flank")
    assert (window.start, window.stop) == (20, 61)


def test_thumbnail_duration_uses_clock_time_not_epoch_count():
    trace = np.zeros((4, 5))
    assert thumbnail_duration_seconds(trace, np.array([10., 15., 20., 25.]), slice(1, 3)) == 10.0


def test_feature_labels_keep_val_and_raw_in_separate_human_readable_groups():
    assert hdstats_feature_group("VAL__H|NSS=5") == "HDSTATS values (VAL)"
    assert hdstats_feature_group("RAW__H|NSS=5") == "HDSTATS raw values (RAW)"
    assert hdstats_feature_group("H__NSS=5") == "HDSTATS features"
    assert hdstats_feature_label("VAL__H|NSS=5") == "Value — H · state model: 5-state"
    assert matching_hdstats_feature("H__NSS/5", ["ID", "VAL__H__NSS/5"]) == "VAL__H__NSS/5"


def test_luna_model_reader_response_flattens_oriented_component_loadings():
    result = hdstats_model_loadings({
        "panels": [{"id": "AMBIGUITY", "singular_values": [2.0, 1.0]}],
        "components": [
            {"id": "PC_AM_1", "panel": "AMBIGUITY", "panel_pc": 1,
             "loadings": [{"feature": "H__NSS/5", "loading": 0.5},
                          {"feature": "CONF__NSS/5", "loading": -0.5}]},
            {"id": "PC_AM_2", "panel": "AMBIGUITY", "panel_pc": 2,
             "loadings": [{"feature": "H__NSS/5", "loading": 0.1},
                          {"feature": "CONF__NSS/5", "loading": 0.2}]},
        ],
    })
    assert result[["PC", "Feature", "Loading"]].to_dict("records") == [
        {"PC": "PC_AM_1", "Feature": "H__NSS/5", "Loading": 0.5},
        {"PC": "PC_AM_1", "Feature": "CONF__NSS/5", "Loading": -0.5},
        {"PC": "PC_AM_2", "Feature": "H__NSS/5", "Loading": 0.1},
        {"PC": "PC_AM_2", "Feature": "CONF__NSS/5", "Loading": 0.2},
    ]


def test_model_loadings_expose_raw_live_score_id_distinct_from_descriptor_id():
    # A descriptor renames "id" to a human label (e.g. AMIN_PC1) and may flip
    # orientation, but live pc-proj scoring (model.json alone) always reports
    # the fit-time auto id in the model's own component_id_style.
    result = hdstats_model_loadings({
        "spec": {"component_id_style": "r_safe"},
        "panels": [{"id": "AMIN", "singular_values": [2.0]}],
        "components": [
            {"id": "AMIN_PC1", "key": "AMIN/1", "panel": "AMIN", "panel_pc": 1,
             "label": "Posterior ambiguity", "orientation": -1,
             "loadings": [{"feature": "H__NSS/5", "loading": -0.5}]},
        ],
    })
    row = result.iloc[0]
    assert row["PC"] == "AMIN_PC1"
    assert row["PC raw ID"] == "AMIN_1"
    assert row["PC orientation"] == -1


def test_model_loadings_raw_id_uses_slash_for_legacy_style():
    result = hdstats_model_loadings({
        "spec": {"component_id_style": "legacy"},
        "panels": [{"id": "AMIN", "singular_values": [2.0]}],
        "components": [
            {"id": "AMIN/1", "panel": "AMIN", "panel_pc": 1,
             "loadings": [{"feature": "H__NSS/5", "loading": 0.5}]},
        ],
    })
    assert result.iloc[0]["PC raw ID"] == "AMIN/1"
    assert result.iloc[0]["PC orientation"] == 1


def test_apply_pc_orientation_rekeys_and_sign_flips_live_scores():
    lookup = {"AMIN_1": "AMIN_PC1", "TR_1": "TR_PC1"}
    metadata = {"AMIN_PC1": {"PC orientation": -1}, "TR_PC1": {"PC orientation": 1}}
    oriented = apply_pc_orientation({"AMIN_1": 1.25, "TR_1": 0.5}, lookup, metadata)
    assert oriented == {"AMIN_PC1": -1.25, "TR_PC1": 0.5}


def test_apply_pc_orientation_passes_through_unmatched_scores_unchanged():
    # No model loaded yet, or a PC absent from the lookup: identity, as before.
    assert apply_pc_orientation({"AMIN_1": 2.0}, {}, {}) == {"AMIN_1": 2.0}


def test_model_reader_uses_optional_luna_descriptor(tmp_path, monkeypatch):
    model_path = tmp_path / "study.model.json"
    descriptor_path = tmp_path / "study.desc.json"
    model_path.touch(); descriptor_path.touch()
    calls = []

    class FakeProject:
        def read_hdstats_pc_descriptor(self, model, descriptor):
            calls.append((model, descriptor))
            return {"panels": [], "components": []}

        def read_hdstats_pc_model(self, model):
            raise AssertionError("descriptor reader should be used")

    monkeypatch.setattr("lunascope.components.hypnodensity_viewer.lp.proj", lambda **_kwargs: FakeProject())
    assert parse_hdstats_model(str(model_path)).empty
    assert calls == [(str(model_path), str(descriptor_path))]


def test_collects_projected_scores_from_hdstats_pc_table():
    class FakeInstance:
        def strata(self):
            return pd.DataFrame({"Command": ["HDSTATS"], "Strata": ["PC"]})

        def table(self, command, strata):
            assert (command, strata) == ("HDSTATS", "PC")
            return pd.DataFrame({"PC": ["PC_AM_1"], "SCORE": [1.25]})

    values, pcs = flatten_hdstats_tables(FakeInstance(), "id-1")
    assert pcs == {"PC_AM_1": 1.25}
    assert values["SCORE__PC|PC=PC_AM_1"] == 1.25


def test_analysis_metadata_uses_hdstats_qc_and_emitted_soft_spt_annotations():
    class FakeInstance:
        def strata(self):
            return pd.DataFrame({"Command": ["HDSTATS"], "Strata": ["QC"]})

        def table(self, command, strata):
            return pd.DataFrame({"HD_ANALYSIS": ["spt"], "HD_ANALYSIS_START_SEC": [300.0],
                                 "HD_ANALYSIS_STOP_SEC": [25200.0], "HD_ANALYSIS_FALLBACK": [0]})

        def fetch_annots(self, names):
            return pd.DataFrame({"Class": ["__hdviewer_spt_sleep_onset", "__hdviewer_spt_final_wake"],
                                 "Start": [600.0, 24900.0], "Stop": [600.0, 24900.0]})

    meta = hdstats_analysis_metadata(FakeInstance(), "__hdviewer_spt")
    assert meta == {"HD_ANALYSIS": "spt", "HD_ANALYSIS_START_SEC": 300.0,
                    "HD_ANALYSIS_STOP_SEC": 25200.0, "HD_ANALYSIS_FALLBACK": 0,
                    "soft_onset_sec": 600.0, "soft_final_wake_sec": 24900.0}


def test_individual_pc_panel_follows_sorted_rows_and_clears_filtered_selection(qapp, monkeypatch):
    from types import SimpleNamespace
    from PySide6 import QtCore, QtWidgets
    from lunascope.components.hypnodensity_viewer import HypnodensityViewerDialog

    monkeypatch.setattr(HypnodensityViewerDialog, '_read_model', lambda self: False)
    parent = QtWidgets.QWidget()
    dialog = HypnodensityViewerDialog(SimpleNamespace(ui=parent))
    try:
        dialog._model_pcs = ['PC_A', 'PC_B', 'PC_MISSING']
        dialog._model_pc_metadata = {'PC_A': {'PC label': 'Example', 'PC description': 'Details'}}
        dialog.loading_threshold.setValue(100.0)
        dialog.session.browser_values = ['PC_A', 'PC_B', 'PC_MISSING']
        dialog.session.features = pd.DataFrame({'ID': ['A', 'B']})
        dialog.session.pcs = pd.DataFrame({'ID': ['A', 'B'], 'PC_A': [1.25, -4.5], 'PC_B': [np.nan, 0.]})
        dialog._refresh_views()
        dialog.browser.sortByColumn(0, QtCore.Qt.DescendingOrder)
        dialog.browser.setCurrentIndex(dialog.browser.model().index(0, 0))
        assert dialog.session.browser_values == ['PC_A', 'PC_B']
        assert dialog.browser.model().columnCount() == 5
        assert dialog.browser.model().headerData(3, QtCore.Qt.Horizontal) == 'PC\nA'
        assert dialog.browser.model().headerData(3, QtCore.Qt.Horizontal, QtCore.Qt.ToolTipRole) == 'Example\nPC_A'
        assert dialog.individual_pc_title.text() == 'PC scores — B'
        assert dialog.individual_score_threshold.value() == 0.5
        assert dialog.individual_pcs.topLevelItemCount() == 1
        dialog.individual_score_threshold.setValue(4.5)
        assert dialog.individual_pcs.topLevelItemCount() == 1  # Boundary is inclusive.
        dialog.individual_score_threshold.setValue(4.6)
        assert dialog.individual_pcs.topLevelItemCount() == 0
        assert dialog.loading_threshold.value() == 100.0
        dialog.individual_score_threshold.setValue(0.0)
        assert dialog.individual_pcs.topLevelItemCount() == 3
        first = dialog.individual_pcs.topLevelItem(0)
        assert first.text(1) == '-4.50'  # Display is not clipped to the color limit.
        assert first.text(2) == 'Example'
        negative = first.background(1).color()
        assert negative.red() > negative.blue()
        assert dialog.individual_pcs.topLevelItem(2).text(1) == '—'
        # Scores remain available even when no trace is cached.
        assert dialog._selected_id == 'B'
        dialog._refresh_views()
        assert dialog._selected_id == 'B'
        proxy = dialog.browser.model()
        source = proxy.sourceModel()
        dialog.browser.setCurrentIndex(proxy.mapFromSource(source.index(0, 0)))
        assert dialog._selected_id == 'A'
        positive = dialog.individual_pcs.topLevelItem(0).background(1).color()
        assert positive.blue() > positive.red()
        assert dialog.individual_pcs.topLevelItem(1).text(1) == '—'
        dialog._sort_loading_tree(dialog.individual_pcs, 1)
        dialog.individual_score_threshold.setValue(0.5)
        assert dialog.individual_pcs.topLevelItemCount() == 1
        assert dialog.individual_pcs.topLevelItem(0).text(1) == '+1.25'
        assert dialog.individual_pcs._loading_sort == (1, True)
        dialog.browser_filter.setText('no-such-person')
        assert dialog.individual_pcs.topLevelItemCount() == 0
        assert dialog._selected_id is None
        assert dialog.browser_split.count() == 2
    finally:
        dialog.close()
        dialog.deleteLater()
        parent.deleteLater()


def test_browser_chooser_uses_model_features_and_clear_keeps_empty_slots(qapp, monkeypatch):
    from types import SimpleNamespace
    from PySide6 import QtCore, QtWidgets
    from lunascope.components.hypnodensity_viewer import HypnodensityViewerDialog

    monkeypatch.setattr(HypnodensityViewerDialog, '_read_model', lambda self: False)
    parent = QtWidgets.QWidget()
    dialog = HypnodensityViewerDialog(SimpleNamespace(ui=parent))
    try:
        feature = 'H.NSS_3'
        dialog._model_features = [feature, 'CONF.NSS_3']
        dialog._model_feature_metadata = {
            feature: {'Feature label': 'Entropy [3-stage]', 'Feature description': 'Hypnodensity entropy (3-stage)'},
            'CONF.NSS_3': {'Feature label': 'Confidence [3-stage]', 'Feature description': 'Posterior confidence (3-stage)'},
        }
        dialog.session.features = pd.DataFrame({
            'ID': ['A', 'B'],
            f'RAW__FTR_PC|FTR={feature}|PC=PC_A': [0.4, np.nan],
            f'RAW__FTR_PC|FTR={feature}|PC=PC_B': [0.4, 0.6],
            f'VAL__FTR_PC|FTR={feature}|PC=PC_A': [999., 999.],
        })
        dialog.session.pcs = pd.DataFrame({'ID': ['A', 'B'], 'PC_A': [1., 2.]})
        measures = dialog._browser_measures()
        assert list(measures.columns) == ['ID', feature, 'CONF.NSS_3']
        assert measures[feature].tolist() == [0.4, 0.6]
        assert measures['CONF.NSS_3'].isna().all()
        dialog.session.browser_values = [feature, 'PC_A']
        dialog._refresh_views()
        assert dialog._browser_display_columns[feature] == 'Entropy (3-stage)'

        def choose_and_clear(chooser):
            tree = chooser.findChild(QtWidgets.QTreeWidget)
            titles = [tree.topLevelItem(i).text(0) for i in range(tree.topLevelItemCount())]
            assert titles == ['Measures / features', 'PC scores']
            group = tree.topLevelItem(0)
            assert group.childCount() == 2
            assert group.child(0).text(0) == 'Entropy'
            assert group.child(0).text(1) == '3-stage'
            assert group.child(0).data(0, QtCore.Qt.UserRole) == feature
            buttons = chooser.findChild(QtWidgets.QDialogButtonBox)
            next(b for b in buttons.buttons() if b.text() == 'Clear').click()
            assert group.child(0).checkState(0) == QtCore.Qt.Unchecked
            return QtWidgets.QDialog.Accepted

        monkeypatch.setattr(QtWidgets.QDialog, 'exec', choose_and_clear)
        dialog.choose_browser_values()
        assert dialog.session.browser_values == []
        assert dialog.browser.model().columnCount() == 3  # ID, thumbnail, duration only.
    finally:
        dialog.close()
        dialog.deleteLater()
        parent.deleteLater()


@pytest.mark.parametrize('stop', [False, True])
def test_hdstats_pool_orders_results_and_unlocks_after_stop(qapp, monkeypatch, stop):
    from concurrent.futures import Future
    from types import SimpleNamespace
    from PySide6 import QtCore, QtWidgets
    from lunascope.components import hypnodensity_viewer as hd, anal

    monkeypatch.setattr(hd.HypnodensityViewerDialog, '_read_model', lambda self: False)
    parent = QtWidgets.QWidget()
    events, submitted = [], []
    ctrl = SimpleNamespace(ui=parent, _busy=False,
        _buttons=lambda enabled: events.append(('buttons', enabled)),
        lock_ui=lambda message: events.append(('lock', True)),
        unlock_ui=lambda: events.append(('unlock', True)),
        _project_eval_cancel_shortcuts=lambda: [])
    dialog = hd.HypnodensityViewerDialog(ctrl)

    class Pool:
        def __init__(self, **kwargs):
            assert kwargs['initializer'] is anal._init_project_eval_child
            assert kwargs['mp_context'].get_start_method() == 'spawn'
        def submit(self, fn, task):
            assert fn is hd._hdstats_record_worker
            future = Future()
            submitted.append(task['ordinal'])
            if stop and task['ordinal'] == 1:
                # Complete one record then request Stop before the first poll;
                # the other in-flight record must not become an exclusion.
                future.set_result({'ordinal': 1, 'ID': 'B'})
                def request_stop():
                    progress = dialog.findChild(hd._HDStatsProgress)
                    assert progress.windowModality() == QtCore.Qt.ApplicationModal
                    assert ctrl._busy
                    progress.reject()
                QtCore.QTimer.singleShot(10, request_stop)
            elif not stop:
                delay = 150 if task['ordinal'] == 1 else 1
                QtCore.QTimer.singleShot(delay, lambda: future.set_result(
                    {'ordinal': task['ordinal'], 'ID': task['sample_row'][0]}))
            return future
        def shutdown(self, **kwargs):
            events.append(('shutdown', True))

    monkeypatch.setattr(hd, 'ProcessPoolExecutor', Pool)
    monkeypatch.setattr(anal, '_terminate_process_pool', lambda pool: events.append(('terminate', True)) or False)
    tasks = [{'ordinal': i, 'sample_row': [sid]} for i, sid in enumerate(['B', 'A', 'C'], 1)]
    try:
        results, cancelled = dialog._run_hdstats_tasks(tasks, 2, 3, 0)
        assert cancelled == stop
        assert [r['ordinal'] for r in results] == ([1] if stop else [1, 2, 3])
        assert submitted == ([1, 2] if stop else [1, 2, 3])
        assert not ctrl._busy
        assert events[-2:] == [('unlock', True), ('buttons', True)]
        assert ('terminate', True) in events if stop else ('terminate', True) not in events
        dialog.session.rows = [['B'], ['A']]
        dialog.session.features = pd.DataFrame({'ID': ['A', 'B']})
        dialog.session.pcs = pd.DataFrame({'ID': ['A', 'B'], 'PC': [1., 2.]})
        assert dialog._merged().ID.tolist() == ['B', 'A']
    finally:
        dialog.close()
        dialog.deleteLater()
        parent.deleteLater()


def test_hdstats_calculate_stop_resume_preserves_sample_order(qapp, monkeypatch, tmp_path):
    from types import SimpleNamespace
    from PySide6 import QtCore, QtWidgets
    from lunascope.components import hypnodensity_viewer as hd

    monkeypatch.setattr(hd.HypnodensityViewerDialog, '_read_model', lambda self: True)
    source = tmp_path / 'source.edf'
    source.write_bytes(b'test source signature')
    rows = [[sid, str(source)] for sid in ['B', 'A', 'C']]
    parent = QtWidgets.QWidget()
    parent.txt_param = object()
    ctrl = SimpleNamespace(ui=parent, _busy=False,
        _sample_rows_from_source_model=lambda: rows,
        _parse_tab_pairs=lambda edit: [('path', str(tmp_path))],
        _project_eval_config_dialog=lambda count: {'workers': 2})
    dialog = hd.HypnodensityViewerDialog(ctrl)
    signature = {'sha256': 'same model'}
    dialog.session.model_signature = signature.copy()
    dialog.session.results_model_signature = signature.copy()
    dialog.session.features = pd.DataFrame({'ID': ['B'], 'feature': [2.]})
    dialog.session.pcs = pd.DataFrame({'ID': ['B'], 'PC': [2.]})
    dialog.session.traces = {'B': np.zeros((2, 5))}
    dialog.session.source_signatures = {'B': hd.source_signature(str(source))}
    runs = []

    def run(tasks, workers, total, reused):
        runs.append([t['sample_row'][0] for t in tasks])
        assert total == 3
        assert tasks[0]['param'] == [('path', str(tmp_path))]
        sid = 'C' if len(runs) == 1 else 'A'
        task = next(t for t in tasks if t['sample_row'][0] == sid)
        return [{'ordinal': task['ordinal'], 'ID': sid, 'features': {'feature': 1.},
                 'scores': {'PC': 1.}, 'trace': np.zeros((2, 5)), 'invalid': np.zeros(2, dtype=bool),
                 'axis': np.arange(2), 'analysis': {}}], len(runs) == 1

    monkeypatch.setattr(dialog, '_run_hdstats_tasks', run)
    try:
        dialog.calculate()
        assert dialog.session.features.ID.tolist() == ['B', 'C']
        assert dialog.session.exclusions == []
        assert '1 unfinished' in dialog.status.text()
        dialog.calculate()
        assert runs == [['A', 'C'], ['A']]
        assert dialog.session.pcs.ID.tolist() == ['B', 'A', 'C']
        assert dialog.session.features.ID.tolist() == ['B', 'A', 'C']
        assert list(dialog.session.traces) == ['B', 'A', 'C']
        assert dialog._merged().ID.tolist() == ['B', 'A', 'C']
    finally:
        dialog.close()
        dialog.deleteLater()
        parent.deleteLater()


def test_hdstats_worker_isolates_record_and_cleans_up_on_failure(monkeypatch):
    from types import SimpleNamespace
    from lunascope.components import hypnodensity_viewer as hd, anal
    events = []
    proj = SimpleNamespace(
        clear=lambda: events.append('clear'),
        clear_vars=lambda: events.append('clear_vars'),
        reinit=lambda: events.append('reinit'),
        var=lambda name, value: events.append((name, value)),
        eng=SimpleNamespace(set_sample_list=lambda rows: events.append(rows)),
        inst=lambda sid: SimpleNamespace(edf=SimpleNamespace(channels=lambda: [])))
    monkeypatch.setattr(anal, '_PROJECT_EVAL_CHILD_PROJ', proj)
    task = {'ordinal': 3, 'sample_row': ['C', 'file.edf'], 'param': [('path', '/data')], 'model_path': 'model.json'}
    result = hd._hdstats_record_worker(task)
    assert result['ordinal'] == 3 and result['ID'] == 'C'
    assert 'missing PP_N1' in result['error']
    assert events == ['clear', [['C', 'file.edf']], 'clear_vars', 'reinit', ('path', '/data'), 'clear']


def test_flagged_individual_keeps_displayed_values_and_survives_cache(qapp, monkeypatch, tmp_path):
    from types import SimpleNamespace
    from PySide6 import QtCore, QtWidgets
    from lunascope.components import hypnodensity_viewer as hd

    monkeypatch.setattr(hd.HypnodensityViewerDialog, '_read_model', lambda self: False)
    dialog = hd.HypnodensityViewerDialog(SimpleNamespace(ui=QtWidgets.QWidget()))
    try:
        dialog.session.features = pd.DataFrame({'ID': ['A', 'B']})
        dialog.session.pcs = pd.DataFrame({'ID': ['A', 'B'], 'PC_A': [1.25, -2.5]})
        dialog.session.browser_values = ['PC_A']
        dialog._refresh_views()
        index = dialog.browser.model().index(1, 0)
        dialog._flag_individual(index)
        assert dialog.session.flagged['B']['values']['PC A'] == '-2.500'
        assert index.data(QtCore.Qt.BackgroundRole) is not None
        assert not index.data(QtCore.Qt.DecorationRole).isNull()
        assert dialog.flagged_button.text() == 'Flagged (1)…'
        dialog.browser.setCurrentIndex(index)
        dialog._toggle_selected_flag()
        assert not dialog.session.flagged
        dialog._toggle_selected_flag()
        assert 'B' in dialog.session.flagged
        dialog.session.flagged['B']['note'] = 'Interesting trace'
        path = tmp_path / 'cohort.hdstats.json'
        assert dialog.save_cache(str(path), quiet=True)
        monkeypatch.setattr(hd, 'open_file_name', lambda *_args: (str(path), ''))
        dialog.open_cache()
        assert dialog.session.flagged['B']['note'] == 'Interesting trace'
        assert dialog.session.flagged['B']['values']['PC A'] == '-2.500'
        dialog._remove_flag('B')
        assert not dialog.session.flagged
    finally:
        dialog.close()


def test_flagged_open_uses_current_sample_list_and_reports_missing_id(qapp, monkeypatch):
    from types import SimpleNamespace
    from PySide6 import QtWidgets
    from lunascope.components import hypnodensity_viewer as hd

    monkeypatch.setattr(hd.HypnodensityViewerDialog, '_read_model', lambda self: False)
    main = QtWidgets.QWidget()
    dock = QtWidgets.QWidget(main)
    view = QtWidgets.QTableView(main)
    calls = []
    controller = SimpleNamespace(ui=main, _restore_attached_subject=lambda sid: calls.append(sid) or sid == 'A')
    main.dock_slist = dock
    main.tbl_slist = view
    dialog = hd.HypnodensityViewerDialog(controller)
    messages = []
    monkeypatch.setattr(QtWidgets.QMessageBox, 'information', lambda *_args: messages.append(_args[-1]))
    try:
        assert dialog._open_in_sample_list('A') is True
        assert dialog._open_in_sample_list('B') is False
        assert calls == ['A', 'B']
        assert 'B is not in the current sample list.' in messages
    finally:
        dialog.close()


def test_individual_pc_can_be_inserted_as_first_browser_value(qapp, monkeypatch):
    from types import SimpleNamespace
    from PySide6 import QtCore, QtWidgets
    from lunascope.components import hypnodensity_viewer as hd

    monkeypatch.setattr(hd.HypnodensityViewerDialog, '_read_model', lambda self: False)
    dialog = hd.HypnodensityViewerDialog(SimpleNamespace(ui=QtWidgets.QWidget()))
    try:
        dialog.session.features = pd.DataFrame({'ID': ['A']})
        dialog.session.pcs = pd.DataFrame({'ID': ['A'], 'PC_A': [1.0], 'PC_B': [2.0], 'PC_C': [3.0]})
        dialog.session.browser_values = ['PC_A', 'PC_B']
        dialog._refresh_views()
        dialog.browser.setCurrentIndex(dialog.browser.model().index(0, 0))
        dialog._add_individual_pc_to_browser('PC_C')
        assert dialog.session.browser_values == ['PC_C', 'PC_A']
        assert dialog.browser.model().headerData(3, QtCore.Qt.Horizontal) == 'PC\nC'
        dialog._add_individual_pc_to_browser('PC_A')
        assert dialog.session.browser_values == ['PC_C', 'PC_A']
    finally:
        dialog.close()


def test_individual_plot_keyboard_zoom_pan_and_reset_keep_overview_fixed(qapp, monkeypatch):
    from types import SimpleNamespace
    from PySide6 import QtWidgets
    from lunascope.components import hypnodensity_viewer as hd

    monkeypatch.setattr(hd.HypnodensityViewerDialog, '_read_model', lambda self: False)
    dialog = hd.HypnodensityViewerDialog(SimpleNamespace(ui=QtWidgets.QWidget()))
    try:
        dialog.session.features = pd.DataFrame({'ID': ['A']})
        trace = np.tile([.1, .2, .4, .2, .1], (120, 1))
        dialog.session.traces['A'] = trace
        dialog.session.time_axes['A'] = np.arange(120) * 30.0
        dialog.session.analysis['A'] = {'soft_onset_sec': 300., 'soft_final_wake_sec': 3300.}
        dialog._refresh_views()
        dialog.browser.setCurrentIndex(dialog.browser.model().index(0, 0))
        full = dialog.canvas.ax.get_xlim()
        overview = dialog._plot_window_patch
        dialog._plot_key(SimpleNamespace(key='up'))
        zoomed = dialog.canvas.ax.get_xlim()
        assert zoomed[1] - zoomed[0] < full[1] - full[0]
        assert overview.get_visible()
        assert dialog._plot_duration_text.get_visible()
        assert 'SPT' in dialog._plot_duration_text.get_text()
        dialog._plot_key(SimpleNamespace(key='right'))
        assert dialog.canvas.ax.get_xlim()[0] > zoomed[0]
        dialog._set_plot_window(*full)
        assert not overview.get_visible()
        assert not dialog._plot_duration_text.get_visible()
    finally:
        dialog.close()
