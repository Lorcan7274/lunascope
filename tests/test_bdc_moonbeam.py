from lunascope.components.moonbeam_dock import MoonbeamMixin


def test_bdc_groups_build_slist_rows_with_annotation(tmp_path):
    obj = object.__new__(MoonbeamMixin)
    obj._mb_cdir = str(tmp_path)
    obj._bdc_project = "study/project"
    group = {
        "recording": {"name": "subject-01.edf", "size": 4},
        "sidecars": [{"name": "subject-01.annot"}],
    }
    cache = obj._bdc_cache_dir()
    cache.mkdir(parents=True)
    (cache / "subject-01.edf").write_bytes(b"edf")
    (cache / "subject-01.annot").write_text("annotation", encoding="utf-8")

    rows = obj._bdc_build_slist_rows([group])

    assert rows == [["subject-01", str(cache / "subject-01.edf"), str(cache / "subject-01.annot")]]


def test_bdc_id_removes_compressed_edf_suffix():
    group = {"recording": {"name": "subject-02.edf.gz"}}

    assert MoonbeamMixin._bdc_id(group) == "subject-02"
