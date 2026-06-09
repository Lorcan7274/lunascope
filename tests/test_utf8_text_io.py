import pandas as pd

from lunascope.components.annot_explorer_funcs import (
    ANNEX_CACHE_HEADER,
    save_annex_cache,
)
from lunascope.components.explorer_hypnoscope import save_hypnoscope_cache
from lunascope.components.moonbeam_dock import MoonbeamMixin, _load_cdir, _save_cdir


def test_annotation_explorer_cache_writes_utf8_chinese(tmp_path):
    path = tmp_path / "标注缓存.annot"
    save_annex_cache(
        str(path),
        {
            "subjects": [
                {
                    "id": "样本一",
                    "duration": 30.0,
                    "events": pd.DataFrame(
                        [{"Class": "睡眠阶段", "Start": 1.0, "Stop": 2.0}]
                    ),
                }
            ]
        },
    )

    text = path.read_text(encoding="utf-8")
    assert ANNEX_CACHE_HEADER in text
    assert "样本一" in text
    assert "睡眠阶段" in text


def test_hypnoscope_cache_writes_utf8_chinese_id(tmp_path):
    path = tmp_path / "睡眠缓存.tsv"
    save_hypnoscope_cache(
        str(path),
        [
            {
                "id": "样本一",
                "start_tod_secs": 0.0,
                "n_epochs": 3,
                "epochs": [0, 1, 2],
                "sol_secs": None,
                "tst_epochs": 2,
                "sleep_efficiency": 0.75,
            }
        ],
    )

    text = path.read_text(encoding="utf-8")
    assert "样本一" in text


def test_moonbeam_cdir_and_manifest_read_utf8_chinese(tmp_path, monkeypatch):
    cdir = tmp_path / "缓存"
    monkeypatch.setattr("lunascope.components.moonbeam_dock._CDIR_PATH", tmp_path / ".cdir")

    _save_cdir(str(cdir))

    assert _load_cdir() == str(cdir)

    manifest = "cohort\tsubcohort\t样本一\t数据/受试者一.edf\t标注/睡眠.annot\n"
    parsed = MoonbeamMixin._mb_parse_manifest(manifest)
    assert parsed["cohort"]["subcohort"]["样本一"]["edf"] == "数据/受试者一.edf"
    assert parsed["cohort"]["subcohort"]["样本一"]["annots"] == ["标注/睡眠.annot"]
