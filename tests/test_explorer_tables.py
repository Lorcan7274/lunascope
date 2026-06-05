import pandas as pd

from lunascope.components.explorer_tables import (
    _publication_table_html,
    _publication_table_plan,
    _summary_stat_rows,
)


def test_summary_stat_rows_unstratified_defaults():
    df = pd.DataFrame(
        {
            "ID": ["S1", "S2", "S3"],
            "TST": [300.0, 360.0, None],
            "SE": [80.0, 90.0, 100.0],
        }
    )

    out = _summary_stat_rows(df, ["TST", "SE"])

    assert list(out.columns) == ["Measure", "N", "Mean", "SD"]
    tst = out[out["Measure"] == "TST"].iloc[0]
    assert tst["N"] == 2
    assert tst["Mean"] == 330.0
    assert round(tst["SD"], 3) == round(42.4264068712, 3)


def test_summary_stat_rows_one_and_two_factor_groups():
    df = pd.DataFrame(
        {
            "Group": ["A", "A", "B", "B"],
            "Sex": ["F", "M", "F", "M"],
            "TST": [300.0, 360.0, 420.0, 480.0],
        }
    )

    out = _summary_stat_rows(df, ["TST"], ["Group", "Sex"], ["N", "Mean"])

    assert list(out.columns) == ["Group", "Sex", "Measure", "N", "Mean"]
    assert len(out) == 4
    row = out[(out["Group"] == "B") & (out["Sex"] == "M")].iloc[0]
    assert row["N"] == 1
    assert row["Mean"] == 480.0


def test_summary_stat_rows_coerces_numeric_like_strings_and_ignores_text_measures():
    df = pd.DataFrame(
        {
            "Arm": ["C", "C", "T"],
            "Value": ["1,000", "1200", "bad"],
        }
    )

    out = _summary_stat_rows(df, ["Value"], ["Arm"], ["N", "Mean", "SD"])

    control = out[out["Arm"] == "C"].iloc[0]
    treated = out[out["Arm"] == "T"].iloc[0]
    assert control["N"] == 2
    assert control["Mean"] == 1100.0
    assert treated["N"] == 0
    assert pd.isna(treated["Mean"])


def test_publication_plan_unstratified_keeps_stats_as_columns():
    df = pd.DataFrame(
        {
            "ID": ["S1", "S2", "S3"],
            "TST": [300.0, 360.0, None],
            "SE": [80.0, 90.0, 100.0],
        }
    )
    summary = _summary_stat_rows(df, ["TST", "SE"], stats=["N", "Mean", "SD"])

    plan = _publication_table_plan(summary, [], ["N", "Mean", "SD"])

    assert plan[0]["columns"] == ["Measure", "N", "Mean", "SD"]
    assert plan[0]["rows"][0][0] == "TST"
    assert plan[0]["rows"][0][1:] == ["2", "330", "42.426"]


def test_publication_plan_one_factor_uses_levels_as_readable_columns():
    df = pd.DataFrame(
        {
            "Arm": ["Control", "Control", "Treat", "Treat"],
            "TST": [300.0, 360.0, 420.0, 480.0],
        }
    )
    summary = _summary_stat_rows(df, ["TST"], ["Arm"], ["N", "Mean", "SD"])

    plan = _publication_table_plan(summary, ["Arm"], ["N", "Mean", "SD"])

    assert plan[0]["layout"] == "wide"
    assert plan[0]["columns"] == ["Measure", "Control", "Treat"]
    assert plan[0]["rows"] == [["TST", "n=2; 330 (42.426)", "n=2; 450 (42.426)"]]


def test_publication_plan_two_factors_sections_first_factor():
    df = pd.DataFrame(
        {
            "Arm": ["A", "A", "B", "B"],
            "Sex": ["F", "M", "F", "M"],
            "TST": [300.0, 360.0, 420.0, 480.0],
        }
    )
    summary = _summary_stat_rows(df, ["TST"], ["Arm", "Sex"], ["N", "Mean"])

    plan = _publication_table_plan(summary, ["Arm", "Sex"], ["N", "Mean"])

    assert [section["title"] for section in plan] == ["Arm: A", "Arm: B"]
    assert plan[0]["columns"] == ["Measure", "F", "M"]
    assert plan[0]["rows"] == [["TST", "n=1; 300", "n=1; 360"]]


def test_publication_html_escapes_text_and_includes_sections():
    summary = pd.DataFrame(
        {
            "Group": ["A&B"],
            "Measure": ["Power <delta>"],
            "N": [1],
            "Mean": [2.5],
        }
    )

    html = _publication_table_html(summary, ["Group"], ["N", "Mean"], title="Demo <Table>")

    assert "Demo &lt;Table&gt;" in html
    assert "A&amp;B" in html
    assert "Power &lt;delta&gt;" in html
