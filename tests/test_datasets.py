from cat2cat.datasets import load_occup_panel


def test_load_occup_panel():
    occup_panel = load_occup_panel()

    assert occup_panel.shape == (3900, 16)
    assert list(occup_panel.columns) == [
        "id",
        "age",
        "sex",
        "edu",
        "exp",
        "district",
        "parttime",
        "salary",
        "code",
        "multiplier",
        "code4",
        "panel_id",
        "cohort",
        "quarter",
        "year",
        "quarter_num",
    ]
    assert occup_panel["panel_id"].nunique() < occup_panel.shape[0]
    assert set(occup_panel["quarter"].unique()) == {
        "2009Q1",
        "2009Q2",
        "2009Q3",
        "2009Q4",
        "2010Q1",
        "2010Q2",
        "2010Q3",
        "2010Q4",
    }
    assert occup_panel["multiplier"].gt(0).all()