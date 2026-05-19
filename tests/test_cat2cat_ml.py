from cat2cat import cat2cat
from cat2cat import cat2cat_ml_run
from cat2cat.dataclass import cat2cat_data, cat2cat_mappings, cat2cat_ml
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from cat2cat.datasets import load_trans, load_occup
from numpy.random import seed
from numpy import isfinite, nan
import pytest

trans = load_trans()
occup = load_occup()
o_old = occup.loc[occup.year == 2008, :].copy()
o_new = occup.loc[occup.year == 2010, :].copy()


def test_cat2cat_ml_run_repr():
    mappings = cat2cat_mappings(trans=trans, direction="backward")
    ml = cat2cat_ml(
        occup.loc[occup.year >= 2010, :].copy(),
        "code",
        ["salary", "age", "edu", "sex", "parttime"],
        [
            DecisionTreeClassifier(random_state=1234),
            LinearDiscriminantAnalysis(),
        ],
    )
    expected = repr(cat2cat_ml_run(mappings=mappings, ml=ml))
    assert "Average Accuracy naive" in expected
    assert "Average Brier naive" in expected
    assert "Average Mean P(true) naive" in expected
    assert "Average Accuracy DecisionTreeClassifier" in expected
    assert "Average Brier LinearDiscriminantAnalysis" in expected

    expected = repr(cat2cat_ml_run(mappings=mappings, ml=ml, test_prop=0.3))
    assert "Test sample size: 30.0" in expected

    expected = repr(
        cat2cat_ml_run(mappings=mappings, ml=ml, test_prop=0.9, split_seed=1234)
    )
    assert "Test sample size: 90.0" in expected

    mappings = cat2cat_mappings(trans=trans, direction="forward")
    ml = cat2cat_ml(
        occup.loc[occup.year <= 2008, :].copy(),
        "code",
        ["salary", "age", "edu", "sex"],
        [DecisionTreeClassifier(random_state=1234), LinearDiscriminantAnalysis()],
    )
    expected = repr(cat2cat_ml_run(mappings=mappings, ml=ml, test_prop=0.3))
    assert "Average Brier DecisionTreeClassifier" in expected


def test_cat2cat_ml_run_get_raw():
    mappings = cat2cat_mappings(trans=trans, direction="backward")
    ml = cat2cat_ml(
        occup.loc[occup.year >= 2010, :].copy(),
        "code",
        ["salary", "age", "edu", "sex"],
        [DecisionTreeClassifier(random_state=1234), LinearDiscriminantAnalysis()],
    )
    expected = cat2cat_ml_run(mappings=mappings, ml=ml).get_raw()["7431"]
    actual = {
        "naive": nan,
        "freq": nan,
        "naive_brier": nan,
        "naive_mean_prob": nan,
        "freq_brier": nan,
        "freq_mean_prob": nan,
        "DecisionTreeClassifier": nan,
        "DecisionTreeClassifier_brier": nan,
        "DecisionTreeClassifier_mean_prob": nan,
        "LinearDiscriminantAnalysis": nan,
        "LinearDiscriminantAnalysis_brier": nan,
        "LinearDiscriminantAnalysis_mean_prob": nan,
    }
    assert str(actual) == str(expected)


def test_cat2cat_ml_run_gaussian_nb_metrics():
    mappings = cat2cat_mappings(trans=trans, direction="backward")
    ml = cat2cat_ml(
        occup.loc[occup.year >= 2010, :].copy(),
        "code",
        ["salary", "age", "edu", "sex", "parttime"],
        [GaussianNB()],
    )
    res = cat2cat_ml_run(mappings=mappings, ml=ml)
    raw = res.get_raw()
    non_missing = [g for g in raw.values() if isfinite(g.get("GaussianNB", nan))]

    assert len(non_missing) > 0
    assert "GaussianNB" in res.mean_acc
    assert "GaussianNB" in res.mean_brier
    assert "GaussianNB" in res.mean_prob
    assert all("GaussianNB_brier" in g for g in raw.values())
    assert all("GaussianNB_mean_prob" in g for g in raw.values())


def test_cat2cat_ml_categorical_feature_and_fallback():
    data = cat2cat_data(o_old, o_new, "code", "code", "year")
    mappings = cat2cat_mappings(trans, "backward")
    ml_data = occup.loc[occup.year >= 2010, :].copy()
    ml_data["edu_group"] = ml_data["edu"].astype(str)
    target_old = o_old.copy()
    target_new = o_new.copy()
    target_old["edu_group"] = target_old["edu"].astype(str)
    target_new["edu_group"] = target_new["edu"].astype(str)
    data = cat2cat_data(target_old, target_new, "code", "code", "year")
    ml = cat2cat_ml(
        ml_data,
        "code",
        ["salary", "age", "edu_group"],
        [GaussianNB()],
        on_fail="naive",
        fail_warn=False,
    )

    c2c = cat2cat(data, mappings, ml)

    assert "wei_GaussianNB_c2c" in c2c["old"].columns
    assert int(c2c["old"].groupby("index_c2c")["wei_GaussianNB_c2c"].sum().round().sum()) == o_old.shape[0]


def test_cat2cat_ml_fallback_error():
    data = cat2cat_data(o_old, o_new, "code", "code", "year")
    mappings = cat2cat_mappings(trans, "backward")
    ml = cat2cat_ml(
        occup.loc[occup.year >= 2010, :].copy(),
        "code",
        ["salary", "age"],
        [KNeighborsClassifier(n_neighbors=100000)],
        on_fail="error",
        fail_warn=False,
    )

    with pytest.raises(RuntimeError, match="ML weights failed"):
        cat2cat(data, mappings, ml)


def test_cat2cat_ml_fallback_freq_replaces_failed_weights():
    data = cat2cat_data(o_old, o_new, "code", "code", "year")
    mappings = cat2cat_mappings(trans, "backward")
    ml = cat2cat_ml(
        occup.loc[occup.year >= 2010, :].copy(),
        "code",
        ["salary", "age"],
        [KNeighborsClassifier(n_neighbors=100000)],
        on_fail="freq",
        fail_warn=False,
    )

    c2c = cat2cat(data, mappings, ml)
    col = "wei_KNeighborsClassifier_c2c"
    target = c2c["old"]

    assert target[col].notna().all()
    assert target.groupby("index_c2c")[col].sum().round(10).eq(1).all()
    assert (target[col] - target["wei_freq_c2c"]).abs().max() < 1e-12


def test_cat2cat_ml_fallback_na_keeps_missing_weights():
    data = cat2cat_data(o_old, o_new, "code", "code", "year")
    mappings = cat2cat_mappings(trans, "backward")
    ml = cat2cat_ml(
        occup.loc[occup.year >= 2010, :].copy(),
        "code",
        ["salary", "age"],
        [KNeighborsClassifier(n_neighbors=100000)],
        on_fail="na",
        fail_warn=False,
    )

    c2c = cat2cat(data, mappings, ml)
    col = "wei_KNeighborsClassifier_c2c"
    target = c2c["old"]

    assert target[col].isna().any()


def test_cat2cat_ml_run_kwargs_validation():
    mappings = cat2cat_mappings(trans=trans, direction="backward")
    ml = cat2cat_ml(
        occup.loc[occup.year >= 2010, :].copy(),
        "code",
        ["salary", "age", "edu", "sex"],
        [DecisionTreeClassifier(random_state=1234)],
    )

    with pytest.raises(ValueError, match="possible kwargs"):
        cat2cat_ml_run(mappings=mappings, ml=ml, wrong_key=1)

    with pytest.raises(ValueError, match="test_prop"):
        cat2cat_ml_run(mappings=mappings, ml=ml, test_prop=1)

    with pytest.raises(ValueError, match="min_match"):
        cat2cat_ml_run(mappings=mappings, ml=ml, min_match=1)


def test_cat2cat_ml_run_rejects_low_mapping_coverage():
    bad_trans = trans.copy()
    bad_trans["new"] = "__missing_category__"
    mappings = cat2cat_mappings(trans=bad_trans, direction="backward")
    ml = cat2cat_ml(
        occup.loc[occup.year >= 2010, :].copy(),
        "code",
        ["salary", "age", "edu", "sex"],
        [DecisionTreeClassifier(random_state=1234)],
    )

    with pytest.raises(ValueError, match="does not cover"):
        cat2cat_ml_run(mappings=mappings, ml=ml, min_match=0.8)


def test_cat2cat_supports_multiple_models_in_one_run():
    data = cat2cat_data(o_old, o_new, "code", "code", "year")
    mappings = cat2cat_mappings(trans, "backward")
    ml = cat2cat_ml(
        occup.loc[occup.year >= 2010, :].copy(),
        "code",
        ["salary", "age", "edu", "sex", "parttime"],
        [
            DecisionTreeClassifier(random_state=1234),
            LinearDiscriminantAnalysis(),
            GaussianNB(),
        ],
        on_fail="freq",
        fail_warn=False,
    )

    diag = cat2cat_ml_run(mappings=mappings, ml=ml)
    c2c = cat2cat(data, mappings, ml)

    expected_models = [
        "DecisionTreeClassifier",
        "LinearDiscriminantAnalysis",
        "GaussianNB",
    ]

    for model_name in expected_models:
        assert model_name in diag.mean_acc
        assert model_name in diag.mean_brier
        assert model_name in diag.mean_prob
        col = f"wei_{model_name}_c2c"
        assert col in c2c["old"].columns
        assert c2c["old"].groupby("index_c2c")[col].sum().round(10).eq(1).all()
