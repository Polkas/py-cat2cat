import numpy as np
import pytest
from pandas import DataFrame
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier

from cat2cat.cat2cat_ml_utils import (
    apply_ml_fallback,
    brier_score,
    mean_true_probability,
    prepare_ml_frames,
    resolve_ml_models,
    safe_nanmean,
)
from cat2cat.dataclass import cat2cat_ml


def test_safe_nanmean_handles_empty_and_all_nan():
    assert np.isnan(safe_nanmean([]))
    assert np.isnan(safe_nanmean([np.nan, np.nan]))
    assert safe_nanmean([1.0, 2.0, np.nan], ndigits=2) == 1.5


def test_brier_and_mean_true_probability_simple_case():
    probs = DataFrame(
        {
            "A": [0.8, 0.1],
            "B": [0.2, 0.9],
        }
    )
    y_true = ["A", "B"]

    # For each row: ((0.8-1)^2 + (0.2-0)^2)/2 = 0.04 and similarly for row 2.
    assert brier_score(probs, y_true, ["A", "B"]) == pytest.approx(0.04)
    assert mean_true_probability(probs, y_true) == pytest.approx(0.85)


def test_prepare_ml_frames_expands_categorical_levels_from_train_and_target():
    train = DataFrame(
        {
            "code": ["x", "y", "x"],
            "num": [1.0, 2.0, 3.0],
            "cat": ["a", "b", "a"],
        }
    )
    target = DataFrame(
        {
            "num": [10.0, 11.0],
            "cat": ["b", "c"],
        }
    )

    ml = cat2cat_ml(
        data=train,
        cat_var="code",
        features=["num", "cat"],
        models=[GaussianNB()],
    )

    train_out, target_out, features = prepare_ml_frames(ml, target.copy())

    assert "num" in features
    assert "cat_a" in features
    assert "cat_b" in features
    assert "cat_c" in features

    assert "cat_c" in train_out.columns
    assert int(train_out["cat_c"].sum()) == 0

    assert target_out is not None
    assert "cat_c" in target_out.columns
    assert int(target_out["cat_c"].sum()) == 1


def test_resolve_ml_models_clones_estimators():
    ml = cat2cat_ml(
        data=DataFrame({"code": [0, 1, 0], "x": [1.0, 2.0, 3.0]}),
        cat_var="code",
        features=["x"],
        models=[DecisionTreeClassifier(random_state=1234), GaussianNB()],
    )

    resolved = resolve_ml_models(ml)

    assert [name for name, _ in resolved] == [
        "DecisionTreeClassifier",
        "GaussianNB",
    ]
    assert resolved[0][1] is not ml.models[0]
    assert resolved[1][1] is not ml.models[1]


def test_apply_ml_fallback_freq_fills_and_normalizes():
    target = DataFrame(
        {
            "index_c2c": [1, 1, 2, 2],
            "wei_freq_c2c": [0.7, 0.3, 0.2, 0.8],
            "wei_naive_c2c": [0.5, 0.5, 0.5, 0.5],
            "wei_TestModel_c2c": [np.nan, np.nan, 0.25, 0.75],
        }
    )

    apply_ml_fallback(
        target_df=target,
        ml_names=["wei_TestModel_c2c"],
        on_fail="freq",
        fail_warn=False,
    )

    assert target["wei_TestModel_c2c"].notna().all()
    assert target.loc[target["index_c2c"] == 1, "wei_TestModel_c2c"].tolist() == [0.7, 0.3]
    assert target.groupby("index_c2c")["wei_TestModel_c2c"].sum().round(10).eq(1).all()


def test_apply_ml_fallback_error_raises_with_method_name():
    target = DataFrame(
        {
            "index_c2c": [1, 1],
            "wei_freq_c2c": [0.5, 0.5],
            "wei_naive_c2c": [0.5, 0.5],
            "wei_GaussianNB_c2c": [np.nan, 0.5],
        }
    )

    with pytest.raises(RuntimeError, match="GaussianNB"):
        apply_ml_fallback(
            target_df=target,
            ml_names=["wei_GaussianNB_c2c"],
            on_fail="error",
            fail_warn=False,
        )
