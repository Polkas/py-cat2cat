import pytest
from pandas import DataFrame
from pandas import concat

from cat2cat import summary_c2c

sm = pytest.importorskip("statsmodels.api")


def test_summary_c2c_ols_adjusts_standard_errors():
    data = DataFrame(
        {
            "y": [2.0, 3.0, 5.0, 7.0, 11.0, 13.0, 17.0, 19.0],
            "x1": [1.0, 1.5, 2.0, 2.7, 3.2, 4.1, 4.8, 5.2],
            "x2": [0, 1, 0, 1, 0, 1, 0, 1],
        }
    )
    model = sm.OLS.from_formula("y ~ x1 + x2", data=data).fit()
    model_rep = sm.OLS.from_formula(
        "y ~ x1 + x2", data=concat([data, data])
    ).fit()

    res = summary_c2c(model_rep, df_old=model.df_resid, df_new=model_rep.df_resid)

    assert all(col in res.columns for col in ["std.error_c", "statistic_c", "p.value_c"])
    assert all(res["reference_dist"] == "t")
    assert res.loc["Intercept", "std.error_c"] > model_rep.bse["Intercept"]


def test_summary_c2c_glm_uses_normal_reference():
    data = DataFrame(
        {
            "y": [0, 0, 0, 1, 0, 1, 1, 1, 0, 1],
            "x1": [1.0, 1.2, 1.4, 2.0, 2.2, 2.8, 3.0, 3.3, 3.5, 4.0],
            "x2": [0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
        }
    )
    model = sm.GLM.from_formula("y ~ x1 + x2", data=data, family=sm.families.Binomial()).fit()
    model_rep = sm.GLM.from_formula(
        "y ~ x1 + x2", data=concat([data, data]), family=sm.families.Binomial()
    ).fit()

    res = summary_c2c(model_rep, df_old=model.df_resid, df_new=model_rep.df_resid)

    assert all(res["reference_dist"] == "normal")
    assert all(res["p.value_c"].between(0, 1))


def test_summary_c2c_rejects_invalid_df():
    data = DataFrame({"y": [1, 2, 3, 4, 5], "x": [1, 2, 4, 8, 16]})
    model = sm.OLS.from_formula("y ~ x", data=data).fit()

    with pytest.raises(ValueError, match="df_old"):
        summary_c2c(model, df_old=0)

    with pytest.raises(ValueError, match="df_new"):
        summary_c2c(model, df_old=model.df_resid, df_new=-1)


def test_summary_c2c_example_like_usage_runs():
    data = DataFrame(
        {
            "y": [2.0, 3.0, 5.0, 7.0, 11.0, 13.0, 17.0, 19.0],
            "x1": [1.0, 1.5, 2.0, 2.7, 3.2, 4.1, 4.8, 5.2],
            "x2": [0, 1, 0, 1, 0, 1, 0, 1],
        }
    )
    model = sm.OLS.from_formula("y ~ x1 + x2", data=data).fit()
    model_rep = sm.OLS.from_formula(
        "y ~ x1 + x2", data=concat([data, data])
    ).fit()

    out = summary_c2c(model_rep, df_old=model.df_resid, df_new=model_rep.df_resid)

    assert all(
        col in out.columns for col in ["std.error_c", "statistic_c", "p.value_c"]
    )