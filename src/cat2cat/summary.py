from math import erfc, isfinite, sqrt
from typing import Any, Optional

import pandas as pd

__all__ = ["summary_c2c"]


def summary_c2c(
    model: Any, df_old: float, df_new: Optional[float] = None
) -> pd.DataFrame:
    """Adjust regression summaries fitted on replicated cat2cat data.

    Args:
        model: A fitted statsmodels-like result object with ``params``, ``bse``,
            and ``tvalues`` attributes.
        df_old: Residual degrees of freedom on the original observation scale.
        df_new: Residual degrees of freedom on the replicated data scale.
            Defaults to ``model.df_resid``.

    Returns:
        pandas.DataFrame: coefficient table with corrected standard errors,
        corrected statistics, corrected p-values, and reference distribution.

    Examples:
        >>> from pandas import DataFrame, concat
        >>> import statsmodels.api as sm
        >>> from cat2cat import summary_c2c
        >>> data = DataFrame({
        ...     "y": [2.0, 3.0, 5.0, 7.0, 11.0, 13.0, 17.0, 19.0],
        ...     "x1": [1.0, 1.5, 2.0, 2.7, 3.2, 4.1, 4.8, 5.2],
        ...     "x2": [0, 1, 0, 1, 0, 1, 0, 1],
        ... })
        >>> model = sm.OLS.from_formula("y ~ x1 + x2", data=data).fit()
        >>> model_rep = sm.OLS.from_formula(
        ...     "y ~ x1 + x2", data=concat([data, data])
        ... ).fit()
        >>> out = summary_c2c(model_rep, df_old=model.df_resid, df_new=model_rep.df_resid)
        >>> all(col in out.columns for col in ["std.error_c", "statistic_c", "p.value_c"])
        True
    """
    _validate_df(df_old, "df_old")
    if df_new is None:
        if not hasattr(model, "df_resid"):
            raise ValueError(
                "df_new must be provided when model has no df_resid attribute"
            )
        df_new = model.df_resid
    _validate_df(df_new, "df_new")

    params = _as_series(model, "params")
    std_error = _as_series(model, "bse")
    statistic = _as_series(model, "tvalues")

    if not (len(params) == len(std_error) == len(statistic)):
        raise ValueError("model coefficient arrays must have the same length")

    correct = sqrt(float(df_new) / float(df_old))
    reference_dist = "t" if bool(getattr(model, "use_t", True)) else "normal"
    statistic_c = statistic / correct

    res = pd.DataFrame(
        {
            "Estimate": params,
            "Std. Error": std_error,
            "statistic": statistic,
            "correct": correct,
            "std.error_c": std_error * correct,
            "statistic_c": statistic_c,
            "p.value_c": [
                _p_value(abs(value), float(df_old), reference_dist)
                for value in statistic_c
            ],
            "reference_dist": reference_dist,
        }
    )
    return res


def _validate_df(value: float, name: str) -> None:
    if not isinstance(value, (int, float)) or not isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be a single positive finite numeric value")


def _as_series(model: Any, attr: str) -> pd.Series:
    if not hasattr(model, attr):
        raise ValueError(f"model must have a {attr} attribute")
    value = getattr(model, attr)
    if isinstance(value, pd.Series):
        return value
    return pd.Series(value)


def _p_value(statistic: float, df_old: float, reference_dist: str) -> float:
    if reference_dist == "normal":
        return erfc(statistic / sqrt(2))

    try:
        from scipy import stats
    except ImportError as exc:
        raise ImportError("scipy is required for t-distribution p-values") from exc

    return float(2 * stats.t.sf(statistic, df_old))