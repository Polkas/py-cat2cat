import warnings
from typing import Any, List, Optional, Sequence, Tuple

import numpy as np
from pandas import DataFrame
from pandas.api.types import is_bool_dtype, is_numeric_dtype
from sklearn.base import ClassifierMixin, clone

from cat2cat.dataclass import cat2cat_ml


def safe_nanmean(values: Sequence[Any], ndigits: int = 3) -> float:
    vals = np.asarray(values, dtype=float)
    if vals.size == 0 or np.all(np.isnan(vals)):
        return np.nan
    return float(np.round(np.nanmean(vals), ndigits))


def resolve_ml_models(ml: cat2cat_ml) -> List[Tuple[str, ClassifierMixin]]:
    return [(type(model).__name__, clone(model)) for model in ml.models]


def prepare_ml_frames(
    ml: cat2cat_ml, target_data: Optional[DataFrame] = None
) -> Tuple[DataFrame, Optional[DataFrame], List[str]]:
    train_data = ml.data.copy()
    target_work = target_data
    features: List[str] = []

    for feature in ml.features:
        train_col = train_data[feature]
        target_col = target_work[feature] if target_work is not None else None

        target_direct = target_col is None or _feature_is_direct(target_col)
        if _feature_is_direct(train_col) and target_direct:
            features.append(feature)
            continue

        train_values = train_col.dropna().astype(str)
        target_values = (
            target_col.dropna().astype(str)
            if target_col is not None
            else train_values.iloc[0:0]
        )
        levels = list(dict.fromkeys(list(train_values) + list(target_values)))

        for level in levels:
            col_name = f"{feature}_{level}"
            train_data[col_name] = (
                train_col.notna() & (train_col.astype(str) == level)
            ).astype(int)
            if target_work is not None and target_col is not None:
                target_work[col_name] = (
                    target_col.notna() & (target_col.astype(str) == level)
                ).astype(int)
            features.append(col_name)

    return train_data, target_work, features


def brier_score(
    prob_matrix: DataFrame, true_cats: Sequence[Any], classes: Sequence[Any]
) -> float:
    classes_list = list(classes)
    probs = prob_matrix.copy()
    for cls in classes_list:
        if cls not in probs.columns:
            probs[cls] = 0.0
    probs = probs.loc[:, classes_list].astype(float).to_numpy()

    truth = np.zeros((len(true_cats), len(classes_list)))
    class_index = {cls: idx for idx, cls in enumerate(classes_list)}
    for row_index, true_cat in enumerate(true_cats):
        if true_cat in class_index:
            truth[row_index, class_index[true_cat]] = 1.0

    return float(np.mean(np.sum((probs - truth) ** 2, axis=1) / 2))


def mean_true_probability(
    prob_matrix: DataFrame, true_cats: Sequence[Any]
) -> float:
    vals = []
    for row_index, true_cat in enumerate(true_cats):
        vals.append(float(prob_matrix.iloc[row_index].get(true_cat, 0.0)))
    return float(np.mean(vals)) if vals else np.nan


def apply_ml_fallback(
    target_df: DataFrame, ml_names: Sequence[str], on_fail: str, fail_warn: bool
) -> None:
    if on_fail == "freq":
        fallback = target_df["wei_freq_c2c"]
    elif on_fail == "naive":
        fallback = target_df["wei_naive_c2c"]
    else:
        fallback = DataFrame({"fallback": np.nan}, index=target_df.index)["fallback"]

    total_rows = target_df.shape[0]
    total_obs = target_df["index_c2c"].nunique()

    for ml_name in ml_names:
        col = target_df[ml_name].astype(float)
        failed = col.isna() | ~np.isfinite(col)

        if failed.any():
            n_rows = int(failed.sum())
            n_obs = int(target_df.loc[failed, "index_c2c"].nunique())
            pct_rows = 100 * n_rows / total_rows
            pct_obs = 100 * n_obs / total_obs
            method_name = _weight_column_method_name(ml_name)

            if on_fail == "error":
                raise RuntimeError(
                    "ML weights failed for method '{}': {:.1f}% rows ({}/{}) "
                    "and {:.1f}% observations ({}/{}).".format(
                        method_name,
                        pct_rows,
                        n_rows,
                        total_rows,
                        pct_obs,
                        n_obs,
                        total_obs,
                    )
                )

            if fail_warn:
                warnings.warn(
                    "ML weights failed for method '{}': {:.1f}% rows ({}/{}) "
                    "and {:.1f}% observations ({}/{}); on_fail = '{}' was applied.".format(
                        method_name,
                        pct_rows,
                        n_rows,
                        total_rows,
                        pct_obs,
                        n_obs,
                        total_obs,
                        on_fail,
                    ),
                    stacklevel=2,
                )

            col.loc[failed] = fallback.loc[failed]

        scale_factor = col.groupby(target_df["index_c2c"]).transform("sum")
        can_scale = np.isfinite(scale_factor) & (scale_factor > 0)
        col.loc[can_scale] = col.loc[can_scale] / scale_factor.loc[can_scale]
        target_df[ml_name] = col


def _feature_is_direct(feature: Any) -> bool:
    return is_numeric_dtype(feature) or is_bool_dtype(feature)


def _weight_column_method_name(ml_name: str) -> str:
    method_name = ml_name
    if method_name.startswith("wei_"):
        method_name = method_name[4:]
    if method_name.endswith("_c2c"):
        method_name = method_name[:-4]
    return method_name