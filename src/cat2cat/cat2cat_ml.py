from typing import Any, Dict

import numpy as np
from pandas import DataFrame, concat
from sklearn.model_selection import train_test_split

from cat2cat.dataclass import cat2cat_mappings, cat2cat_ml
from cat2cat.cat2cat_ml_utils import (
    apply_ml_fallback,
    brier_score,
    mean_true_probability,
    prepare_ml_frames,
    resolve_ml_models,
    safe_nanmean,
)
from cat2cat.mappings import get_mappings

__all__ = ["cat2cat_ml_run"]


class cat2cat_ml_run_results:
    """Container for diagnostics returned by ``cat2cat_ml_run``.

    The object stores per-group raw diagnostics and aggregated statistics across
    mapping groups for baseline and model-based solutions.

    Attributes:
        res: Raw nested diagnostics dictionary by mapping target category.
        mappings: Mapping configuration used in diagnostics.
        ml: ML configuration used in diagnostics.
        kwargs: Runtime options passed to ``cat2cat_ml_run``.
        models_names: Names of evaluated model classes.
        mean_acc: Mean accuracy by baseline/model name.
        mean_brier: Mean Brier score by baseline/model name.
        mean_prob: Mean probability assigned to the true class by
            baseline/model name.
        percent_failed: Percent of mapping groups where a given model failed.
        percent_better_most: Share of valid groups where model accuracy exceeds
            the most-frequent baseline.
        percent_better_naive: Share of valid groups where model accuracy exceeds
            the naive baseline.

    Notes:
        Use ``get_raw()`` to access per-group metrics and ``repr(result)``
        to print an aggregated summary.
    """

    def __init__(
        self, res: Dict, mappings: cat2cat_mappings, ml: cat2cat_ml, kwargs: Dict
    ) -> None:
        self.res = res
        self.mappings = mappings
        self.ml = ml
        self.kwargs = kwargs
        self.models_names = [type(model).__name__ for model in self.ml.models]

        self.mean_acc = {
            "naive": safe_nanmean(
                [self.res.get(g, {}).get("naive", np.nan) for g in self.res.keys()]
            ),
            "most_freq": safe_nanmean(
                [self.res.get(g, {}).get("freq", np.nan) for g in self.res.keys()], 2
            ),
        }
        self.mean_brier = {
            "naive": safe_nanmean(
                [
                    self.res.get(g, {}).get("naive_brier", np.nan)
                    for g in self.res.keys()
                ]
            ),
            "most_freq": safe_nanmean(
                [
                    self.res.get(g, {}).get("freq_brier", np.nan)
                    for g in self.res.keys()
                ]
            ),
        }
        self.mean_prob = {
            "naive": safe_nanmean(
                [
                    self.res.get(g, {}).get("naive_mean_prob", np.nan)
                    for g in self.res.keys()
                ]
            ),
            "most_freq": safe_nanmean(
                [
                    self.res.get(g, {}).get("freq_mean_prob", np.nan)
                    for g in self.res.keys()
                ]
            ),
        }

        self.percent_failed = {}
        self.percent_better_most = {}
        self.percent_better_naive = {}
        for model_name in self.models_names:
            vals = [self.res.get(g, {}).get(model_name, np.nan) for g in self.res.keys()]
            vals_arr = np.asarray(vals, dtype=float)
            self.mean_acc[model_name] = safe_nanmean(vals)
            self.mean_brier[model_name] = safe_nanmean(
                [
                    self.res.get(g, {}).get(f"{model_name}_brier", np.nan)
                    for g in self.res.keys()
                ]
            )
            self.mean_prob[model_name] = safe_nanmean(
                [
                    self.res.get(g, {}).get(f"{model_name}_mean_prob", np.nan)
                    for g in self.res.keys()
                ]
            )
            self.percent_failed[model_name] = float(
                np.round(np.mean(np.isnan(vals_arr)) * 100, 3)
            )

            better_most = []
            better_naive = []
            for group in self.res.values():
                value = group.get(model_name, np.nan)
                if not np.isnan(value):
                    better_most.append(value > group.get("freq", np.nan))
                    better_naive.append(value > group.get("naive", np.nan))
            self.percent_better_most[model_name] = safe_nanmean(better_most)
            self.percent_better_naive[model_name] = safe_nanmean(better_naive)

    def get_raw(self) -> Dict:
        """Get raw results."""
        return self.res

    def __repr__(self) -> str:
        res = ""
        for key, value in self.mean_acc.items():
            res += f"Average Accuracy {key}: {value}\n"
        res += "\n"
        for key, value in self.mean_brier.items():
            res += f"Average Brier {key}: {value}\n"
        res += "\n"
        for key, value in self.mean_prob.items():
            res += f"Average Mean P(true) {key}: {value}\n"
        res += "\n"
        for key, value in self.percent_failed.items():
            res += f"Percent of failed {key}: {value}\n"
        res += "\n"
        for key, value in self.percent_better_most.items():
            res += (
                f"Percent of better {key} over most frequent category solution: "
                f"{value}\n"
            )
        for key, value in self.percent_better_naive.items():
            res += f"Percent of better {key} over naive solution: {value}\n"
        res += "\n"
        res += f"Features: {self.ml.features}\n"
        res += f"Test sample size: {self.kwargs.get('test_prop', 0.2) * 100}\n"
        return res


def cat2cat_ml_run(
    mappings: cat2cat_mappings, ml: cat2cat_ml, **kwargs: Any
) -> cat2cat_ml_run_results:
    """Run model diagnostics before using ML-based cat2cat weights.

    This helper evaluates baseline and model-based classification quality within
    each mapping group and aggregates summary statistics across groups.

    Args:
        mappings: Mapping configuration created with ``cat2cat_mappings``.
        ml: ML configuration created with ``cat2cat_ml``.
        **kwargs: Optional diagnostics settings:
            - ``test_prop`` (float): test split proportion in ``(0, 1)``.
              Default is ``0.2``.
            - ``split_seed`` (int): random seed for train/test split.
              Default is ``42``.
            - ``min_match`` (float): minimum fraction of records in ``ml.data``
              whose category appears in the mapping table. Must be in
              ``[0, 1)``. Default is ``0.8``.

    Returns:
        cat2cat_ml_run_results: object with per-group raw diagnostics and
        aggregated metrics such as mean accuracy, mean Brier score,
        mean P(true class), failure rates, and model-vs-baseline comparisons.

    Raises:
        TypeError: if ``mappings`` or ``ml`` has invalid type.
        ValueError: if kwargs names/ranges are invalid or mapping coverage is
            below ``min_match``.

    Examples:
        >>> from sklearn.ensemble import RandomForestClassifier
        >>> from cat2cat import cat2cat_ml_run
        >>> from cat2cat.dataclass import cat2cat_mappings, cat2cat_ml
        >>> from cat2cat.datasets import load_trans, load_occup
        >>> trans = load_trans()
        >>> occup = load_occup()
        >>> data_2010 = occup.loc[occup.year == 2010, :].copy()
        >>> mappings = cat2cat_mappings(trans, "backward")
        >>> ml = cat2cat_ml(
        ...     data=data_2010,
        ...     cat_var="code",
        ...     features=["salary", "age", "edu", "sex"],
        ...     models=[RandomForestClassifier(n_estimators=50, random_state=1234)],
        ... )
        >>> out = cat2cat_ml_run(mappings=mappings, ml=ml, test_prop=0.2)
        >>> hasattr(out, "mean_acc")
        True
    """
    if not isinstance(mappings, cat2cat_mappings):
        raise TypeError("mappings arg has to be cat2cat_mappings instance")
    if not isinstance(ml, cat2cat_ml):
        raise TypeError("ml arg has to be cat2cat_ml instance")
    if not isinstance(kwargs, dict):
        raise TypeError("kwargs arg has to be a dict")
    if not set(kwargs.keys()).issubset(["min_match", "test_prop", "split_seed"]):
        raise ValueError("possible kwargs are min_match, split_seed and test_prop")
    if not 0 < kwargs.get("test_prop", 0.2) < 1:
        raise ValueError("test_prop has to be between 0 and 1")
    if not 0 <= kwargs.get("min_match", 0.8) < 1:
        raise ValueError("min_match has to be between 0 and 1")

    train_data, _, features = prepare_ml_frames(ml)
    models = resolve_ml_models(ml)
    mapps = get_mappings(mappings.trans)

    if mappings.direction == "forward":
        base_name = "old"
    elif mappings.direction == "backward":
        base_name = "new"

    mapp = mapps["to_" + base_name]

    cat_var = train_data[ml.cat_var].values
    cat_var_vals = mappings.trans[base_name].unique()

    if not (np.sum(np.isin(cat_var, cat_var_vals)) / len(cat_var)) > kwargs.get(
        "min_match", 0.8
    ):
        raise ValueError(
            "The mapping table does not cover all categories in the data. Please check the direction in the mapping table."
        )

    train_g = {
        n: g for n, g in train_data[features + [ml.cat_var]].groupby(ml.cat_var)
    }

    res = dict()
    for cat in mapp.keys():
        try:
            matched_cat = mapp.get(cat, [])
            res[cat] = {
                "naive": np.nan,
                "freq": np.nan,
                "naive_brier": np.nan,
                "naive_mean_prob": np.nan,
                "freq_brier": np.nan,
                "freq_mean_prob": np.nan,
            }
            for model_name, _ in models:
                res[cat][model_name] = np.nan
                res[cat][f"{model_name}_brier"] = np.nan
                res[cat][f"{model_name}_mean_prob"] = np.nan

            data_small_g_list = [train_g.get(g) for g in matched_cat if g in train_g]
            if len(data_small_g_list) == 0:
                continue

            data_small_g = concat(data_small_g_list, axis=0)
            if (
                (data_small_g.shape[0] < 10)
                or (len(matched_cat) < 2)
                or (np.sum(np.isin(matched_cat, data_small_g[ml.cat_var])) == 1)
            ):
                continue

            n_categories = len(matched_cat)
            res[cat]["naive"] = 1 / n_categories
            res[cat]["naive_mean_prob"] = 1 / n_categories
            res[cat]["naive_brier"] = (1 - 1 / n_categories) / 2

            X_train, X_test, y_train, y_test = train_test_split(
                data_small_g[features],
                data_small_g[ml.cat_var],
                test_size=kwargs.get("test_prop", 0.2),
                random_state=kwargs.get("split_seed", 42),
            )

            gcounts = y_train.value_counts()
            gfreq_max = gcounts.index[0]
            res[cat]["freq"] = float(np.nanmean(gfreq_max == y_test))

            train_freqs = y_train.value_counts(normalize=True)
            freq_probs = DataFrame(0.0, index=range(len(y_test)), columns=matched_cat)
            for freq_cat, freq_value in train_freqs.items():
                if freq_cat in freq_probs.columns:
                    freq_probs[freq_cat] = float(freq_value)
            res[cat]["freq_brier"] = brier_score(freq_probs, y_test.values, matched_cat)
            res[cat]["freq_mean_prob"] = mean_true_probability(freq_probs, y_test.values)

            if (X_test.shape[0] == 0) or (X_train.shape[0] < 5):
                continue

            for model_name, model in models:
                try:
                    model.fit(X_train, y_train)  # type: ignore
                    preds = model.predict(X_test)  # type: ignore
                    probs = DataFrame(model.predict_proba(X_test))  # type: ignore
                    probs.columns = model.classes_  # type: ignore
                    res[cat][model_name] = float(np.nanmean(preds == y_test))
                    res[cat][f"{model_name}_brier"] = brier_score(
                        probs, y_test.values, matched_cat
                    )
                    res[cat][f"{model_name}_mean_prob"] = mean_true_probability(
                        probs, y_test.values
                    )
                except Exception:
                    continue
        except Exception:
            continue

    return cat2cat_ml_run_results(res, mappings, ml, kwargs)


def _cat2cat_ml(
    ml: cat2cat_ml, mapp: Dict[Any, Any], target_df: DataFrame, cat_var_target: str
) -> None:
    """cat2cat ml optional part."""
    train_data, _, features = prepare_ml_frames(ml, target_df)
    models = resolve_ml_models(ml)
    ml_names = ["wei_" + model_name + "_c2c" for model_name, _ in models]
    target_df[ml_names] = np.nan

    for target_cat in list(mapp.keys()):
        base_cats = mapp[target_cat]
        ml_cat_var = train_data[ml.cat_var]

        if not any(np.isin(base_cats, ml_cat_var.unique())):
            continue

        target_cat_index = np.isin(target_df[cat_var_target].values, target_cat)
        ml_cat_index = np.isin(train_data[ml.cat_var].values, base_cats)

        data_ml_train = train_data.loc[ml_cat_index, :]
        data_ml_target = target_df.loc[target_cat_index, :]

        if (data_ml_target.shape[0] == 0) or (data_ml_train.shape[0] < 5):
            continue

        target_cats = data_ml_target["g_new_c2c"]
        data_ml_target_uniq = data_ml_target.drop_duplicates(
            subset=["index_c2c"] + features
        )
        index_c2c = data_ml_target_uniq["index_c2c"].values

        train_complete = data_ml_train[features].notna().all(axis=1)
        target_complete = data_ml_target_uniq[features].notna().all(axis=1)
        if train_complete.sum() < 5 or target_complete.sum() == 0:
            continue

        X_train = data_ml_train.loc[train_complete, features]
        y_train = data_ml_train.loc[train_complete, ml.cat_var]
        X_test = data_ml_target_uniq.loc[target_complete, features]
        test_index_c2c = index_c2c[target_complete.values]

        for model_name, model in models:
            ml_colname = "wei_" + model_name + "_c2c"

            try:
                model.fit(X=X_train, y=y_train)  # type: ignore
                preds = model.predict_proba(X=X_test)  # type: ignore

                preds_df = DataFrame(preds)
                preds_df.columns = model.classes_  # type: ignore
                preds_df[np.setdiff1d(target_cats.unique(), model.classes_)] = 0  # type: ignore
                preds_df["index_c2c"] = test_index_c2c
                preds_df_melt = preds_df.melt(id_vars="index_c2c", var_name="g_new_c2c")
                merge_on = ["index_c2c", "g_new_c2c"]
                p_order = target_df.loc[target_cat_index, merge_on].merge(
                    preds_df_melt, on=merge_on, how="left", sort=False
                )
                target_df.loc[target_cat_index, ml_colname] = p_order["value"].values
            except Exception:
                continue

    apply_ml_fallback(target_df, ml_names, ml.on_fail.lower(), ml.fail_warn)