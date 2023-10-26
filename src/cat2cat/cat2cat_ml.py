from pandas import DataFrame, concat
from numpy import repeat, setdiff1d, in1d, sum, NaN, nanmean, isnan, round

from sklearn.model_selection import train_test_split

from cat2cat.mappings import get_mappings
from cat2cat.dataclass import cat2cat_mappings, cat2cat_ml

from typing import Any, Dict

__all__ = ["cat2cat_ml_run"]


class cat2cat_ml_run_results:
    """The class to represent the results of the cat2cat_ml_run function call
    Args:
        res (Dict): raw results from the cat2cat_ml_run function call
        mappings (cat2cat_mappings): dataclass with mappings related arguments.
            Please check out the `cat2cat.dataclass.cat2cat_mappings` for more information.
        ml (cat2cat_ml): dataclass with ml related arguments.
            Please check out the `cat2cat.dataclass.cat2cat_ml` for more information.
        kwargs (Dict): additional arguments passed to the `cat2cat_ml_run` function.
    Returns:
        cat2cat_ml_run_results class instance with the following attributes:
        res (Dict): raw results from the cat2cat_ml_run function call
        mean_acc (Dict): mean accuracy for each model
        percent_failed (Dict): percent of failed models for each model
        percent_better (Dict): percent of better models over most frequent category solution for each model
        mappings (cat2cat_mappings): initial mappings dataclass with mappings related arguments.
        ml (cat2cat_ml): initial ml dataclass with ml related arguments.
    Methods:
        get_raw: get raw results
    """

    def __init__(
        self, res: Dict, mappings: cat2cat_mappings, ml: cat2cat_ml, kwargs: Dict
    ) -> None:
        self.res = res
        self.mappings = mappings
        self.ml = ml
        self.kwargs = kwargs
        self.models_names = [type(m).__name__ for m in self.ml.models]

        mean_acc = dict()
        percent_failed = dict()
        percent_better = dict()

        mean_acc["naive"] = round(
            nanmean(
                [self.res.get(g, {"naive": NaN}).get("naive") for g in self.res.keys()]
            ),
            3,
        )
        mean_acc["most_freq"] = round(
            nanmean(
                [self.res.get(g, {"freq": NaN}).get("freq") for g in self.res.keys()]
            ),
            2,
        )
        for m in self.models_names:
            vals = [
                self.res.get(g, {"acc": {m: NaN}}).get("acc").get(m, NaN)
                for g in self.res.keys()
            ]
            mean_acc[m] = round(nanmean(vals), 3)
            percent_failed[m] = round(sum(isnan(vals)) / len(vals) * 100, 3)
            percent_better[m] = round(
                sum(vals > mean_acc["most_freq"]) / len(vals) * 100, 3
            )

        self.mean_acc = mean_acc
        self.percent_failed = percent_failed
        self.percent_better = percent_better

    def get_raw(self) -> Dict:
        """Get raw results"""
        return self.res

    def __repr__(self) -> str:
        res = ""
        for k, v in self.mean_acc.items():
            res += "Average Accuracy {}: {}".format(k, v) + "\n"
        res += "\n"
        for k, v in self.percent_failed.items():
            res += "Percent of failed {}: {}".format(k, v) + "\n"
        res += "\n"
        for k, v in self.percent_better.items():
            res += (
                "Percent of better {} over most frequent category solution: {}".format(
                    k, v
                )
                + "\n"
            )
        res += "\n"
        res += "Features: {}".format(self.ml.features) + "\n"
        res += "Test sample size: {}".format(self.kwargs.get("test_size", 0.2)) + "\n"
        return res


def cat2cat_ml_run(
    mappings: cat2cat_mappings, ml: cat2cat_ml, **kwargs: Any
) -> cat2cat_ml_run_results:
    """Automatic mapping in a panel dataset - cat2cat procedure

    Args:
        mappings (cat2cat_mappings): dataclass with mappings related arguments.
            Please check out the `cat2cat.dataclass.cat2cat_mappings` for more information.
        ml (Optional[cat2cat_ml]): dataclass with ml related arguments.
            Please check out the `cat2cat.dataclass.cat2cat_ml` for more information.
        **kwargs: additional arguments passed to the `cat2cat_ml_run` function.
            min_match (float): minimum share of categories from the base period that have to be matched in the mapping table. Between 0 and 1. Default 0.8.
            test_size (float): share of the data used for testing. Between 0 and 1. Default 0.2.

    Returns:
        cat2cat_ml_run_class

    Note:
        Please check out the `cat2cat.cat2cat.cat2cat` for more information.


    >>> from cat2cat import cat2cat
    >>> from cat2cat.cat2cat_ml import cat2cat_ml_run
    >>> from cat2cat.dataclass import cat2cat_data, cat2cat_mappings, cat2cat_ml
    >>> from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    >>> from sklearn.tree import DecisionTreeClassifier
    >>> from cat2cat.datasets import load_trans, load_occup
    >>> trans = load_trans()
    >>> occup = load_occup()
    >>> o_old = occup.loc[occup.year == 2008, :].copy()
    >>> o_new = occup.loc[occup.year == 2010, :].copy()
    >>> mappings = cat2cat_mappings(trans = trans, direction = "forward")
    >>> ml = cat2cat_ml(
    ...    occup.loc[occup.year <= 2008, :].copy(),
    ...    "code",
    ...    ["salary", "age", "edu", "sex"],
    ...    [DecisionTreeClassifier(), LinearDiscriminantAnalysis()]
    ... )
    >>> cat2cat_ml_run(mappings = mappings, ml = ml)

    """
    assert isinstance(
        mappings, cat2cat_mappings
    ), "mappings arg has to be cat2cat_mappings instance"
    assert isinstance(ml, cat2cat_ml), "ml arg has to be cat2cat_ml instance"

    mapps = get_mappings(mappings.trans)

    if mappings.direction == "forward":
        target_name = "new"
        base_name = "old"
    elif mappings.direction == "backward":
        target_name = "old"
        base_name = "new"

    mapp = mapps["to_" + base_name]

    cat_var = ml.data[ml.cat_var].values
    cat_var_vals = mappings.trans[base_name].unique()

    assert (sum(in1d(cat_var, cat_var_vals)) / len(cat_var)) > kwargs.get(
        "min_match", 0.8
    ), "The mapping table does not cover all categories in the data. Please check the direction in the mapping table."

    features = ml.features
    models = ml.models
    models_names = [type(m).__name__ for m in models]

    train_g = {
        n: g for n, g in ml.data[list(features) + [ml.cat_var]].groupby(ml.cat_var)
    }

    res = dict()
    for cat in mapp.keys():
        try:
            matched_cat = mapp.get(cat, [])
            acc_na = dict(zip(models_names, repeat(NaN, len(models_names))))
            res[cat] = {
                "ncat": len(matched_cat),
                "naive": 1 / len(matched_cat),
                "acc": acc_na,
                "freq": NaN,
            }
            data_small_g_list = list()
            for g in matched_cat:
                if g not in train_g.keys():
                    continue
                data_small_g_list.append(train_g.get(g))
            if len(data_small_g_list) == 0:
                continue

            data_small_g = concat([train_g.get(g) for g in matched_cat], axis=0)

            if (
                (data_small_g.shape[0] < 5)
                or (len(matched_cat) < 2)
                or (sum(in1d(matched_cat, data_small_g[ml.cat_var])) == 1)
            ):
                continue

            X_train, X_test, y_train, y_test = train_test_split(
                data_small_g[features],
                data_small_g[ml.cat_var],
                test_size=kwargs.get("test_size", 0.2),
                random_state=42,
            )

            gcounts = y_train.value_counts()
            gfreq_max = gcounts.index[0]
            res[cat]["freq"] = nanmean(gfreq_max == y_test)

            if X_test.shape[0] == 0 | X_train.shape[0] < 5:
                continue

            for m in models:
                ml_name = str(type(m).__name__)
                m.fit(X_train, y_train)
                res[cat]["acc"][ml_name] = m.score(X_test, y_test)  # type: ignore
        except:
            continue

    return cat2cat_ml_run_results(res, mappings, ml, kwargs)


def _cat2cat_ml(
    ml: cat2cat_ml, mapp: Dict[Any, Any], target_df: DataFrame, cat_var_target: str
) -> None:
    """cat2cat ml optional part"""
    for target_cat in list(mapp.keys()):
        base_cats = mapp[target_cat]
        ml_cat_var = ml.data[ml.cat_var]
        if (not any(in1d(base_cats, ml_cat_var.unique()))) or (len(base_cats) == 1):
            continue

        target_cat_index = in1d(target_df[cat_var_target].values, target_cat)
        ml_cat_index = in1d(ml.data[ml.cat_var].values, base_cats)

        data_ml_train = ml.data.loc[ml_cat_index, :]
        data_ml_target = target_df.loc[target_cat_index, :]

        target_cats = data_ml_target["g_new_c2c"]
        data_ml_target_uniq = data_ml_target.drop_duplicates(
            subset=["index_c2c"] + list(ml.features)
        )
        index_c2c = data_ml_target_uniq["index_c2c"].values

        for m in ml.models:
            ml_name = type(m).__name__
            ml_colname = "wei_" + ml_name + "_c2c"

            try:
                m.fit(X=data_ml_train.loc[:, ml.features], y=data_ml_train[ml.cat_var])

                X_test = data_ml_target_uniq.loc[:, ml.features]
                preds = m.predict_proba(X=X_test)

                preds_df = DataFrame(preds)
                preds_df.columns = m.classes_
                preds_df[setdiff1d(target_cats.unique(), m.classes_)] = 0
                preds_df["index_c2c"] = index_c2c
                preds_df_melt = preds_df.melt(id_vars="index_c2c", var_name="g_new_c2c")
                merge_on = ["index_c2c", "g_new_c2c"]
                p_order = target_df.loc[target_cat_index, merge_on].merge(
                    preds_df_melt, on=merge_on, how="left"
                )
                target_df.loc[target_cat_index, ml_colname] = p_order["value"].values
            except:
                pass
