from dataclasses import dataclass
from typing import Any, Dict, Optional, Sequence

from pandas import DataFrame
from sklearn.base import ClassifierMixin

__all__ = ["cat2cat_data", "cat2cat_mappings", "cat2cat_ml"]


@dataclass(frozen=True)
class cat2cat_data:
    """The dataclass to represent the data argument used in the cat2cat procedure

    Args:
        old (DataFrame): older time point in a panel, has to have all columns set in the rest of arguments.
        new (DataFrame): newer time point in a panel, has to have all columns set in the rest of arguments.
        cat_var_old (str): name of the categorical variable in the older time point.
        cat_var_new (str): name of the categorical variable in the newer time point.
        time_var (str): name of the time variable.
        id_var (Optional[str]): name of the unique identifier variable - if this is specified then for subjects observe in both periods the direct mapping is applied.
        multiplier_var (Optional[str]): name of the multiplier variable - number of replication needed to reproduce the population.
    """

    old: DataFrame
    new: DataFrame
    cat_var_old: str
    cat_var_new: str
    time_var: str
    id_var: Optional[str] = None
    multiplier_var: Optional[str] = None

    def __post_init__(self) -> None:
        if not isinstance(self.old, DataFrame):
            raise TypeError("old has to be a pandas.DataFrame")
        if not isinstance(self.new, DataFrame):
            raise TypeError("new has to be a pandas.DataFrame")
        if not isinstance(self.cat_var_old, str) or self.cat_var_old not in self.old.columns:
            raise ValueError("cat_var_old has to be a str and the old DataFrame column")
        if not isinstance(self.cat_var_new, str) or self.cat_var_new not in self.new.columns:
            raise ValueError("cat_var_new has to be a str and the new DataFrame column")
        if (
            not isinstance(self.time_var, str)
            or self.time_var not in self.old.columns
            or self.time_var not in self.new.columns
        ):
            raise ValueError("time_var has to be a str and in the both (old and new) DataFrames")
        if (len(self.old[self.time_var].unique()) != 1) or (
            len(self.new[self.time_var].unique()) != 1
        ):
            raise ValueError(
                "time_var has to have only the one period in each DataFrame (old and new)"
            )
        if self.id_var is not None and (
            not isinstance(self.id_var, str)
            or self.id_var not in self.old.columns
            or self.id_var not in self.new.columns
        ):
            raise ValueError(
                "id_var has to be a str and in the both (old and new) DataFrames, or None"
            )
        if self.multiplier_var is not None and (
            not isinstance(self.multiplier_var, str)
            or (
                self.multiplier_var not in self.old.columns
                and self.multiplier_var not in self.new.columns
            )
        ):
            raise ValueError(
                "multiplier_var has to be a str and in one of (old and new) DataFrames, or None"
            )


@dataclass(frozen=True)
class cat2cat_mappings:
    """The dataclass to represent the mappings argument used in the cat2cat procedure

    Args:
        trans (DataFrame): mapping (transition) table (with 2 columns, old and new encoding) - all categories for cat_var in old and new datasets have to be included.
        diretion (str): "backward" or "forward"
        freqs (Optional[Dict[Any, int]]): If It is not provided then is assessed automatically.
                            Artificial counts for each variable level in the base period.
                            It is optional nevertheless will be often needed, as gives more control.

    Note:
        The mapping (transition) table should to have a candidate for each category from the targeted for an update period.
        The observation from targeted for an updated period without a matched category from base period is removed.
    """

    trans: DataFrame
    direction: str
    freqs: Optional[Dict[Any, int]] = None

    def __post_init__(self) -> None:
        if not isinstance(self.trans, DataFrame):
            raise TypeError("trans has to be a pandas.DataFrame")
        if self.trans.shape[1] != 2:
            raise ValueError("trans has to have two columns")
        if not isinstance(self.direction, str):
            raise TypeError("direction has to be a str")
        if self.direction not in ["forward", "backward"]:
            raise ValueError("direction has to be one of 'forward' or 'backward'")
        if self.freqs is not None and not isinstance(self.freqs, dict):
            raise TypeError("freqs has to be a dict, or None")


@dataclass(frozen=True)
class cat2cat_ml:
    """The dataclass to represent the ml argument used in the cat2cat procedure

    Args:
        data (DataFrame): dataset with features and the `cat_var`.
        cat_var (str): the dependent variable name.
        features (Sequence[str]): list of feature names. Numeric/logical columns are used directly;
                      categorical/object/string columns are one-hot encoded by the ML helpers.
        models (Sequence[ClassifierMixin]): scikit-learn classifier instances.
        on_fail (str): how failed ML weights are handled: "freq", "naive", "na", or "error".
        fail_warn (bool): warn when failed ML weights are replaced or retained as missing.
    """

    data: DataFrame
    cat_var: str
    features: Sequence[str]
    models: Sequence[ClassifierMixin]
    on_fail: str = "freq"
    fail_warn: bool = True

    def __post_init__(self) -> None:
        if not isinstance(self.data, DataFrame):
            raise TypeError("data has to be a pandas.DataFrame")
        if not isinstance(self.cat_var, str) or self.cat_var not in self.data.columns:
            raise ValueError("cat_var has to be a str and a data argument column")
        if not isinstance(self.features, Sequence) or not all(
            e in self.data.columns for e in self.features
        ):
            raise ValueError(
                "features has to be a list-like and each have to be a column in the data argument."
            )
        if not isinstance(self.models, Sequence) or len(self.models) == 0:
            raise ValueError("models has to be a list-like of length at least 1.")
        if not all(issubclass(type(e), ClassifierMixin) for e in self.models):
            raise TypeError(
                "models arg elements have to be subclass of ClassifierMixin each"
            )
        if not all(hasattr(e, "fit") for e in self.models):
            raise TypeError("each model has to have the fit method")
        if not all(hasattr(e, "predict_proba") for e in self.models):
            raise TypeError("each model has to have the (multi-label) predict_proba method")
        if not isinstance(self.on_fail, str) or self.on_fail.lower() not in {
            "freq",
            "naive",
            "na",
            "error",
        }:
            raise ValueError("on_fail has to be one of: 'freq', 'naive', 'na', or 'error'")
        if not isinstance(self.fail_warn, bool):
            raise TypeError("fail_warn has to be a bool")
