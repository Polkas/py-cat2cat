from pandas import DataFrame
from sklearn.base import ClassifierMixin
from dataclasses import dataclass
from typing import Sequence, Dict, Any, Optional

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

    def __post_init__(self):
        assert isinstance(self.old, DataFrame), "old has to be a pandas.DataFrame"
        assert isinstance(self.new, DataFrame), "new has to be a pandas.DataFrame"
        assert isinstance(self.cat_var_old, str) and (
            self.cat_var_old in self.old.columns
        ), "cat_var_old has to be a str and the old DataFrame column"
        assert isinstance(self.cat_var_new, str) and (
            self.cat_var_new in self.new.columns
        ), "cat_var_new has to be a str and the new DataFrame column"
        assert isinstance(self.time_var, str) and (
            (self.time_var in self.new.columns) and (self.time_var in self.old.columns)
        ), "time_var has to be a str and in the both (old and new) DataFrames"
        assert (len(self.old[self.time_var].unique()) == 1) and (
            len(self.new[self.time_var].unique()) == 1
        ), "time_var has to have only the one period in each DataFrame (old and new)"
        assert (
            isinstance(self.id_var, str)
            and (
                (self.id_var in self.old.columns) and (self.id_var in self.new.columns)
            )
        ) or (
            self.id_var is None
        ), "id_var has to be a str and in the both (old and new) DataFrames, or None"
        assert (
            isinstance(self.multiplier_var, str)
            and (
                (self.multiplier_var in self.new.columns)
                or (self.multiplier_var in self.old.columns)
            )
        ) or (
            self.multiplier_var == None
        ), "multiplier_var has to be a str and in one of (old and new) DataFrames, or None"


@dataclass(frozen=True)
class cat2cat_mappings:
    """The dataclass to represent the mappings argument used in the cat2cat procedure

    Args:
        trans (DataFrame): mapping (transition) table (with 2 columns, old and new encoding) - all categories for cat_var in old and new datasets have to be included.
        diretion (str): "backward" or "forward"
        freqs (Optional[str]): If It is not provided then is assessed automatically.
                               Artificial counts for each variable level in the base period.
                               It is optional nevertheless will be often needed, as gives more control.

    Note:
        The mapping (transition) table should to have a candidate for each category from the targeted for an update period.
        The observation from targeted for an updated period without a matched category from base period is removed.
    """

    trans: DataFrame
    direction: str
    freqs: Optional[Dict[Any, int]] = None

    def __post_init__(self):
        assert isinstance(self.trans, DataFrame), "trans has to be a pandas.DataFrame"
        assert self.trans.shape[1] == 2, "trans has to have two columns"
        assert isinstance(self.direction, str), "direction has to be a str"
        assert self.direction in [
            "forward",
            "backward",
        ], "direction has to be one of 'forward' or 'backward'"
        assert (self.freqs == None) or isinstance(
            self.freqs, dict
        ), "freqs has to be a pandas.DataFrame with 2 columns, or None"


@dataclass(frozen=True)
class cat2cat_ml:
    """The dataclass to represent the ml argument used in the cat2cat procedure

    Args:
        data (DataFrame): dataset with features and the `cat_var`.
        cat_var (str): the dependent variable name.
        features (Sequence[str]): list of features names where all have to be numeric or logical
        models (Sequence[ClassifierMixin]): scikit-learn instances (classes inherit from ClassifierMixin) like,
                                        RandomForestClassifier() or LinearDiscriminantAnalysis() instances.
    """

    data: DataFrame
    cat_var: str
    features: Sequence[str]
    models: Sequence[ClassifierMixin]

    def __post_init__(self):
        assert isinstance(self.data, DataFrame), "data has to be a pandas.DataFrame"
        assert isinstance(self.cat_var, str) and (
            self.cat_var in self.data.columns
        ), "cat_var has to be a str and a data argument column"
        assert isinstance(self.features, Sequence) and all(
            [e in self.data.columns for e in self.features]
        ), "features has to be a list-like and each have to be a column in the data argument."
        assert isinstance(self.models, Sequence) and (
            len(self.models) > 0
        ), "models has to be a list-like of length at least 1."
        assert all(
            [issubclass(type(e), ClassifierMixin) for e in self.models]
        ), "models arg elements have to be subclass of ClassifierMixin each"
        assert all(
            [hasattr(e, "fit") for e in self.models]
        ), "each model has to have the fit method"
        assert all(
            [hasattr(e, "predict_proba") for e in self.models]
        ), "each model has to have the (multi-label) predict_proba method"
