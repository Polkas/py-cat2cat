from cat2cat.mappings import get_mappings, get_freqs, cat_apply_freq
from cat2cat.datasets import load_trans, load_occup

from pandas import DataFrame
from sklearn.base import ClassifierMixin
from dataclasses import dataclass
from typing import Type, List, Dict, Any, Optional, Union


@dataclass(frozen=True)
class cat2cat_data:
    """The dataclass to represent a data argument used in cat2cat procedure

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
        assert (
            isinstance(self.id_var, str)
            and (
                (self.id_var in self.old.columns) and (self.id_var in self.new.columns)
            )
        ) or (
            self.id_var is None
        ), "id_var has to be a str and the new DataFrame column, or None"
        assert isinstance(self.multiplier_var, str) or (
            self.multiplier_var == None
        ), "multiplier_var has to be a str or None"


@dataclass(frozen=True)
class cat2cat_mappings:
    """The dataclass to represent a mappings argument used in cat2cat procedure

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
    freqs: Union[Dict[Any, int], None] = None

    def get_mappings(self) -> Dict[str, Dict[str, List[Any]]]:
        return get_mappings(self.trans)

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
        ), "freqs has to be a pandas.DataFrame with 2 columns"


@dataclass(frozen=True)
class cat2cat_ml:
    """The dataclass to represent a ml argument used in cat2cat procedure

    Args:
        data (DataFrame): dataset with features and the `cat_var`.
        cat_var (str): the dependent variable name.
        features (List[str]): list of features names where all have to be numeric or logical
        models (List[ClassifierMixin]): scikit-learn instances (classes inherit from ClassifierMixin) like,
         RandomForestClassifier or LinearDiscriminantAnalysis
    """

    data: DataFrame
    cat_var: str
    features: List[str]
    models: List[ClassifierMixin]

    def __post_init__(self):
        assert isinstance(self.data, DataFrame), "data has to be a pandas.DataFrame"
        assert isinstance(self.cat_var, str), "cat_var has to be a str"
        assert isinstance(self.features, list), "features has to be a list"
        assert isinstance(self.models, list), "models has to be a list"
        assert all(
            [issubclass(type(e), ClassifierMixin) for e in self.models]
        ), "models arg elements have to be subclass of ClassifierMixin each"
        assert all(
            [hasattr(e, "predict_proba") for e in self.models]
        ), "models have to have the (multi-label) predict_proba method"
