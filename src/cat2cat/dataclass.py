from cat2cat.mappings import get_mappings, get_freqs, cat_apply_freq
from cat2cat.datasets import load_trans, load_occup

from pandas import DataFrame
from sklearn.base import ClassifierMixin
from dataclasses import dataclass
from typing import Optional
from typing import Type, List


@dataclass(frozen=True)
class cat2cat_data:
    """The dataclass to represent a data argument used in cat2cat procedure"""

    old: DataFrame
    new: DataFrame
    cat_var_old: str
    cat_var_new: str
    time_var: str
    id_var: Optional[str] = None
    multiplier_varid_var: Optional[str] = None
    multiplier_var: Optional[str] = None

    def __post_init__(self):
        assert isinstance(self.old, DataFrame), "old has to be a pandas.DataFrame"
        assert isinstance(self.new, DataFrame), "new has to be a pandas.DataFrame"
        assert isinstance(self.cat_var_old, str) and (
            self.cat_var_old in self.old
        ), "cat_var_old has to be a str and the old DataFrame column"
        assert isinstance(self.cat_var_new, str) and (
            self.cat_var_new in self.new
        ), "cat_var_new has to be a str and the new DataFrame column"
        assert (
            isinstance(self.id_var, str)
            and ((self.id_var in self.old) and (self.id_var in self.new))
        ) or (
            self.id_var is None
        ), "id_var has to be a str and the new DataFrame column, or None"
        assert isinstance(self.multiplier_var, str) or (
            self.multiplier_var == None
        ), "multiplier_var has to be a str or None"


@dataclass(frozen=True)
class cat2cat_mappings:
    """The dataclass to represent a mappings argument used in cat2cat procedure"""

    trans: DataFrame
    direction: str
    freqs: dict = None

    def get_mappings(self):
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
    """The dataclass to represent a ml argument used in cat2cat procedure"""

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
