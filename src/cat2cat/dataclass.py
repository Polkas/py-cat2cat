import pandas as pd
from cat2cat.mappings import get_mappings, get_freqs, cat_apply_freq
from cat2cat.datasets import load_trans, load_occup

from sklearn.base import ClassifierMixin
from dataclasses import dataclass
from typing import Optional
from typing import Type


@dataclass
class cat2cat_data:
    """
    The dataclass to represent a data argument used in cat2cat procedure
    """

    old: pd.DataFrame
    new: pd.DataFrame
    cat_var_old: str
    cat_var_new: str
    time_var: str
    id_var: str = None
    multiplier_varid_var: Optional[str] = None
    multiplier_var: Optional[str] = None

    def __post_init__(self):
        assert isinstance(self.old, pd.DataFrame), "old has to be a pandas.DataFrame"
        assert isinstance(
            self.old, pd.core.frame.DataFrame
        ), "new has to be a pandas.DataFrame"
        assert isinstance(self.cat_var_old, str), "cat_var_old has to be a str"
        assert isinstance(self.cat_var_new, str), "cat_var_new has to be a str"
        assert isinstance(self.time_var, str), "time_var has to be a str"
        assert isinstance(self.id_var, str) or (
            self.id_var == None
        ), "id_var has to be a str or None"
        assert isinstance(self.multiplier_var, str) or (
            self.multiplier_var == None
        ), "multiplier_var has to be a str or None"


@dataclass
class cat2cat_mappings:
    """
    The dataclass to represent a mappings argument used in cat2cat procedure
    """

    trans: pd.DataFrame
    direction: str
    freqs_df: pd.DataFrame = None

    def get_mappings(self):
        return get_mappings(self.trans)

    def __post_init__(self):
        assert isinstance(
            self.trans, pd.core.frame.DataFrame
        ), "trans has to be a pandas.DataFrame"
        assert self.trans.shape[1] == 2, "trans has to have two columns"
        assert (self.freqs_df == None) or (
            isinstance(self.freqs_df, pd.core.frame.DataFrame)
            and (self.freqs_df.shape[1] == 2)
        ), "freqs_df has to be a pandas.DataFrame with 2 columns"
        assert isinstance(self.direction, str), "direction has to be a str"
        assert self.direction in [
            "forward",
            "backward",
        ], "direction has to be one of 'forward' or 'backward'"


@dataclass
class cat2cat_ml:
    """
    The dataclass to represent a ml argument used in cat2cat procedure
    """

    data: pd.DataFrame
    cat_var: str
    features: list[str]
    model: ClassifierMixin

    def __post_init__(self):
        assert isinstance(self.data, pd.DataFrame), "data has to be a pandas.DataFrame"
        assert isinstance(self.cat_var, str), "cat_var has to be a str"
        assert isinstance(self.features, list), "features has to be a list"
        assert issubclass(
            type(self.model), ClassifierMixin
        ), "model has to be subclass of ClassifierMixin"
        assert hasattr(
            self.model, "predict_proba"
        ), "model has to have (multi-label) predict_proba method"
