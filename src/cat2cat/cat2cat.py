import pandas as pd
from cat2cat.mappings import get_mappings, get_freqs, cat_apply_freq
from cat2cat.datasets import load_trans, load_occup
from cat2cat.dataclass import cat2cat_data, cat2cat_mappings, cat2cat_ml

from sklearn.base import BaseEstimator

from dataclasses import dataclass
from typing import Optional


def cat2cat(data: cat2cat_data, mappings: cat2cat_mappings, ml: cat2cat_ml):
    """

    Args:
        data (cat2cat_data): _description_
        mappings (cat2cat_mappings): _description_
        ml (cat2cat_ml): _description_

    Returns:
        _type_: _description_
    """
    isinstance(data, cat2cat_data)
    isinstance(mappings, cat2cat_mappings)
    isinstance(ml, cat2cat_ml)

    mapps = mappings.get_mappings()

    if mappings.direction == "forward":
        mapp = mapps["to_old"]
        return None
    elif mappings.direction == "backward":
        mapp = mapps["to_new"]
        return None
    return None
