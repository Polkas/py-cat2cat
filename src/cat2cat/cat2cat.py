from pandas import DataFrame
from numpy import arange, repeat, array

from cat2cat.mappings import get_mappings, get_freqs, cat_apply_freq
from cat2cat.datasets import load_trans, load_occup
from cat2cat.dataclass import cat2cat_data, cat2cat_mappings, cat2cat_ml
from cat2cat.cat2cat_utils import dummy_c2c

from sklearn.base import BaseEstimator

from dataclasses import dataclass
from typing import Optional


def cat2cat(
    data: cat2cat_data, mappings: cat2cat_mappings, ml: cat2cat_ml
):
    """Automatic mapping in a panel dataset - cat2cat procedure

    Args:
        data (cat2cat_data): dataclass with data related arguments
        mappings (cat2cat_mappings): dataclass with mappings related arguments
        ml (cat2cat_ml): dataclass with ml related arguments

    Returns:
        dict: _description_

    >>> from cat2cat import cat2cat
    >>> from cat2cat.dataclass import cat2cat_data, cat2cat_mappings, cat2cat_ml
    >>> from sklearn.ensemble import RandomForestClassifier
    >>> trans = load_trans()
    >>> occup = load_occup()
    >>> o_old = occup.loc[occup.year == 2008, :].copy()
    >>> o_new = occup.loc[occup.year == 2010, :].copy()
    >>> data_c2c = cat2cat_data(o_old, o_new, "code", "code", "year")
    >>> mappings_c2c = cat2cat_mappings(trans, "forward")
    >>> ml_c2c = cat2cat_ml(o_new, "code", ["salary", "age"], [RandomForestClassifier()])
    >>> cat2cat(data_c2c, mappings_c2c, ml_c2c)
    {...

    """

    isinstance(data, cat2cat_data)
    isinstance(mappings, cat2cat_mappings)
    isinstance(ml, cat2cat_ml)

    mapps = mappings.get_mappings()

    if mappings.direction == "forward":
        cat_var_base = data.cat_var_old
        cat_var_target = data.cat_var_new
        cat_base_year = data.old
        cat_target_year = data.new  # if direct
        cat_mid = None  # if direct
        mapp = mapps["to_old"]
        target_name = "new"
        base_name = "old"
    elif mappings.direction == "backward":
        cat_var_base = data.cat_var_new
        cat_var_target = data.cat_var_old
        cat_base_year = data.new
        cat_target_year = data.old  # if direct
        cat_mid = None  # if direct
        mapp = mapps["to_new"]
        target_name = "old"
        base_name = "new"

    freqs = get_freqs(cat_base_year[cat_var_base])
    mapp_f = cat_apply_freq(mapp, freqs)

    a_mapp = [mapp.get(e, []) for e in cat_target_year[cat_var_target]]
    a_mapp_f = [mapp_f.get(e, []) for e in cat_target_year[cat_var_target]]
    lens = [len(e) for e in a_mapp]
    nrow_target = cat_target_year.shape[0]
    cat_target_year = cat_target_year.iloc[repeat(arange(nrow_target), lens), :]
    nrow_targe_after = cat_target_year.shape[0]
    cat_target_year = cat_target_year.assign(
        index_c2c=arange(nrow_targe_after),
        g_new_c2c=[e for l in a_mapp for e in l],
        rep_c2c=repeat(lens, lens),
    )
    cat_target_year = cat_target_year.assign(
        wei_naive_c2c=1 / cat_target_year.rep_c2c,
        wei_freq_c2c=[e for l in a_mapp_f for e in l],
    )
    cat_target_year = cat_target_year.reset_index(drop=True)
    cat_base_year = dummy_c2c(cat_base_year, cat_var_base)

    # ML
    ml_names = [type(m).__name__ for m in ml.models]
    for k in list(mapp.keys()):
        # data for each possible category
        # predict probabilities for each of them
        pass

    res = dict()
    res[target_name] = cat_target_year
    res[base_name] = cat_base_year

    return res
