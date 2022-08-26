import pytest

from cat2cat.datasets import load_trans, load_occup
from cat2cat import cat2cat
from cat2cat.dataclass import cat2cat_data, cat2cat_mappings, cat2cat_ml
from cat2cat.cat2cat_utils import prune_c2c, dummy_c2c

from pandas import DataFrame
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis


trans = load_trans()

occup = load_occup()
o_2006 = occup.loc[occup.year == 2006, :]
o_2008 = o_old = occup.loc[occup.year == 2008, :]
o_2010 = o_new = occup.loc[occup.year == 2010, :]
o_2012 = occup.loc[occup.year == 2012, :]

data = cat2cat_data(o_old, o_new, "code", "code", "year")
mappings = cat2cat_mappings(trans, "backward")

c2c = cat2cat(data, mappings)


def test_prune():
    assert isinstance(prune_c2c(c2c["old"], lambda x: [e > 0 for e in x]), DataFrame)


def test_dummy_return():
    expected_cols = list(occup.columns) + [
        "index_c2c",
        "g_new_c2c",
        "rep_c2c",
        "wei_naive_c2c",
        "wei_freq_c2c",
    ]
    assert isinstance(dummy_c2c(occup, "code"), DataFrame)
    assert list(dummy_c2c(occup, "code").columns) == expected_cols


def test_dummy_return_ml():
    expected_cols = list(occup.columns) + [
        "index_c2c",
        "g_new_c2c",
        "rep_c2c",
        "wei_naive_c2c",
        "wei_freq_c2c",
        "wei_LinearDiscriminantAnalysis_c2c",
    ]
    assert isinstance(dummy_c2c(occup, "code"), DataFrame)
    assert (
        list(dummy_c2c(occup, "code", ["LinearDiscriminantAnalysis"]).columns)
        == expected_cols
    )
