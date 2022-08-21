import pytest

from cat2cat.datasets import load_trans, load_occup
from cat2cat import cat2cat
from cat2cat.dataclass import cat2cat_data, cat2cat_mappings, cat2cat_ml
from cat2cat.cat2cat_utils import prune_c2c, cross_c2c, dummy_c2c

from pandas import DataFrame
from sklearn.ensemble import RandomForestClassifier


trans = load_trans()

occup = load_occup()
o_2006 = occup.loc[occup.year == 2006, :]
o_2008 = o_old = occup.loc[occup.year == 2008, :]
o_2010 = o_new = occup.loc[occup.year == 2010, :]
o_2012 = occup.loc[occup.year == 2012, :]

data = cat2cat_data(o_old, o_new, "cat_var", "cat_var", "year")
mappings = cat2cat_mappings(trans, "forward")
ml = cat2cat_ml(o_new, "cat_var", ["salary", "age"], [RandomForestClassifier()])


def test_prune():
    assert isinstance(prune_c2c(DataFrame()), DataFrame)


def test_cross():
    assert isinstance(cross_c2c(DataFrame()), DataFrame)


def test_dummy():
    assert isinstance(dummy_c2c(occup, "code"), DataFrame)
