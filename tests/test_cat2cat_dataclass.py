from cat2cat.datasets import load_trans, load_occup
from cat2cat import cat2cat
from cat2cat.dataclass import cat2cat_data, cat2cat_mappings, cat2cat_ml
from pandas import DataFrame
from sklearn.ensemble import RandomForestClassifier

from dataclasses import FrozenInstanceError
import pytest

trans = load_trans()

occup = load_occup()
o_2006 = occup.loc[occup.year == 2006, :].copy()
o_2008 = o_old = occup.loc[occup.year == 2008, :].copy()
o_2010 = o_new = occup.loc[occup.year == 2010, :].copy()
o_2012 = occup.loc[occup.year == 2012, :].copy()

# cat2cat_data
def test_cat2cat_data():
    data = cat2cat_data(o_old, o_new, "code", "code", "year")
    assert isinstance(data, cat2cat_data)
    with pytest.raises(FrozenInstanceError):
        data.old = 1


# cat2cat_mappings
def test_cat2cat_mappings():
    mappings = cat2cat_mappings(trans, "backward")
    assert isinstance(mappings, cat2cat_mappings)
    with pytest.raises(FrozenInstanceError):
        mappings.trans = 1


# cat2cat_ml
def test_cat2cat_ml():
    ml = cat2cat_ml(o_new, "code", ["salary", "age"], [RandomForestClassifier()])
    assert isinstance(ml, cat2cat_ml)
    with pytest.raises(FrozenInstanceError):
        ml.data = 1
