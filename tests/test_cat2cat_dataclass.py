from cat2cat.datasets import load_trans, load_occup

from cat2cat.dataclass import cat2cat_data, cat2cat_mappings, cat2cat_ml
from sklearn.ensemble import RandomForestClassifier

from dataclasses import FrozenInstanceError
from pandas import concat

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

    with pytest.raises(AssertionError):
        # only one period in each
        cat2cat_data(concat([o_2006, o_2008]), o_new, "code", "code", "year")

    with pytest.raises(AssertionError):
        # only one period in each
        cat2cat_data(o_old, concat([o_2010, o_2012]), "code", "code", "year")

    with pytest.raises(AssertionError):
        cat2cat_data(o_old, o_new, "WRONG", "code", "year")

    with pytest.raises(AssertionError):
        cat2cat_data(o_old, o_new, "code", "WRONG", "year")

    with pytest.raises(AssertionError):
        cat2cat_data(o_old, o_new, "code", "code", "WRONG")

    with pytest.raises(AssertionError):
        cat2cat_data(1, o_new, "code", "code", "year")

    with pytest.raises(AssertionError):
        cat2cat_data(o_old, 1, "code", "code", "year")

    with pytest.raises(FrozenInstanceError):
        data.old = 1


# cat2cat_mappings
def test_cat2cat_mappings():
    mappings = cat2cat_mappings(trans, "backward")

    assert isinstance(mappings, cat2cat_mappings)

    with pytest.raises(AssertionError):
        cat2cat_mappings(1, "backward")

    with pytest.raises(AssertionError):
        cat2cat_mappings(1, "WRONG")

    with pytest.raises(FrozenInstanceError):
        mappings.trans = 1


# cat2cat_ml
def test_cat2cat_ml():
    ml = cat2cat_ml(o_new, "code", ["salary", "age"], [RandomForestClassifier()])

    assert isinstance(ml, cat2cat_ml)

    ml = cat2cat_ml(o_new, "code", ("salary", "age"), (RandomForestClassifier(),))

    assert isinstance(ml, cat2cat_ml)

    with pytest.raises(AssertionError):
        cat2cat_ml(o_new, "code", ["salary", "age"], [])

    with pytest.raises(AssertionError):
        cat2cat_ml(o_new, "code", ["WRONG", "age"], [RandomForestClassifier()])

    with pytest.raises(AssertionError):
        cat2cat_ml(o_new, "WRONG", ["salary", "age"], [RandomForestClassifier()])

    with pytest.raises(AssertionError):
        cat2cat_ml(1, "WRONG", ["salary", "age"], [RandomForestClassifier()])

    with pytest.raises(FrozenInstanceError):
        ml.data = 1
