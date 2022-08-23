import pytest
from cat2cat.datasets import load_trans, load_occup
from cat2cat import cat2cat
from cat2cat.dataclass import cat2cat_data, cat2cat_mappings, cat2cat_ml
from pandas import DataFrame
from sklearn.ensemble import RandomForestClassifier

trans = load_trans()

occup = load_occup()
o_2006 = occup.loc[occup.year == 2006, :].copy()
o_2008 = o_old = occup.loc[occup.year == 2008, :].copy()
o_2010 = o_new = occup.loc[occup.year == 2010, :].copy()
o_2012 = occup.loc[occup.year == 2012, :].copy()

# cat2cat_data
def test_cat2cat_data():
    cat2cat_data(o_old, o_new, "code", "code", "year")


# cat2cat_mappings
# cat2cat_mappings(trans, "backward")


# cat2cat_ml
# cat2cat_ml(o_new, "code", ["salary", "age"], [RandomForestClassifier()])
