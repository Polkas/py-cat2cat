import pytest
from cat2cat.datasets import load_trans, load_occup
from cat2cat import cat2cat
from cat2cat.dataclass import cat2cat_data, cat2cat_mappings, cat2cat_ml
from pandas import DataFrame

from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

trans = load_trans()

occup = load_occup()
occup_small = load_occup(small=True)
o_2006 = occup.loc[occup.year == 2006, :].copy()
o_2008 = o_old = occup.loc[occup.year == 2008, :].copy()
o_2010 = o_new = occup.loc[occup.year == 2010, :].copy()
o_2012 = occup.loc[occup.year == 2012, :].copy()

data = cat2cat_data(o_old, o_new, "code", "code", "year")
mappings = cat2cat_mappings(trans, "backward")


def test_cat2cat_base():
    c2c = cat2cat(data, mappings)
    assert isinstance(c2c, dict)
    assert sorted(list(c2c.keys())) == ["new", "old"]


ml = cat2cat_ml(
    occup.loc[occup.year >= 2010, :].copy(),
    "code",
    ["salary", "age", "edu", "sex"],
    [LinearDiscriminantAnalysis(), DecisionTreeClassifier()],
)


def test_cat2cat_ml():
    c2c = cat2cat(data, mappings, ml)
    assert isinstance(c2c, dict)
    assert sorted(list(c2c.keys())) == ["new", "old"]


# import time
# res = list()
# for i in range(5):
#     start_time = time.time()
#     cat2cat(data, mappings, ml)
#     res.append(time.time() - start_time)
# sum(res) / 5


# import cProfile
# cProfile.run("cat2cat(data, mappings, ml)", "program.prof")
# snakeviz program.prof
