import pytest
from cat2cat.datasets import load_trans, load_occup
from cat2cat import cat2cat
from cat2cat.dataclass import cat2cat_data, cat2cat_mappings, cat2cat_ml
from pandas import DataFrame
from sklearn.ensemble import RandomForestClassifier


data = cat2cat_data(DataFrame(), DataFrame(), "cat_var", "cat_var", "year")
mappings = cat2cat_mappings(DataFrame({"a": [1], "b": [2]}), "forward")
ml = cat2cat_ml(DataFrame(), "cat_var", ["salary", "age"], RandomForestClassifier())


def test_cat2cat_base():
    assert cat2cat(data, mappings, ml) == None
