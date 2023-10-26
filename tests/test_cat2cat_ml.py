from cat2cat import cat2cat
from cat2cat import cat2cat_ml_run
from cat2cat.dataclass import cat2cat_data, cat2cat_mappings, cat2cat_ml
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.tree import DecisionTreeClassifier
from cat2cat.datasets import load_trans, load_occup
from numpy.random import seed
from numpy import nan

trans = load_trans()
occup = load_occup()
o_old = occup.loc[occup.year == 2008, :].copy()
o_new = occup.loc[occup.year == 2010, :].copy()
mappings = cat2cat_mappings(trans=trans, direction="backward")
ml = cat2cat_ml(
    occup.loc[occup.year >= 2010, :].copy(),
    "code",
    ["salary", "age", "edu", "sex"],
    [DecisionTreeClassifier(), LinearDiscriminantAnalysis()],
)


def test_cat2cat_ml_run_repr():
    seed(1234)
    expected = repr(cat2cat_ml_run(mappings=mappings, ml=ml))
    actual = (
        "Average Accuracy naive: 0.302\nAverage Accuracy most_freq: 0.53\n"
        + "Average Accuracy DecisionTreeClassifier: 0.486\n"
        + "Average Accuracy LinearDiscriminantAnalysis: 0.545\n"
        + "\nPercent of failed DecisionTreeClassifier: 29.771\n"
        + "Percent of failed LinearDiscriminantAnalysis: 29.771\n"
        + "\nPercent of better DecisionTreeClassifier over most frequent category solution: 31.043\n"
        + "Percent of better LinearDiscriminantAnalysis over most frequent category solution: 35.115\n"
        + "\nFeatures: ['salary', 'age', 'edu', 'sex']\nTest sample size: 0.2\n"
    )
    assert actual == expected

    seed(1234)
    expected = repr(cat2cat_ml_run(mappings=mappings, ml=ml, test_size=0.3))
    actual = (
        "Average Accuracy naive: 0.302\nAverage Accuracy most_freq: 0.54\n"
        + "Average Accuracy DecisionTreeClassifier: 0.475\n"
        + "Average Accuracy LinearDiscriminantAnalysis: 0.535\n"
        + "\nPercent of failed DecisionTreeClassifier: 29.771\n"
        + "Percent of failed LinearDiscriminantAnalysis: 30.025\n"
        + "\nPercent of better DecisionTreeClassifier over most frequent category solution: 28.244\n"
        + "Percent of better LinearDiscriminantAnalysis over most frequent category solution: 32.824\n"
        + "\nFeatures: ['salary', 'age', 'edu', 'sex']\nTest sample size: 0.3\n"
    )
    assert actual == expected


def test_cat2cat_ml_run_get_raw():
    seed(1234)
    expected = cat2cat_ml_run(mappings=mappings, ml=ml).get_raw()["7431"]
    actual = {
        "naive": 0.3333333333333333,
        "freq": nan,
        "DecisionTreeClassifier": nan,
        "LinearDiscriminantAnalysis": nan,
    }
    assert str(actual) == str(expected)
