from cat2cat import cat2cat
from cat2cat import cat2cat_ml_run
from cat2cat.dataclass import cat2cat_data, cat2cat_mappings, cat2cat_ml
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from cat2cat.datasets import load_trans, load_occup
from numpy.random import seed
from numpy import nan

trans = load_trans()
occup = load_occup()
o_old = occup.loc[occup.year == 2008, :].copy()
o_new = occup.loc[occup.year == 2010, :].copy()


def test_cat2cat_ml_run_repr():
    mappings = cat2cat_mappings(trans=trans, direction="backward")
    ml = cat2cat_ml(
        occup.loc[occup.year >= 2010, :].copy(),
        "code",
        ["salary", "age", "edu", "sex", "parttime"],
        [
            DecisionTreeClassifier(random_state=1234),
            LinearDiscriminantAnalysis(),
        ],
    )
    expected = repr(cat2cat_ml_run(mappings=mappings, ml=ml))
    actual = "Average Accuracy naive: 0.18\nAverage Accuracy most_freq: 0.54\nAverage Accuracy DecisionTreeClassifier: 0.488\nAverage Accuracy LinearDiscriminantAnalysis: 0.542\n\nPercent of failed DecisionTreeClassifier: 35.369\nPercent of failed LinearDiscriminantAnalysis: 35.369\n\nPercent of better DecisionTreeClassifier over most frequent category solution: 42.52\nPercent of better LinearDiscriminantAnalysis over most frequent category solution: 49.606\nPercent of better DecisionTreeClassifier over naive solution: 88.976\nPercent of better LinearDiscriminantAnalysis over naive solution: 91.732\n\nFeatures: ['salary', 'age', 'edu', 'sex', 'parttime']\nTest sample size: 20.0\n"
    assert actual == expected

    expected = repr(cat2cat_ml_run(mappings=mappings, ml=ml, test_prop=0.3))
    actual = "Average Accuracy naive: 0.18\nAverage Accuracy most_freq: 0.54\nAverage Accuracy DecisionTreeClassifier: 0.488\nAverage Accuracy LinearDiscriminantAnalysis: 0.542\n\nPercent of failed DecisionTreeClassifier: 35.369\nPercent of failed LinearDiscriminantAnalysis: 35.369\n\nPercent of better DecisionTreeClassifier over most frequent category solution: 42.52\nPercent of better LinearDiscriminantAnalysis over most frequent category solution: 49.606\nPercent of better DecisionTreeClassifier over naive solution: 88.976\nPercent of better LinearDiscriminantAnalysis over naive solution: 91.732\n\nFeatures: ['salary', 'age', 'edu', 'sex', 'parttime']\nTest sample size: 30.0\n"
    assert actual == expected

    expected = repr(
        cat2cat_ml_run(mappings=mappings, ml=ml, test_prop=0.9, split_seed=1234)
    )
    actual = "Average Accuracy naive: 0.18\nAverage Accuracy most_freq: 0.53\nAverage Accuracy DecisionTreeClassifier: 0.48\nAverage Accuracy LinearDiscriminantAnalysis: 0.551\n\nPercent of failed DecisionTreeClassifier: 35.369\nPercent of failed LinearDiscriminantAnalysis: 35.369\n\nPercent of better DecisionTreeClassifier over most frequent category solution: 43.701\nPercent of better LinearDiscriminantAnalysis over most frequent category solution: 53.15\nPercent of better DecisionTreeClassifier over naive solution: 86.614\nPercent of better LinearDiscriminantAnalysis over naive solution: 91.339\n\nFeatures: ['salary', 'age', 'edu', 'sex', 'parttime']\nTest sample size: 90.0\n"
    assert actual == expected

    mappings = cat2cat_mappings(trans=trans, direction="forward")
    ml = cat2cat_ml(
        occup.loc[occup.year <= 2008, :].copy(),
        "code",
        ["salary", "age", "edu", "sex"],
        [DecisionTreeClassifier(random_state=1234), LinearDiscriminantAnalysis()],
    )
    expected = repr(cat2cat_ml_run(mappings=mappings, ml=ml, test_prop=0.3))
    actual = "Average Accuracy naive: 0.439\nAverage Accuracy most_freq: 0.69\nAverage Accuracy DecisionTreeClassifier: 0.63\nAverage Accuracy LinearDiscriminantAnalysis: 0.692\n\nPercent of failed DecisionTreeClassifier: 98.291\nPercent of failed LinearDiscriminantAnalysis: 98.291\n\nPercent of better DecisionTreeClassifier over most frequent category solution: 43.182\nPercent of better LinearDiscriminantAnalysis over most frequent category solution: 54.545\nPercent of better DecisionTreeClassifier over naive solution: 81.818\nPercent of better LinearDiscriminantAnalysis over naive solution: 84.091\n\nFeatures: ['salary', 'age', 'edu', 'sex']\nTest sample size: 30.0\n"
    assert actual == expected


def test_cat2cat_ml_run_get_raw():
    mappings = cat2cat_mappings(trans=trans, direction="backward")
    ml = cat2cat_ml(
        occup.loc[occup.year >= 2010, :].copy(),
        "code",
        ["salary", "age", "edu", "sex"],
        [DecisionTreeClassifier(random_state=1234), LinearDiscriminantAnalysis()],
    )
    expected = cat2cat_ml_run(mappings=mappings, ml=ml).get_raw()["7431"]
    actual = {
        "naive": nan,
        "freq": nan,
        "DecisionTreeClassifier": nan,
        "LinearDiscriminantAnalysis": nan,
    }
    assert str(actual) == str(expected)
