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


def test_cat2cat_ml_run_repr():
    mappings = cat2cat_mappings(trans=trans, direction="backward")
    ml = cat2cat_ml(
        occup.loc[occup.year >= 2010, :].copy(),
        "code",
        ["salary", "age", "edu", "sex"],
        [
            DecisionTreeClassifier(random_state=1234),
            LinearDiscriminantAnalysis(),
        ],
    )
    expected = repr(cat2cat_ml_run(mappings=mappings, ml=ml))
    actual = "Average Accuracy naive: 0.302\nAverage Accuracy most_freq: 0.54\nAverage Accuracy DecisionTreeClassifier: 0.487\nAverage Accuracy LinearDiscriminantAnalysis: 0.553\n\nPercent of failed DecisionTreeClassifier: 35.369\nPercent of failed LinearDiscriminantAnalysis: 35.369\n\nPercent of better DecisionTreeClassifier over most frequent category solution: 27.99\nPercent of better LinearDiscriminantAnalysis over most frequent category solution: 32.57\n\nFeatures: ['salary', 'age', 'edu', 'sex']\nTest sample size: 0.2\n"
    assert actual == expected

    expected = repr(cat2cat_ml_run(mappings=mappings, ml=ml, test_size=0.3))
    actual = "Average Accuracy naive: 0.302\nAverage Accuracy most_freq: 0.55\nAverage Accuracy DecisionTreeClassifier: 0.477\nAverage Accuracy LinearDiscriminantAnalysis: 0.54\n\nPercent of failed DecisionTreeClassifier: 35.369\nPercent of failed LinearDiscriminantAnalysis: 35.369\n\nPercent of better DecisionTreeClassifier over most frequent category solution: 25.954\nPercent of better LinearDiscriminantAnalysis over most frequent category solution: 30.025\n\nFeatures: ['salary', 'age', 'edu', 'sex']\nTest sample size: 0.3\n"
    assert actual == expected

    expected = repr(
        cat2cat_ml_run(mappings=mappings, ml=ml, test_size=0.9, split_seed=1234)
    )
    actual = "Average Accuracy naive: 0.302\nAverage Accuracy most_freq: 0.49\nAverage Accuracy DecisionTreeClassifier: 0.47\nAverage Accuracy LinearDiscriminantAnalysis: 0.491\n\nPercent of failed DecisionTreeClassifier: 60.814\nPercent of failed LinearDiscriminantAnalysis: 60.814\n\nPercent of better DecisionTreeClassifier over most frequent category solution: 16.794\nPercent of better LinearDiscriminantAnalysis over most frequent category solution: 18.83\n\nFeatures: ['salary', 'age', 'edu', 'sex']\nTest sample size: 0.9\n"
    assert actual == expected

    mappings = cat2cat_mappings(trans=trans, direction="forward")
    ml = cat2cat_ml(
        occup.loc[occup.year <= 2008, :].copy(),
        "code",
        ["salary", "age", "edu", "sex"],
        [DecisionTreeClassifier(random_state=1234), LinearDiscriminantAnalysis()],
    )
    expected = repr(cat2cat_ml_run(mappings=mappings, ml=ml, test_size=0.3))
    actual = "Average Accuracy naive: 0.987\nAverage Accuracy most_freq: 0.69\nAverage Accuracy DecisionTreeClassifier: 0.647\nAverage Accuracy LinearDiscriminantAnalysis: 0.708\n\nPercent of failed DecisionTreeClassifier: 98.291\nPercent of failed LinearDiscriminantAnalysis: 98.291\n\nPercent of better DecisionTreeClassifier over most frequent category solution: 0.699\nPercent of better LinearDiscriminantAnalysis over most frequent category solution: 0.932\n\nFeatures: ['salary', 'age', 'edu', 'sex']\nTest sample size: 0.3\n"
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
        "naive": 0.3333333333333333,
        "freq": nan,
        "DecisionTreeClassifier": nan,
        "LinearDiscriminantAnalysis": nan,
    }
    assert str(actual) == str(expected)
