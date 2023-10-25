from cat2cat import cat2cat
from cat2cat import cat2cat_ml_run
from cat2cat.dataclass import cat2cat_data, cat2cat_mappings, cat2cat_ml
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.tree import DecisionTreeClassifier
from cat2cat.datasets import load_trans, load_occup

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
cat2cat_ml_run(mappings=mappings, ml=ml)

