# Choosing Weights And Validating ML

Use `wei_freq_c2c` as the default transparent baseline. Add ML only when the
features are informative and validation shows that probability weights improve
over naive or frequency baselines.

`cat2cat` weights are probabilities, not hard classifications. A model can have
reasonable accuracy while still assigning poor probabilities to the true class.
That is why `cat2cat_ml_run()` reports three complementary diagnostics.

## Baseline Weights

- `wei_naive_c2c`: assigns equal probability to every candidate category.
- `wei_freq_c2c`: assigns probabilities from observed base-period frequencies.
- `wei_<Estimator>_c2c`: assigns probabilities predicted by a scikit-learn
    estimator.

Frequency weights are a strong first choice when you want a reproducible and
easy-to-explain method. ML weights are useful when features such as salary,
education, age, region, or contract type help distinguish candidate categories.

```python
from cat2cat.dataclass import cat2cat_mappings
from cat2cat.datasets import load_occup, load_trans

occup = load_occup()
trans = load_trans()

new = occup.loc[occup.year == 2010, :].copy()
mappings = cat2cat_mappings(trans, "backward")
```

```python
from cat2cat import cat2cat_ml_run
from cat2cat.dataclass import cat2cat_ml
from sklearn.ensemble import RandomForestClassifier

ml = cat2cat_ml(
    data=new,
    cat_var="code",
    features=["salary", "age", "edu"],
    models=[RandomForestClassifier(n_estimators=50, random_state=1234)],
    on_fail="freq",
)

diagnostics = cat2cat_ml_run(mappings=mappings, ml=ml)
print(diagnostics)
```

Diagnostics include accuracy, Brier score, and mean P(true class). Accuracy only
checks the top predicted class; Brier score and mean P(true class) evaluate the
full probability vector, which is closer to how `cat2cat` uses ML weights.

## Reading Diagnostics

Accuracy is useful when you care about the most likely category. It does not
tell you whether the remaining probability mass is well calibrated.

Brier score is a bounded squared-error score for probabilities. Lower is better.
If ML has a Brier score similar to or worse than the naive baseline, the model is
not adding useful probability information.

Mean P(true class) is the average probability assigned to the correct category.
Higher is better. This metric is especially intuitive for `cat2cat`: if the true
category usually receives low probability, the resulting weights are weak even
when the top prediction is sometimes correct.

## Using Several Estimators

Pass any scikit-learn classifiers that implement `predict_proba()`.

```python
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB

ml = cat2cat_ml(
    data=new,
    cat_var="code",
    features=["salary", "age", "edu"],
    models=[
        RandomForestClassifier(n_estimators=50, random_state=1234),
        LinearDiscriminantAnalysis(),
        GaussianNB(),
    ],
)
```

Naive Bayes is available through scikit-learn like any other estimator; it is not
a special string method in the Python API.

## Failed ML Weights

If ML fails for some rows, `on_fail` controls the behavior: `"freq"`, `"naive"`,
`"na"`, or `"error"`. Use `"error"` for strict diagnostics and `"freq"` for a
conservative production default.

```python
cat2cat_ml(
    data=new,
    cat_var="code",
    features=["salary", "age", "edu"],
    models=[RandomForestClassifier(n_estimators=50, random_state=1234)],
    on_fail="error",
    fail_warn=True,
)
```

Use `"na"` when you want to inspect missing ML weights manually. Use `"naive"`
when you want a neutral fallback that ignores base frequencies.