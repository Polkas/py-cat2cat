# Advanced Workflows

This guide collects workflows that go beyond a single two-period mapping:
categorical ML features, direct matching, multi-period chains, and regression
after harmonisation.

```python
from cat2cat import cat2cat
from cat2cat.dataclass import cat2cat_data, cat2cat_mappings
from cat2cat.datasets import load_occup, load_trans

occup = load_occup()
trans = load_trans()

old = occup.loc[occup.year == 2008, :].copy()
new = occup.loc[occup.year == 2010, :].copy()

old_2006 = occup.loc[occup.year == 2006, :].copy()
old_2008 = old.copy()
new_2010 = new.copy()
```

## Categorical ML Features

Numeric and boolean ML features are used directly. Categorical/object/string
features are one-hot encoded using the union of levels observed in `ml.data` and
the target period.

```python
ml_data = new.copy()
ml_data["edu_group"] = ml_data["edu"].astype(str)
old["edu_group"] = old["edu"].astype(str)
new["edu_group"] = new["edu"].astype(str)
```

Then include the categorical feature in `cat2cat_ml.features` as usual.

```python
from sklearn.ensemble import RandomForestClassifier
from cat2cat.dataclass import cat2cat_ml

ml = cat2cat_ml(
    data=ml_data,
    cat_var="code",
    features=["salary", "age", "edu_group"],
    models=[RandomForestClassifier(n_estimators=50, random_state=1234)],
)
```

The generated indicator columns are internal. They are built consistently across
the training and target data so unseen target levels are still represented.

## Direct Matching

When the same subject identifier appears in both periods, pass `id_var` to map
those subjects directly and avoid unnecessary replication.

```python
data = cat2cat_data(
    old=old,
    new=new,
    cat_var_old="code",
    cat_var_new="code",
    time_var="year",
    id_var="id",
)
```

Direct matching assumes the subject's true category does not change between the
adjacent waves being linked. This is reasonable for short rotational panels when
the coding change is the main source of inconsistency, but it is not appropriate
when real category transitions are expected.

The package includes `load_occup_panel()` for this workflow.

```python
from cat2cat.datasets import load_occup_panel, load_trans

panel = load_occup_panel()
trans = load_trans()

old = panel.loc[panel.quarter == "2009Q4", :].copy()
new = panel.loc[panel.quarter == "2010Q1", :].copy()
```

## Multi-Period Chaining

For more than two periods, apply `cat2cat()` one transition at a time. The
harmonised column from one step can become the category input for the next step.

```python
first = cat2cat(
    cat2cat_data(old_2008, new_2010, "code", "code", "year"),
    cat2cat_mappings(trans, "backward"),
)

second = cat2cat(
    cat2cat_data(old_2006, first["old"], "code", "g_new_c2c", "year"),
    cat2cat_mappings(trans, "backward"),
)
```

Check row counts and weight sums after every step. Replication can compound when
several periods are chained.

## Regression After Harmonisation

Use `summary_c2c()` with statsmodels result objects to adjust standard errors
after fitting on replicated data.

```python
import importlib.util
from cat2cat import summary_c2c

if importlib.util.find_spec("statsmodels") is None:
    print("Install optional dependency: pip install cat2cat[summary]")
else:
    import statsmodels.api as sm

    data_rep = first["old"]
    model = sm.WLS.from_formula(
        "salary ~ age + sex + edu",
        data=data_rep,
        weights=data_rep["wei_freq_c2c"],
    ).fit()

    summary_c2c(model, df_old=len(old_2008) - len(model.params))
```

The correction factor is based on the ratio of replicated residual degrees of
freedom to original-scale residual degrees of freedom. Ordinary fit statistics
should still be interpreted with care when the harmonised category enters the
model.