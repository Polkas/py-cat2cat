# cat2cat

[![cat2cat logo](https://raw.githubusercontent.com/Polkas/cat2cat/master/man/figures/cat2cat_logo.png)](https://github.com/polkas/py-cat2cat)

[![Python build status](https://github.com/polkas/py-cat2cat/workflows/ci/badge.svg)](https://github.com/polkas/py-cat2cat/actions)
[![PyPI](https://img.shields.io/pypi/v/cat2cat.svg)](https://pypi.org/project/cat2cat/)
[![codecov](https://codecov.io/gh/Polkas/py-cat2cat/branch/main/graph/badge.svg)](https://app.codecov.io/gh/Polkas/py-cat2cat)

## Handling an Inconsistent Coded Categorical Variable in a Longitudinal Dataset

**cat2cat** provides a statistical solution for harmonising categorical variables whose encoding changes between survey waves or data releases.
If you work with longitudinal data where classification schemes evolve (occupations, diseases, industries, products, or fields of education), this package helps produce valid cross-temporal analyses.

### The Problem

Real-world classifications change.
When one classification replaces another, a single old code may map to multiple new codes (and vice versa).
Naive responses are unsatisfactory: separate analyses by period limit comparability, manual recoding is hard to reproduce, and ignoring coding changes can bias results.

### The Solution

**cat2cat** maps a categorical variable using a transition table between two time points.
The transition table should list candidate categories for each code in the period being harmonised.
When one observed code corresponds to several target categories, **cat2cat** replicates the observation across candidates and assigns probability weights using frequencies or ML-based predictions.

The method follows a replication-and-weighting procedure that:

1. Replicates each observation onto all candidate categories from the mapping table for a chosen direction.
2. Assigns probability weights that sum to 1 per original observation.
3. Preserves distributional properties of non-mapped variables for valid downstream analysis.

NOTE: If you have a fully linked panel where each subject appears in both periods and target-period categories are directly available, probabilistic harmonisation may be unnecessary. `cat2cat()` is most useful when direct linking is incomplete (for example repeated cross-sections, rotational panels, entrants/leavers).

### Value Added of cat2cat

cat2cat separates true structural change from coding-system change.

After harmonisation, you can:

- Track trends within groups across waves.
- Compare subgroup dynamics on one consistent coding scheme.
- Estimate models with group effects/interactions.
- Run sensitivity checks across weighting assumptions.

### Direction

You can harmonise in both directions:

#### Forward Mapping (Old -> New)

![Forward mapping](https://raw.githubusercontent.com/Polkas/cat2cat/master/man/figures/for_nom.png)

#### Backward Mapping (New -> Old)

![Backward mapping](https://raw.githubusercontent.com/Polkas/cat2cat/master/man/figures/back_nom.png)

### Key Features

| Feature | Benefit |
| ------- | ------- |
| Probability weights | Naive, frequency, and ML-based weights |
| ML validation | `cat2cat_ml_run()` reports accuracy, Brier, and mean P(true class) |
| Multi-period chaining | Harmonise 3+ waves iteratively |
| Regression support | `summary_c2c()` adjusts inference for replicated-data workflows |
| Aggregated workflows | Harmonisation tools for grouped data use-cases |

### References

- **Method**: [Nasinski, Majchrowska & Broniatowska (2020)](https://doi.org/10.24425/cejeme.2020.134747)
- **Software**: [Nasinski & Gajowniczek (2023)](https://doi.org/10.1016/j.softx.2023.101525)

### Ecosystem

| | |
| --- | --- |
| [**R Package**](https://cran.r-project.org/package=cat2cat) | CRAN, production-ready |
| [**Python Package**](https://pypi.org/project/cat2cat/) | PyPI |
| [**Documentation**](https://py-cat2cat.readthedocs.io/en/latest/) | API and guides |

## Documentation

- [Get Started](https://py-cat2cat.readthedocs.io/en/latest/get-started.html) - core concepts and a two-period workflow.
- [Choosing Weights and Validating ML](https://py-cat2cat.readthedocs.io/en/latest/choosing-weights-and-validating-ml.html) - weight strategy and ML validation.
- [Advanced Workflows](https://py-cat2cat.readthedocs.io/en/latest/advanced-workflows.html) - multi-period and advanced usage patterns.

## Installation

```bash
pip install cat2cat
```

## Quick Start

```python
from pandas import concat
from cat2cat import cat2cat, cat2cat_ml_run
from cat2cat.dataclass import cat2cat_data, cat2cat_mappings, cat2cat_ml
from cat2cat.datasets import load_occup, load_trans
from sklearn.ensemble import RandomForestClassifier

occup = load_occup()
trans = load_trans()

old = occup.loc[occup.year == 2008, :].copy()
new = occup.loc[occup.year == 2010, :].copy()

data = cat2cat_data(old=old, new=new, cat_var_old="code", cat_var_new="code", time_var="year")
mappings = cat2cat_mappings(trans=trans, direction="backward")

c2c = cat2cat(data=data, mappings=mappings)
harmonised = concat([c2c["old"], c2c["new"]])

new["edu_group"] = new["edu"].astype(str)
old["edu_group"] = old["edu"].astype(str)
ml = cat2cat_ml(
    data=new,
    cat_var="code",
    features=["salary", "age", "edu_group"],
    models=[RandomForestClassifier(n_estimators=50, random_state=1234)],
    on_fail="freq",
    fail_warn=True,
)

diagnostics = cat2cat_ml_run(mappings=mappings, ml=ml)
print(diagnostics)
```

## Citation

If you use cat2cat in your research, please cite:

```text
Nasinski M, Gajowniczek K (2023). "cat2cat: Handling an Inconsistently Coded
Categorical Variable in a Longitudinal Dataset." SoftwareX, 24, 101525.
doi:10.1016/j.softx.2023.101525
```

```bibtex
@article{nasinski2023cat2cat,
  title={cat2cat: Handling an Inconsistently Coded Categorical Variable in a Longitudinal Dataset},
  author={Nasinski, Maciej and Gajowniczek, Krzysztof},
  journal={SoftwareX},
  volume={24},
  pages={101525},
  year={2023},
  doi={10.1016/j.softx.2023.101525}
}
```

## Contributing

Interested in contributing? Check the contributing guidelines and code of conduct.

## License

`cat2cat` is licensed under Apache License 2.0. See [LICENSE](LICENSE).
