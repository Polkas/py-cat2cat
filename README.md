# cat2cat
[![Build Status](https://github.com/polkas/py-cat2cat/workflows/ci-cd/badge.svg)](https://github.com/polkas/py-cat2cat/actions)
[![codecov](https://codecov.io/gh/Polkas/py-cat2cat/branch/main/graph/badge.svg)](https://codecov.io/gh/Polkas/py-cat2cat)

Unifying an inconsistent coded categorical variable in a panel/longtitudal dataset.

## Installation

```bash
$ pip install cat2cat
```

## Usage

### load data

```python
# cat2cat datasets
from cat2cat.datasets import load_trans, load_occup
trans = load_trans()
occup = load_occup()
```

### Low-level functions

```python
# Low-level functions
from cat2cat.mappings import get_mappings, get_freqs, cat_apply_freq

mappings = get_mappings(trans)
codes_new = occup.code[occup.year == 2010].values
freqs = get_freqs(codes_new)
mapp_new_p = cat_apply_freq(mappings["to_new"], freqs)
mappings["to_new"]['3481']
mapp_new_p['3481']
```

### cat2cat function

```python
from cat2cat import cat2cat
from cat2cat.dataclass import cat2cat_data, cat2cat_mappings, cat2cat_ml

from pandas import DataFrame
from sklearn.ensemble import RandomForestClassifier

o_2006 = occup.loc[occup.year == 2006, :].copy()
o_2008 = o_old = occup.loc[occup.year == 2008, :].copy()
o_2010 = o_new = occup.loc[occup.year == 2010, :].copy()
o_2012 = occup.loc[occup.year == 2012, :].copy()

data = cat2cat_data(old = o_old, new = o_new, "code", "code", "year")
mappings = cat2cat_mappings(trans, "backward")
ml = cat2cat_ml(o_new, "code", ["salary", "age"], [RandomForestClassifier()])

res = cat2cat(data, mappings, ml)
data_final = concat([res["old"], res["new"]])
sub_cols = [0,1,2,8,10,12,13,14,15,16]
data_final.groupby(["year"]).sample(5).iloc[:, sub_cols]
```

## Contributing

Interested in contributing? Check out the contributing guidelines. Please note that this project is released with a Code of Conduct. By contributing to this project, you agree to abide by its terms.

## License

`cat2cat` was created by Maciej Nasinski. It is licensed under the terms of the MIT license.

## Credits

`cat2cat` was created with [`cookiecutter`](https://cookiecutter.readthedocs.io/en/latest/) and the `py-pkgs-cookiecutter` [template](https://github.com/py-pkgs/py-pkgs-cookiecutter).
