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

o_old = occup.loc[occup.year == 2008, :].copy()
o_new = occup.loc[occup.year == 2010, :].copy()

data = cat2cat_data(old = o_old, new = o_new, "code", "code", "year")
mappings = cat2cat_mappings(trans, "backward")

c2c = cat2cat(data, mappings)
data_final = concat([c2c["old"], c2c["new"]])

sub_cols = ["id", "edu", "code", "year", "index_c2c",
 "g_new_c2c", "rep_c2c", "wei_naive_c2c", "wei_freq_c2c"]
data_final.groupby(["year"]).sample(5).loc[:, sub_cols]
```

with ml models:

```python
from sklearn.neighbors import KNeighborsClassifier

ml = cat2cat_ml(
    occup.loc[occup.year >= 2010, :].copy(), 
    "code", 
    ["salary", "age", "edu"], 
    [KNeighborsClassifier()]
)

c2c = cat2cat(data, mappings, ml)
data_final = concat([c2c["old"], c2c["new"]])
```

with 4 periods , one mapping table and backward direction:

```
from cat2cat.cat2cat_utils import dummy_c2c

o_2006 = occup.loc[occup.year == 2006, :].copy()
o_2008 = occup.loc[occup.year == 2008, :].copy()
o_2010 = occup.loc[occup.year == 2010, :].copy()
o_2012 = occup.loc[occup.year == 2012, :].copy()


data = cat2cat_data(o_2008, o_2010, "code", "code", "year")
mappings = cat2cat_mappings(trans, "backward")

occup_back_2008_2010 = cat2cat(data, mappings)
data = cat2cat_data(
    o_2006, occup_back_2008_2010["old"], 
    "code", "g_new_c2c", "year"
)
occup_back_2006_2008 = cat2cat(data, mappings)

o_2006_n = occup_back_2006_2008["old"]
o_2008_n = occup_back_2006_2008["new"] # or occup_back_2008_2010["old"]
o_2010_n = occup_back_2008_2010["new"]
o_2012_n = dummy_c2c(o_2012, "code")

data_final = concat([o_2006_n, o_2008_n, o_2010_n, o_2012_n])

sub_cols = ["id", "edu", "code", "year", "index_c2c",
 "g_new_c2c", "rep_c2c", "wei_naive_c2c", "wei_freq_c2c"]
data_final.groupby(["year"]).sample(5).loc[:, sub_cols]
```

## Contributing

Interested in contributing? Check out the contributing guidelines. Please note that this project is released with a Code of Conduct. By contributing to this project, you agree to abide by its terms.

## License

`cat2cat` was created by Maciej Nasinski. It is licensed under the terms of the MIT license.

## Credits

`cat2cat` was created with [`cookiecutter`](https://cookiecutter.readthedocs.io/en/latest/) and the `py-pkgs-cookiecutter` [template](https://github.com/py-pkgs/py-pkgs-cookiecutter).
