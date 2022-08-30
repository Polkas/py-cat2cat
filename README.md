# cat2cat 
<a href='https://github.com/polkas/py-cat2cat'><img src='https://raw.githubusercontent.com/Polkas/cat2cat/master/man/figures/cat2cat_logo.png' align="right" width="200px" style="margin:5px;"/></a>
[![Build Status](https://github.com/polkas/py-cat2cat/workflows/ci-cd/badge.svg)](https://github.com/polkas/py-cat2cat/actions)
[![codecov](https://codecov.io/gh/Polkas/py-cat2cat/branch/main/graph/badge.svg)](https://codecov.io/gh/Polkas/py-cat2cat)

Unifying an inconsistently coded categorical variable in a panel/longtitudal dataset.

## Installation

```bash
$ pip install cat2cat
```

## Usage

For more examples and descriptions please vist [**the example notebook**](https://py-cat2cat.readthedocs.io/en/latest/example.html)

### load example data

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

# dataclasses a core arguments for cat2cat function
data = cat2cat_data(old = o_old, new = o_new, "code", "code", "year")
mappings = cat2cat_mappings(trans, "backward")

c2c = cat2cat(data, mappings)
data_final = concat([c2c["old"], c2c["new"]])
```

## Contributing

Interested in contributing? Check out the contributing guidelines. Please note that this project is released with a Code of Conduct. By contributing to this project, you agree to abide by its terms.

## License

`cat2cat` was created by Maciej Nasinski. It is licensed under the terms of the MIT license.

## Credits

`cat2cat` was created with [`cookiecutter`](https://cookiecutter.readthedocs.io/en/latest/) and the `py-pkgs-cookiecutter` [template](https://github.com/py-pkgs/py-pkgs-cookiecutter).
