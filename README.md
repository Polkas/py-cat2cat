# cat2cat 

<a href='https://github.com/polkas/py-cat2cat'>
<img src='https://raw.githubusercontent.com/Polkas/cat2cat/master/man/figures/cat2cat_logo.png'  style="display:block;margin-left:auto;margin-right:auto;width:200px;" width="200px" alt="cat2cat logo"/>
</a>

<hr>

<div>
<a href="https://github.com/polkas/py-cat2cat/actions">
<img src="https://github.com/polkas/py-cat2cat/workflows/ci/badge.svg" alt="Build Status">
</a>
<a href="https://codecov.io/gh/Polkas/py-cat2cat">
<img src="https://codecov.io/gh/Polkas/py-cat2cat/branch/main/graph/badge.svg" alt="codecov">
</a>
<a href="https://pypi.org/project/cat2cat/">
<img src="https://img.shields.io/pypi/v/cat2cat.svg" alt="pypi">
</a>
<div>

<br>

### Unifying an inconsistently coded categorical variable in a panel/longtitudal dataset

There is offered the cat2cat procedure to map a categorical variable according to a mapping (transition) table between two different time points. The mapping (transition) table should to have a candidate for each category from the targeted for an update period. The main rule is to replicate the observation if it could be assigned to a few categories, then using simple frequencies or statistical methods to approximate probabilities of being assigned to each of them.

This algorithm was invented and implemented in the paper by (Nasinski, Majchrowska and Broniatowska (2020) doi:10.24425/cejeme.2020.134747).

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
from cat2cat.mappings import get_mappings, get_freqs, cat_apply_freq

# convert the mapping table to two association lists
mappings = get_mappings(trans)
# get a variable levels freqencies
codes_new = occup.code[occup.year == 2010].values
freqs = get_freqs(codes_new)
# apply the frequencies to the (one) association list
mapp_new_p = cat_apply_freq(mappings["to_new"], freqs)

# mappings for a specific category
mappings["to_new"]['3481']
# probability mappings for a specific category
mapp_new_p['3481']
```

### cat2cat function

```python
from cat2cat import cat2cat
from cat2cat.dataclass import cat2cat_data, cat2cat_mappings, cat2cat_ml

from pandas import concat

# split the panel by the time variale
# here only two periods
o_old = occup.loc[occup.year == 2008, :].copy()
o_new = occup.loc[occup.year == 2010, :].copy()

# dataclasses, core arguments for the cat2cat function
data = cat2cat_data(
    old = o_old, 
    new = o_new,
    cat_var_old = "code", 
    cat_var_new = "code", 
    time_var = "year"
)
mappings = cat2cat_mappings(trans = trans, direction = "backward")

# apply the cat2cat procedure
c2c = cat2cat(data = data, mappings = mappings)
# pandas.concat used to bind per period datasets
data_final = concat([c2c["old"], c2c["new"]])
```

## Contributing

Interested in contributing? Check out the contributing guidelines. Please note that this project is released with a Code of Conduct. By contributing to this project, you agree to abide by its terms.

## License

`cat2cat` was created by Maciej Nasinski. It is licensed under the terms of the MIT license.

## Credits

`cat2cat` was created with [`cookiecutter`](https://cookiecutter.readthedocs.io/en/latest/) and the `py-pkgs-cookiecutter` [template](https://github.com/py-pkgs/py-pkgs-cookiecutter).
