# cat2cat

## About

Unifying an inconsistent coded categorical variable in a panel/longtitudal dataset

There is offered the cat2cat procedure to map a categorical variable according to a mapping (transition) table between two different time points. The mapping (transition) table should to have a candidate for each category from the targeted for an update period. The main rule is to replicate the observation if it could be assigned to a few categories, then using simple frequencies or statistical methods to approximate probabilities of being assigned to each of them.

**This algorithm was invented and implemented in the paper by [(Nasinski, Majchrowska and Broniatowska (2020))](https://doi.org/10.24425/cejeme.2020.134747).**

**For more details please read the paper by [(Nasinski, Gajowniczek (2023))](https://doi.org/10.1016/j.softx.2023.101525).**


## Graph - cat2cat procedure

The graphs present how the `cat2cat` function (and the underlying procedure) works, in this case under a panel dataset without the unique identifiers and only two periods.

![Backward Mapping](https://raw.githubusercontent.com/Polkas/cat2cat/master/man/figures/back_nom.png)

![Forward Mapping](https://raw.githubusercontent.com/Polkas/cat2cat/master/man/figures/for_nom.png)


## Example usage

To use `cat2cat` in a project:

### Load example data

```python
# cat2cat datasets
from cat2cat.datasets import load_trans, load_occup, load_verticals
from numpy.random import seed

seed(1234)

trans = load_trans()
occup = load_occup()
verticals = load_verticals()
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
print(mappings["to_new"]['3481'])
# probability mappings for a specific category
print(mapp_new_p['3481'])
```

### cat2cat procedure - one iteration

```python
from cat2cat import cat2cat
from cat2cat.dataclass import cat2cat_data, cat2cat_mappings, cat2cat_ml

from pandas import concat

# split the panel by the time variale
# here only two periods
o_old = occup.loc[occup.year == 2008, :].copy()
o_new = occup.loc[occup.year == 2010, :].copy()

# dataclasses, two core arguments for the cat2cat function
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

sub_cols = ["id", "edu", "code", "year", "index_c2c", "g_new_c2c", "rep_c2c", "wei_naive_c2c", "wei_freq_c2c"]
data_final.groupby(["year"]).sample(5).loc[:, sub_cols]
```

### With ML

```python
from sklearn.neighbors import KNeighborsClassifier
from cat2cat import cat2cat_ml_run

# ml dataclass, one of the arguments for the cat2cat function
ml = cat2cat_ml(
    data = o_new, 
    cat_var = "code", 
    features = ["salary", "age", "edu"], 
    models = [KNeighborsClassifier(random_state = 1234)]
)

cat2cat_ml_run(mappings, ml)

# apply the cat2cat procedure
c2c = cat2cat(data = data, mappings = mappings, ml = ml)
# pandas.concat used to bind per period datasets
data_final = concat([c2c["old"], c2c["new"]])

sub_cols = ["id", "year", "wei_naive_c2c", "wei_freq_c2c", "wei_KNeighborsClassifier_c2c"]
data_final.groupby(["year"]).sample(3).loc[:, sub_cols]
```

With 4 periods, one mapping table and backward direction:

```python
from cat2cat.cat2cat_utils import dummy_c2c

# split the panel by the time variale
# here four periods
o_2006 = occup.loc[occup.year == 2006, :].copy()
o_2008 = occup.loc[occup.year == 2008, :].copy()
o_2010 = occup.loc[occup.year == 2010, :].copy()
o_2012 = occup.loc[occup.year == 2012, :].copy()

# dataclasses, two core arguments for the cat2cat function
data = cat2cat_data(
    old = o_2008, 
    new = o_2010, 
    cat_var_old = "code", 
    cat_var_new = "code", 
    time_var = "year"
)
mappings = cat2cat_mappings(trans = trans, direction = "backward")

# apply the cat2cat procedure
occup_back_2008_2010 = cat2cat(data = data, mappings = mappings)

# updated for the next iteration data cat2cat argument
data = cat2cat_data(
    old = o_2006, 
    new = occup_back_2008_2010["old"], 
    cat_var_old = "code", 
    cat_var_new = "g_new_c2c", 
    time_var = "year"
)

# apply the cat2cat procedure
occup_back_2006_2008 = cat2cat(data = data, mappings = mappings)

# gather the datasets for each period
o_2006_n = occup_back_2006_2008["old"]
o_2008_n = occup_back_2006_2008["new"] # or occup_back_2008_2010["old"]
o_2010_n = occup_back_2008_2010["new"]
o_2012_n = dummy_c2c(o_2012, "code")

# pandas.concat used to bind per period datasets
data_final = concat([o_2006_n, o_2008_n, o_2010_n, o_2012_n])

sub_cols = ["id", "edu", "code", "year", "index_c2c",
 "g_new_c2c", "rep_c2c", "wei_naive_c2c", "wei_freq_c2c"]
data_final.groupby(["year"]).sample(2).loc[:, sub_cols]
```

### Prune - prune_c2c


Pruning which could be useful after the mapping process, the custom prune_fun is provided by the end user.
The prune_fun is a function to process a 1D-array of weights (float) and return a 1D-array of boolean of the same length. The weighs will be reweighted automatically to still to sum to one per each original observation.

- non-zero - lambda x: x > 0
- highest1 - lambda x: arange(len(x)) == argmax(x)
- highest - lambda x: x == max(x)

```python
from cat2cat.cat2cat_utils import prune_c2c
from numpy import arange, argmax

# prune_c2c
# highest1 leave only one observation with the highest probability for each orginal one
(o_2006_n.shape[0], 
 prune_c2c(o_2006_n, lambda x: arange(len(x)) == argmax(x)).shape[0])
```

### Direct match


It is important to set the `id_var` argument as then we merging categories 1 to 1
for this identifier which exists in both periods.

```python
# split the panel by the time variable
vert_old = verticals.loc[verticals["v_date"] == "2020-04-01", :]
vert_new = verticals.loc[verticals["v_date"] == "2020-05-01", :]

## extract mapping (transition) table from data using identifier
trans_v = vert_old.merge(vert_new, on = "ean", how = "inner")\
.loc[:, ["vertical_x", "vertical_y"]]\
.drop_duplicates()
```

```python
# dataclasses, two core arguments for the cat2cat function
data = cat2cat_data(
  old = vert_old, 
  new = vert_new, 
  id_var = "ean", 
  cat_var_old = "vertical", 
  cat_var_new = "vertical", 
  time_var = "v_date"
)
mappings = cat2cat_mappings(trans = trans_v, direction = "backward")

# apply the cat2cat procedure
verts = cat2cat(
  data = data,
  mappings = mappings
)

# pandas.concat used to bind per period datasets
data_final = concat([verts["old"], verts["new"]])
```

### Direct match with ML

```python
# ml dataclass, one of the arguments for the cat2cat function
ml = cat2cat_ml(
    data = vert_old, 
    cat_var = "vertical", 
    features = ["sales"], 
    models = [KNeighborsClassifier()]
)

# apply the cat2cat procedure
verts_ml = cat2cat(
  data = data,
  mappings = mappings,
  ml = ml
)

# pandas.concat used to bind per period datasets
data_final = concat([verts_ml["old"], verts_ml["new"]])
```

