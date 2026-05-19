# Get Started

`cat2cat` harmonises a categorical variable when the category system changes
between two periods. The core idea is simple: if one target-period category can
map to several base-period categories, the observation is replicated once for
each candidate and receives probability weights.

Start with a two-period harmonisation. Split the data into an old and new period,
create `cat2cat_data` and `cat2cat_mappings`, then call `cat2cat()`.

```python
from pandas import concat
from cat2cat import cat2cat
from cat2cat.dataclass import cat2cat_data, cat2cat_mappings
from cat2cat.datasets import load_occup, load_trans

occup = load_occup()
trans = load_trans()

old = occup.loc[occup.year == 2008, :].copy()
new = occup.loc[occup.year == 2010, :].copy()

data = cat2cat_data(old, new, "code", "code", "year")
mappings = cat2cat_mappings(trans, "backward")

res = cat2cat(data=data, mappings=mappings)
harmonised = concat([res["old"], res["new"]])
```

The replicated period receives `index_c2c`, `g_new_c2c`, `rep_c2c`,
`wei_naive_c2c`, and `wei_freq_c2c`. Weights are probabilities and should sum to
one per original observation.

## Direction

The mapping table has two columns: the first is the old encoding and the second
is the new encoding. Direction controls which period is harmonised:

- `"backward"`: map the old period into the new coding system.
- `"forward"`: map the new period back into the old coding system.

Choose the direction that matches the category system you want in the final
dataset. If you want all periods expressed in the newest coding system, use
`"backward"` step by step from older periods toward newer periods.

## Inspecting The Result

The output is a dictionary with `"old"` and `"new"` data frames. One of them is
replicated and weighted; the other receives dummy cat2cat columns with weight 1.

```python
target = res["old"]

target.groupby("index_c2c")["wei_freq_c2c"].sum().round(10).head()
target[["code", "g_new_c2c", "rep_c2c", "wei_freq_c2c"]].head()
```

Use `g_new_c2c` as the harmonised category and a `wei_*_c2c` column as the
probability weight. `wei_freq_c2c` is the usual transparent baseline because it
uses observed category frequencies in the base period.

## Typical Two-Period Workflow

1. Check that `trans` covers the categories in the period being harmonised.
2. Run `cat2cat()` without ML and inspect row counts and weight sums.
3. Use `wei_freq_c2c` for descriptive tables or regression weights.
4. Add ML only after validating that its probability weights improve on simple
   baselines.

## Keeping Multiple Periods Together

Apply `cat2cat()` iteratively to neighbouring periods. After each step, the
harmonised category is stored in `g_new_c2c`, so later steps can use it as the
category variable for the already-harmonised period.

```python
data_next = cat2cat_data(
	old=occup.loc[occup.year == 2006, :].copy(),
	new=res["old"],
	cat_var_old="code",
	cat_var_new="g_new_c2c",
	time_var="year",
)
```

For long chains, replicated rows can grow quickly. Consider pruning or using
direct matching with `id_var` when identifiers are available.