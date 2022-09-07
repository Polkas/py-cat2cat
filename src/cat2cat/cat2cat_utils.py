from pandas import DataFrame
from numpy import arange, concatenate, ndarray
from typing import Optional, Callable, Sequence

__all__ = ["prune_c2c", "dummy_c2c"]


def prune_c2c(
    df: DataFrame,
    prune_fun: Callable[[ndarray], ndarray],
    wei_var: str = "wei_freq_c2c",
    index_var: str = "index_c2c",
    inplace: bool = False,
) -> DataFrame:
    """Pruning which could be useful after the mapping process

    Args:
        df (DataFrame): a specific period from the cat2cat function result.
        prune_fun (callable): a function to process a 1D-array of weights (float) and return a 1D-array of boolean of the same length.
                              The weighs will be reweighted automatically to still to sum to one per each original observation.
        wei_var (str): By default "wei_freq_c2c".
        index_var (str): By default "index_c2c".
        inplace (bool): Whether to perform the operation inplace. By default False.
    Returns:
        DataFrame: df argument with possibly reduced number of rows.
    Note:
        - non-zero prune_fun - lambda x: x > 0
        - highest1 prune_fun - lambda x: arange(len(x)) == argmax(x)
        - highest prune_fun - lambda x: x == max(x)

    >>> from cat2cat import cat2cat
    >>> from cat2cat.dataclass import cat2cat_data, cat2cat_mappings, cat2cat_ml
    >>> from sklearn.ensemble import RandomForestClassifier
    >>> from cat2cat.datasets import load_trans, load_occup
    >>> trans = load_trans()
    >>> occup = load_occup()
    >>> o_old = occup.loc[occup.year == 2008, :].copy()
    >>> o_new = occup.loc[occup.year == 2010, :].copy()
    >>> data_c2c = cat2cat_data(o_old, o_new, "code", "code", "year")
    >>> mappings_c2c = cat2cat_mappings(trans, "forward")
    >>> c2c = cat2cat(data_c2c, mappings_c2c)
    >>> #
    >>> # non-zero - lambda x: x > 0
    >>> # highest1 - lambda x: arange(len(x)) == argmax(x)
    >>> # highest - lambda x: x == max(x)
    >>> #
    >>> # non-zero
    >>> prune_c2c(c2c["old"], lambda x: x > 0)
              id        age    sex  edu        exp  ...  index_c2c  g_new_c2c  rep_c2c wei_naive_c2c  wei_freq_c2c
    ...
    """
    assert isinstance(index_var, str), "index argument has to be a str"
    assert isinstance(wei_var, str), "col argument has to be a str"
    assert isinstance(df, DataFrame) and (
        (index_var in df.columns) and (wei_var in df.columns)
    ), "df argument has to be a DataFrame with the index and col args columns"
    assert callable(prune_fun), "prune_fun argument has to be a callable"
    assert isinstance(inplace, bool), "inplace argument has to be a bool"

    df2 = df if inplace else df.copy()
    final_rows = (
        df2.groupby(index_var, sort=False)
        .apply(lambda x: prune_fun(x[wei_var].values))
        .values
    )
    df2 = df2.loc[concatenate(final_rows)]
    # reweight
    df2[wei_var] = (
        df2.groupby(index_var, sort=False)[wei_var].apply(lambda x: x / sum(x)).values
    )
    return df2


def dummy_c2c(
    df: DataFrame,
    cat_var: str,
    models: Optional[Sequence] = None,
    inplace: bool = False,
) -> DataFrame:
    """Add default cat2cat columns to a `data.frame`

    The function is useful to achive consitent columns across all panel periods,
    even for ones for which cat2cat procedure was not applied.

    Args:
        df (DataFrame): a specific period from the cat2cat function result.
        cat_car (str): name of categorial variable
        models (Optional[Sequence]): an optional list of str, ml models applied (class name).
                                 By default turn off, equal None.
        inplace (bool): Whether to perform the operation inplace. By default False.

    Returns:
        DataFrame: df arg DataFrame but with additional columns connected with cat2cat procedure.
        The base added columns if not already exist: index_c2c, g_new_c2c, rep_c2c, wei_naive_c2c, wei_freq_c2c.
        Additionaly ml models connected columns like wei_MLNAME_c2c.
    """
    assert isinstance(cat_var, str), "cat_var argument has to be a str"
    assert isinstance(df, DataFrame) and (
        (cat_var in df.columns)
    ), "df argument has to be a DataFrame with the cat_var column"
    assert (models == None) or isinstance(
        models, Sequence
    ), "models has to be None or list-like of str (ml models names)"
    assert isinstance(inplace, bool), "inplace argument has to be a bool"

    df2 = df if inplace else df.copy()
    nrow_df = df2.shape[0]

    if "index_c2c" not in df2.columns:
        df2["index_c2c"] = arange(nrow_df)
    if "g_new_c2c" not in df2.columns:
        df2["g_new_c2c"] = df2[cat_var]
    if "rep_c2c" not in df2.columns:
        df2["rep_c2c"] = 1
    if "wei_naive_c2c" not in df2.columns:
        df2["wei_naive_c2c"] = 1
    if "wei_freq_c2c" not in df2.columns:
        df2["wei_freq_c2c"] = 1

    if models is not None:
        for m in models:
            ml_col = "wei_" + m + "_c2c"
            if ml_col not in df2.columns:
                df2[ml_col] = 1

    return df2
