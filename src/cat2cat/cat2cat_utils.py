from pandas import DataFrame
from numpy import arange
from typing import Optional


def prune_c2c(df: DataFrame, inplace: bool = False) -> DataFrame:
    """Pruning which could be useful after the mapping process

    Args:
        df (DataFrame): a specific period from the cat2cat function result.
        inplace (bool): Whether to perform the operation inplace. By default False.
    Returns:
        DataFrame: df argument with possibly reduced number of rows.
    """
    return DataFrame()


def cross_c2c(df: DataFrame, inplace: bool = False) -> DataFrame:
    """Make a combination of weights from different methods

    Args:
        df (DataFrame): a specific period from the cat2cat function result.
        inplace (bool): Whether to perform the operation inplace. By default False.

    Returns:
        DataFrame: df argument with additional column wei_cross_c2c which is a combination of other weights.
    """
    df2 = df if inplace else df.copy()

    return DataFrame()


def dummy_c2c(
    df: DataFrame, cat_var: str, models: Optional[list] = None, inplace: bool = False
) -> DataFrame:
    """Add default cat2cat columns to a `data.frame`

    The function is useful to achive consitent columns across all panel periods,
    even for ones for which cat2cat procedure was not applied.

    Args:
        df (DataFrame): a specific period from the cat2cat function result.
        cat_car (str): name of categorial variable
        models (Optional[list]): an optional list of str, ml models applied (class name).
        By default turn off, equal None.
        inplace (bool): Whether to perform the operation inplace. By default False.

    Returns:
        DataFrame: df arg DataFrame but with additional columns connected with cat2cat procedure.
        The base added columns if not already exist: index_c2c, g_new_c2c, rep_c2c, wei_naive_c2c, wei_freq_c2c.
        Additionaly ml models connected columns like wei_MLNAME_c2c.
    """

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
