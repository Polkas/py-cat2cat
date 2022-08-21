from pandas import DataFrame
from numpy import arange


def prune_c2c(df: DataFrame) -> DataFrame:
    """Pruning which could be useful after the mapping process

    Args:
        df (DataFrame): _description_

    Returns:
        DataFrame: _description_
    """
    return DataFrame()


def cross_c2c(df: DataFrame) -> DataFrame:
    """Make a combination of weights from different methods

    Args:
        df (DataFrame): _description_

    Returns:
        DataFrame: _description_
    """
    return DataFrame()


def dummy_c2c(df: DataFrame, cat_var: str, models=None) -> DataFrame:
    """Add default cat2cat columns to a `data.frame`

    Args:
        df (DataFrame): _description_

    Returns:
        DataFrame: _description_
    """

    nrow_df = df.shape[0]

    df["index_c2c"] = arange(nrow_df)
    df["g_new_c2c"] = df[cat_var]
    df["rep_c2c"] = 1
    df["wei_naive_c2c"] = 1
    df["wei_freq_c2c"] = 1

    if models is not None:
        for m in models:
            df[m] = 1

    return df
