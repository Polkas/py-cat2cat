from pandas import DataFrame
from numpy import ndarray, unique, repeat, array, round, unique, sort, isnan

from collections.abc import Iterable
from collections import OrderedDict
from typing import Union, Optional, Any, List, Dict, Sequence, TypeVar

__all__ = ["get_mappings", "cat_apply_freq", "get_freqs"]

Table = TypeVar("Table", DataFrame, ndarray)

def get_mappings(x: Table) -> Dict[str, Dict[Any, List[Any]]]: 
    """Transforming a mapping table with mappings to two associative lists

    Transforming a transition table with mappings to two associative lists
    to rearrange the one classification encoding into another, an associative list that maps keys to values is used.
    More precisely, an association list is used which is a linked list in which each list element consists of a key and value or values.
    An association list where unique categories codes are keys and matching categories from next or previous time point are values.
    A transition table is used to build such associative lists.

    Args:
        x (pandas.DataFrame or numpy.ndarray): transition table with 2 columns where first column is assumed to be the older encoding.

    Returns:
        Dict[str, Dict[Any, List[Any]]]: dict with 2 internal dicts, `to_old` and `to_new`.
    Note:
        There was made an effort to handle missings properly but please try to avoid of using NaN or None.
        It is recommended to use string or float types.
        Alternative solution can be representing missing values as a specific number (9999) or string ("Missing").

    >>> from cat2cat.mappings import get_mappings
    >>> from numpy import array, nan
    >>> trans = array([
    ...   [1111, 111101], [1111, 111102], [1123, 111405], [nan, 111405],
    ...   [1212, 112006], [1212, 112008], [1212, 112090], [1212, nan],
    ... ])
    >>> mappings = get_mappings(trans)
    >>> mappings["to_old"]
    {111101.0: [1111.0], 111102.0: [1111.0], 111405.0: [1123.0, nan], 112006.0: [1212.0], 112008.0: [1212.0], 112090.0: [1212.0], nan: [1212.0]}
    >>> mappings["to_new"]
    {1111.0: [111101.0, 111102.0], 1123.0: [111405.0], 1212.0: [112006.0, 112008.0, 112090.0, nan], nan: [111405.0]}
    """
    assert hasattr(x, "shape") and ((len(x.shape) == 2) and (
        x.shape[1] == 2
    )), "x should have 2 dimensions and the second one is equal to 2 (columns)"

    if isinstance(x, DataFrame):
        return get_mappings_df(x)
    elif isinstance(x, ndarray):
        return get_mappings_array(x)
    else:
        raise TypeError("get_mappings input has to be ndarray or DataFrame")

def get_mappings_array(x: ndarray) -> Dict[str, Dict[Any, List[Any]]]:
    ff = x[:, 0].copy()
    ss = x[:, 1].copy()

    assert ff.dtype == ss.dtype
    col_type = ff.dtype

    from_old = list(OrderedDict.fromkeys(ff))
    from_new = list(OrderedDict.fromkeys(ss))

    to_old = dict()
    for e in from_new:
        if (col_type in [float, int]) and isnan(e):
            idx = isnan(ss)
        else:
            idx = ss == e

        # sorted so results are stable
        to_old[e] = sorted(unique(ff[idx]))

    to_new = dict()
    for e in from_old:
        if (col_type in [float, int]) and isnan(e):
            idx = isnan(ff)
        else:
            idx = ff == e

        # sorted so results are stable
        to_new[e] = sorted(unique(ss[idx]))

    return dict(to_old=to_old, to_new=to_new)

def get_mappings_df(x: DataFrame) -> Dict[str, Dict[Any, List[Any]]]:
    ff = x.iloc[:, 0].copy()
    which_ff_null = ff.isnull()
    ff = ff.values
    ss = x.iloc[:, 1].copy()
    which_ss_null = ss.isnull()
    ss = ss.values

    assert ff.dtype == ss.dtype
    col_type = ff.dtype

    if col_type == "O":
        ff[which_ff_null | (ff == None)] = "None"
        ss[which_ss_null | (ss == None)] = "None"

    from_old = list(OrderedDict.fromkeys(ff))
    from_new = list(OrderedDict.fromkeys(ss))

    to_old = dict()
    for e in from_new:
        if (col_type in [float, int]) and isnan(e):
            idx = isnan(ss)
        else:
            idx = ss == e

        # sorted so results are stable
        to_old[e] = sorted(unique(ff[idx]))

    to_new = dict()
    for e in from_old:
        if (col_type in [float, int]) and isnan(e):
            idx = isnan(ff)
        else:
            idx = ff == e

        # sorted so results are stable
        to_new[e] = sorted(unique(ss[idx]))

    return dict(to_old=to_old, to_new=to_new)

def get_freqs(
    x: Sequence[Any], multiplier: Optional[Sequence[int]] = None
) -> Dict[Any, int]:
    """
    Getting frequencies from a vector with an optional multiplier

    Args:
        x (Sequence[Any]): a list like, categorical variable to summarize.
        multiplier (Optional[Sequence[int]]): a list like, how many times to repeat certain value, additional weights.
                                         Have the same length as the x argument. Defaults to None.

    Returns:
        dict: with unique values and their counts

    >>> get_freqs([1,1,1,2,1,2,2,11])
    {1: 4, 2: 3, 11: 1}
    """
    assert isinstance(x, Iterable), "x has to be at least a Iterable"
    assert (multiplier is None) or isinstance(
        multiplier, Iterable
    ), "multiplier has to be a Iterable"
    input: ndarray
    if multiplier is not None:
        input = repeat(x, multiplier)
    else:
        input = array(x)
    input_unique: tuple = unique(input, return_counts=True)
    res: dict = dict(zip(*input_unique))
    return res


def cat_apply_freq(
    to_x: Dict[Any, List[Any]], freqs: Dict[Any, int]
) -> Dict[Any, List[float]]:
    """
    Applying frequencies to the object returned by the `get_mappings` function

    Args:
        to_x (Dict[Any, List[Any]]): object returned by `get_mappings` function.
        freqs (Dict[Any, int]): object like the one returned by the `get_freqs` function.

    Returns:
        Dict[Any, List[float]]: the same shape as the to_x arg but the values are probabilities now.

    Note:
        freqs arg keys and to_x arg values have to be of the same type

    >>> from cat2cat.mappings import get_mappings, get_freqs, cat_apply_freq
    >>> from cat2cat.datasets import load_trans, load_occup
    >>> mappings = get_mappings(load_trans())
    >>> occup = load_occup()
    >>> codes_new = occup.code[occup.year == 2010].map(str).values
    >>> freqs = get_freqs(codes_new)
    >>> mapp_new_p = cat_apply_freq(mappings["to_new"], freqs)
    >>> mappings["to_new"]['3481']
    ['441401', '441402', '441403', '441490']
    >>> mapp_new_p['3481']
    [0.0, 0.6, 0.0, 0.4]
    """
    assert isinstance(to_x, dict), "to_x has to be a dict"
    assert isinstance(freqs, dict), "freqs has to be dict"
    res = dict()
    for x in to_x:
        cs = [freqs.get(e, 1e-12) for e in to_x[x]]
        fs = round(array(cs) / sum(cs), 10)
        res[x] = list(fs)
    return res
