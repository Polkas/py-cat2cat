import pandas as pd
import numpy as np
from collections.abc import Iterable


def get_mappings(x):
    """
    Transforming a mapping table with mappings to two associative lists

    Transforming a transition table with mappings to two associative lists
    to rearrange the one classification encoding into another, an associative list that maps keys to values is used.
    More precisely, an association list is used which is a linked list in which each list element consists of a key and value or values.
    An association list where unique categories codes are keys and matching categories from next or previous time point are values.
    A transition table is used to build such associative lists.
    Args:
        x (pandas.DataFrame or numpy.matrix): transition table with 2 columns where first column is assumed to be the older encoding.
    Returns:
        dict: with 2 dicts, `to_old` and `to_new`.

    >>> from cat2cat.mappings import get_mappings
    >>> from numpy import array
    >>> trans = trans_small1 = array([
    ...   [1111, 111101], [1111, 111102], [1123, 111405],
    ...   [1212, 112006], [1212, 112008], [1212, 112090],
    ... ])
    >>> mappings = get_mappings(trans)
    >>> mappings["to_old"]
    {112006: [1212], 112008: [1212], 111405: [1123], 112090: [1212], 111101: [1111], 111102: [1111]}
    >>> mappings["to_new"]
    {1123: [111405], 1212: [112006, 112008, 112090], 1111: [111101, 111102]}
    """
    assert isinstance(
        x, (np.ndarray, pd.core.frame.DataFrame)
    ), "x has to be an array or pandas.DataFrame"
    assert (len(x.shape) == 2) and (
        x.shape[1] == 2
    ), "x should have 2 dimensions and the second one is equal to 2"

    if isinstance(x, pd.core.frame.DataFrame):
        ff = x.iloc[:, 0].values
        ss = x.iloc[:, 1].values
    elif isinstance(x, np.ndarray):
        ff = x[:, 0]
        ss = x[:, 1]
    else:
        raise (TypeError)

    from_old = set(ff)
    from_new = set(ss)

    to_old = dict()
    for e in from_new:
        try:
            idx = ss == e
            to_old[e] = sorted(list(set(ff[idx])))
        except:
            to_old[e] = None

    to_new = dict()
    for e in from_old:
        try:
            idx = ff == e
            to_new[e] = sorted(list(set(ss[idx])))
        except:
            to_new[e] = None

    return dict(to_old=to_old, to_new=to_new)


def get_freqs(x, multiplier=None):
    """
    Getting frequencies from a vector with an optional multiplier
    Args:
        x (Iterable): a list like, categorical variable to summarize.
        multiplier (Iterable, optional): a list like, how many times to repeat certain value, additional weights. Defaults to None.
    Returns:
        dict with unique values and their counts
    >>> get_freqs([1,1,1,2,1,2,2,11])
    {1: 4, 2: 3, 11: 1}
    """
    assert isinstance(x, Iterable), "x has to be an iterable"
    assert (multiplier == None) or (len(x) == len(multiplier))
    input = np.repeat(x, multiplier) if multiplier != None else x
    res = dict(zip(*np.unique(input, return_counts=True)))
    return res


def cat_apply_freq(to_x, freqs):
    """
    Applying frequencies to the object returned by the `get_mappings` function

    Args:
        to_x (dict): _description_
        freqs (pandas.DataFrame or dict): _description_

    Returns:
        dict:
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
        ff = [freqs.get(e, 1e-12) for e in to_x[x]]
        fff = np.around(np.array(ff) / sum(ff), 10)
        res[x] = list(fff)
    return res
