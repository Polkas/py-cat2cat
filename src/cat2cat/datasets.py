from importlib_resources import files, as_file
import numpy as np
from pandas import read_pickle, read_csv
import cat2cat.data

__all__ = ["load_trans", "load_occup"]


def get_file_path(file):
    source = files(cat2cat.data).joinpath(file)
    return source


def load_trans():
    """load trans dataset
    trans dataset containing mappings (transitions) between old (2008) and new (2010) occupational codes

    Returns:
        pandas.DataFrame: trans dataset
    """
    sour = get_file_path("trans.csv")
    with as_file(sour) as fil:
        return read_csv(fil, dtype=str)


def load_occup(small=False):
    """load occup dataset

    occup dataset is an example of unbalance panel dataset.
    This is a simulated data although there are applied a real world characteristics from national statistical office survey.
    The original survey is anonymous and take place every two years.
    It is presenting a characteristics from randomly selected company and then using k step procedure employees are chosen.

    Args:
        small (bool): if to use a shrinked version of dataset

    Returns:
        pandas.DataFrame: occup dataset

    Details:

    """
    sour = get_file_path("occup_small.pkl" if small else "occup.pkl")
    with as_file(sour) as fil:
        return read_pickle(fil)
