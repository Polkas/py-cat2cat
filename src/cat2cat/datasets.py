from importlib.resources import files, as_file
import numpy as np
from pandas import read_pickle, read_csv
import cat2cat.data


def get_file_path(file):
    source = files(cat2cat.data).joinpath(file)
    return source


def load_trans():
    """_summary_

    Returns:
        pandas.DataFrame: _description_
    """
    sour = get_file_path("trans.csv")
    return read_csv(sour, dtype=str)


def load_occup(small=False):
    """_summary_
    Parameters:
        small (bool): if to use a shrinked version of dataset
    Returns:
        pandas.DataFrame: _description_
    """
    sour = get_file_path("occup_small.pkl" if small else "occup.pkl")
    return read_pickle(sour)
