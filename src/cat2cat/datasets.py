from importlib_resources import files, as_file
from pandas import read_pickle, DataFrame
import cat2cat.data

__all__ = ["load_trans", "load_occup", "load_verticals"]


def _get_file_path(file: str):
    """Get a file path"""
    source = files(cat2cat.data).joinpath(file)
    return source


def load_verticals() -> DataFrame:
    """load trans dataset
    trans dataset containing mappings (transitions) between old (2008) and new (2010) occupational codes

    Returns:
        pandas.DataFrame: trans dataset
    """
    sour = _get_file_path("verticals.pkl")
    with as_file(sour) as fil:
        return read_pickle(fil)


def load_trans() -> DataFrame:
    """load trans dataset
    trans dataset containing mappings (transitions) between old (2008) and new (2010) occupational codes

    Returns:
        pandas.DataFrame: trans dataset
    """
    sour = _get_file_path("trans.pkl")
    with as_file(sour) as fil:
        return read_pickle(fil)


def load_occup(small: bool = False) -> DataFrame:
    """load occup dataset

    occup dataset is an example of unbalance panel dataset.
    This is a simulated data although there are applied a real world characteristics from national statistical office survey.
    The original survey is anonymous and take place every two years.
    It is presenting a characteristics from randomly selected company and then using k step procedure employees are chosen.

    Args:
        small (bool): if to use a shrinked version of dataset

    Returns:
        pandas.DataFrame: occup dataset
    """
    sour = _get_file_path("occup_small.pkl" if small else "occup.pkl")
    with as_file(sour) as fil:
        return read_pickle(fil)
