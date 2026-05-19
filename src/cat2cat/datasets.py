import sys as _sys

import numpy as _np

# NumPy 2.x moved its internals from numpy.core to numpy._core.
# Pickles created on NumPy 2.x cannot be loaded on NumPy 1.x (Python 3.8/3.9
# CI runners) because numpy._core does not exist there.  Patch sys.modules so
# that any reference to numpy._core.X resolves to numpy.core.X.
if not hasattr(_np, "_core"):
    import numpy.core as _np_core  # noqa: F401 – side-effect import

    _sys.modules.setdefault("numpy._core", _np_core)
    for _submod_name in list(_sys.modules):
        if _submod_name.startswith("numpy.core."):
            _alias = _submod_name.replace("numpy.core.", "numpy._core.", 1)
            _sys.modules.setdefault(_alias, _sys.modules[_submod_name])

from importlib_resources import files, as_file
from importlib_resources.abc import Traversable
from pandas import read_pickle, DataFrame
import cat2cat.data

__all__ = ["load_trans", "load_occup", "load_occup_panel", "load_verticals"]


def _get_file_path(file: str) -> Traversable:
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


def load_occup_panel() -> DataFrame:
    """load occup_panel dataset

    occup_panel is an occupational panel-style example dataset from the R
    package, useful for validating weighted/probabilistic workflows.

    Returns:
        pandas.DataFrame: occup_panel dataset
    """
    sour = _get_file_path("occup_panel.pkl")
    with as_file(sour) as fil:
        return read_pickle(fil)
