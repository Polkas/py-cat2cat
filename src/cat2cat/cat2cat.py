from pandas import DataFrame, concat
from numpy import arange, repeat, setdiff1d, in1d, intersect1d

from cat2cat.mappings import get_mappings, get_freqs, cat_apply_freq
from cat2cat.dataclass import cat2cat_data, cat2cat_mappings, cat2cat_ml
from cat2cat.cat2cat_utils import dummy_c2c
from cat2cat.cat2cat_ml import _cat2cat_ml

from typing import Optional, Any, Dict

__all__ = ["cat2cat"]


def cat2cat(
    data: cat2cat_data, mappings: cat2cat_mappings, ml: Optional[cat2cat_ml] = None
) -> Dict[str, DataFrame]:
    """Automatic mapping in a panel dataset - cat2cat procedure

    Args:
        data (cat2cat_data): dataclass with data related arguments.
            Please check out the `cat2cat.dataclass.cat2cat_data` for more information.
        mappings (cat2cat_mappings): dataclass with mappings related arguments.
            Please check out the `cat2cat.dataclass.cat2cat_mappings` for more information.
        ml (Optional[cat2cat_ml]): dataclass with ml related arguments.
            Please check out the `cat2cat.dataclass.cat2cat_ml` for more information.

    Returns:
        dict: with 2 DataFrames, old and new.
        There will be added additional columns like index_c2c, g_new_c2c, wei_freq_c2c, rep_c2c, wei_(ml method name)_c2c.
        Additional columns will be informative only for a one DataFrame as we always make the changes to one direction.

    Note:
        1. Without ml section only simple frequencies are assessed.
        When ml model is broken then weights from simple frequencies are taken.
        `knn` method is recommended for smaller datasets.

        2. Please be sure that the categorical variable is of the same type in all places.
        `mappings.trans` arg columns and the `data.cat_var` column have to be of the same type.
        When ml part is applied then `ml.cat_var` has to have the same type too.
        Changes have to be made at the same time for the mapping table and datasets.
        
        3. Missing values in the mapping table or categorical variable can cause problems.
        It is recommended to use string or float types in the mapping table and for categorical variable.
        Alternative solution can be representing missing values as a specific number (9999) or string ("Missing").

    >>> from cat2cat import cat2cat
    >>> from cat2cat.dataclass import cat2cat_data, cat2cat_mappings, cat2cat_ml
    >>> from sklearn.ensemble import RandomForestClassifier
    >>> from cat2cat.datasets import load_trans, load_occup
    >>> trans = load_trans()
    >>> occup = load_occup()
    >>> o_old = occup.loc[occup.year == 2008, :].copy()
    >>> o_new = occup.loc[occup.year == 2010, :].copy()
    >>> data = cat2cat_data(old = o_old, new = o_new, cat_var_old = "code",
    ...                     cat_var_new = "code", time_var = "year")
    >>> mappings = cat2cat_mappings(trans = trans, direction = "forward")
    >>> cat2cat(data = data, mappings = mappings)
    {...

    """
    assert isinstance(data, cat2cat_data), "data arg has to be cat2cat_data instance"
    assert isinstance(
        mappings, cat2cat_mappings
    ), "mappings arg has to be cat2cat_mappings instance"
    assert (ml is None) or isinstance(
        ml, cat2cat_ml
    ), "ml arg has to be cat2cat_ml instance"

    mapps = get_mappings(mappings.trans)

    if mappings.direction == "forward":
        target_name = "new"
        base_name = "old"
    elif mappings.direction == "backward":
        target_name = "old"
        base_name = "new"

    cat_var_base = getattr(data, "cat_var_" + base_name)
    cat_var_target = getattr(data, "cat_var_" + target_name)
    base_df = getattr(data, base_name).copy()
    target_df = getattr(data, target_name).copy()
    mid_df = DataFrame()
    mapp = mapps["to_" + base_name]

    is_direct = not data.id_var is None
    if is_direct:
        id_inner = intersect1d(base_df[data.id_var], target_df[data.id_var])
        id_outer = setdiff1d(target_df[data.id_var], base_df[data.id_var])
        mid_df = dummy_c2c(
            target_df.loc[target_df[data.id_var].isin(id_inner), :].copy(),
            cat_var_base,
        )
        target_df = target_df.loc[target_df[data.id_var].isin(id_outer)].copy()

        tos = data.old.loc[:, [data.id_var, data.cat_var_old]].merge(
            data.new.loc[:, [data.id_var, data.cat_var_new]], on=data.id_var
        )
        tos.columns = ["id", "cat_old", "cat_new"]
        tos_dict = dict(zip(tos["id"], tos["cat_" + base_name]))
        mid_df["g_new_c2c"] = [tos_dict.get(e) for e in mid_df[data.id_var]]

    # frequencies
    freqs: Dict[Any, int]
    if isinstance(mappings.freqs, dict):
        freqs = mappings.freqs
    else:
        freqs = _resolve_frequencies(base_df, cat_var_base, data.multiplier_var)

    # frequencies per category
    mapp_f = cat_apply_freq(mapp, freqs)

    # mappings and frequencies per obs
    a_mapp = [mapp.get(e, []) for e in target_df[cat_var_target]]
    a_mapp_f = [mapp_f.get(e, []) for e in target_df[cat_var_target]]
    lens = [len(e) for e in a_mapp]
    nrow_target = target_df.shape[0]

    # target_df
    # index cat2cat
    target_df = target_df.assign(index_c2c=arange(nrow_target))
    # replication process
    target_df = target_df.iloc[repeat(arange(nrow_target), lens), :]
    # remove duplicates in the index
    target_df = target_df.reset_index(drop=True)
    # cat2cat columns
    target_df["g_new_c2c"] = [e for l in a_mapp for e in l]
    target_df["rep_c2c"] = repeat(lens, lens)
    target_df["wei_naive_c2c"] = 1 / target_df.rep_c2c
    target_df["wei_freq_c2c"] = [e for l in a_mapp_f for e in l]

    # base_df
    base_df = dummy_c2c(base_df, cat_var_base)

    # ML
    if ml is not None:
        for m in ml.models:
            ml_name = type(m).__name__
            ml_colname = "wei_" + ml_name + "_c2c"
            target_df[ml_colname] = target_df["wei_freq_c2c"]
            base_df[ml_colname] = 1
            if is_direct:
                mid_df[ml_colname] = 1

        _cat2cat_ml(ml, mapp, target_df, cat_var_target)

    # Final
    res = dict()
    res[target_name] = concat([target_df, mid_df])
    res[base_name] = base_df

    return res


def _resolve_frequencies(
    base_df: DataFrame,
    cat_var_base: str,
    multiplier_var: Optional[str],
) -> Dict[Any, int]:
    """Resolve the frequencies"""
    freqs: Dict[Any, int]
    if "wei_freq_c2c" in base_df.columns:
        freqs = (
            base_df.groupby(cat_var_base)
            .apply(lambda x: sum(x["wei_freq_c2c"] * x.get(multiplier_var, 1)))
            .to_dict()
        )
    else:
        freqs = get_freqs(
            base_df[cat_var_base].values, base_df.get(multiplier_var, None)
        )
    return freqs
