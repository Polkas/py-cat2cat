from cat2cat.datasets import load_trans, load_occup, load_verticals
from cat2cat import cat2cat
from cat2cat.dataclass import cat2cat_data, cat2cat_mappings, cat2cat_ml
from cat2cat.cat2cat_utils import dummy_c2c
from pandas import concat
from numpy import round
import pytest
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis


def int_round(x: float) -> int:
    return int(round(x))


verticals = load_verticals()

trans = load_trans()

occup = load_occup()
occup_small = load_occup(small=True)
o_2006 = occup.loc[occup.year == 2006, :].copy()
o_2008 = o_old = occup.loc[occup.year == 2008, :].copy()
o_2010 = o_new = occup.loc[occup.year == 2010, :].copy()
o_2012 = occup.loc[occup.year == 2012, :].copy()

o_old_int = o_old.copy()
o_old_int["code"] = o_old_int["code"].astype(int)
o_new_int = o_new.copy()
o_new_int["code"] = o_new_int["code"].astype(int)

trans_int = trans.copy()
trans_int.loc[trans_int["old"].isnull(), "old"] = "9999"
trans_int = trans_int.astype({"old": int, "new": int})

nr_rows_old = {"backward": 227662, "forward": 17223}
nr_rows_new = {"backward": 17323, "forward": 18577}
data_dict = {
    "str": {"old": o_old, "new": o_new, "trans": trans},
    "int": {"old": o_old_int, "new": o_new_int, "trans": trans_int},
}
which_target_origin = {"backward": ("old", "new"), "forward": ("new", "old")}


@pytest.mark.parametrize("direction", ["backward", "forward"])
@pytest.mark.parametrize("cat_type", ["str", "int"])
def test_cat2cat_base(direction, cat_type):
    o = data_dict[cat_type]["old"].copy()
    n = data_dict[cat_type]["new"].copy()
    data = cat2cat_data(
        data_dict[cat_type]["old"], data_dict[cat_type]["new"], "code", "code", "year"
    )
    mappings = cat2cat_mappings(data_dict[cat_type]["trans"], direction)
    c2c = cat2cat(data, mappings)
    assert isinstance(c2c, dict)
    assert sorted(list(c2c.keys())) == ["new", "old"]

    assert c2c["old"].shape[0] == nr_rows_old[direction]
    assert c2c["new"].shape[0] == nr_rows_new[direction]

    w_target_p, w_origin_p = which_target_origin[direction]

    assert (
        int_round(c2c[w_origin_p]["wei_freq_c2c"].sum())
        == data_dict[cat_type][w_origin_p].shape[0]
    )
    assert (
        int_round(c2c[w_target_p]["wei_freq_c2c"].sum())
        <= data_dict[cat_type][w_target_p].shape[0]
    )
    assert all(c2c[w_target_p].groupby("index_c2c")["wei_freq_c2c"].sum().round() == 1)
    assert all(c2c[w_origin_p]["wei_freq_c2c"].values == 1)
    assert (
        int_round((c2c[w_target_p]["rep_c2c"] * c2c[w_target_p]["wei_naive_c2c"]).sum())
        == c2c[w_target_p].shape[0]
    )

    assert data_dict[cat_type]["old"].equals(o)
    assert data_dict[cat_type]["new"].equals(n)


def test_cat2cat_custom_freqs():
    freqs = o_new["code"].value_counts().to_dict()
    data = cat2cat_data(o_old, o_new, "code", "code", "year")
    mappings_f = cat2cat_mappings(trans, "backward", freqs)
    c2c = cat2cat(data, mappings_f)

    assert isinstance(c2c, dict)
    assert sorted(list(c2c.keys())) == ["new", "old"]

    assert int_round(c2c["old"]["wei_freq_c2c"].sum()) == o_old.shape[0]
    assert int_round(c2c["new"]["wei_freq_c2c"].sum()) == o_new.shape[0]
    assert c2c["new"].shape[0] == o_new.shape[0]

    assert all(c2c["old"].groupby("index_c2c")["wei_freq_c2c"].sum().round() == 1)
    assert all(c2c["new"]["wei_freq_c2c"].values == 1)

    mappings = cat2cat_mappings(trans, "backward")
    c2c_default = cat2cat(data, mappings)
    assert c2c_default["old"].equals(c2c["old"])

    assert o_old.equals(occup.loc[occup.year == 2008, :])
    assert o_new.equals(occup.loc[occup.year == 2010, :])


def test_cat2cat_ml():
    data = cat2cat_data(o_old, o_new, "code", "code", "year")
    mappings = cat2cat_mappings(trans, "backward")
    ml = cat2cat_ml(
        occup.loc[occup.year >= 2010, :].copy(),
        "code",
        ["salary", "age", "edu", "sex"],
        [LinearDiscriminantAnalysis()],
    )
    c2c = cat2cat(data, mappings, ml)

    assert isinstance(c2c, dict)
    assert sorted(list(c2c.keys())) == ["new", "old"]
    assert (
        int_round(c2c["old"]["wei_LinearDiscriminantAnalysis_c2c"].sum())
        == o_old.shape[0]
    )
    assert c2c["new"].shape[0] == o_new.shape[0]
    assert all(c2c["new"]["wei_LinearDiscriminantAnalysis_c2c"].values == 1)

    assert o_old.equals(occup.loc[occup.year == 2008, :])
    assert o_new.equals(occup.loc[occup.year == 2010, :])


def test_cat2cat_multi():
    data = cat2cat_data(o_2008, o_2010, "code", "code", "year")
    mappings = cat2cat_mappings(trans, "backward")

    occup_back_2008_2010 = cat2cat(data, mappings)
    data = cat2cat_data(
        o_2006, occup_back_2008_2010["old"], "code", "g_new_c2c", "year"
    )
    occup_back_2006_2008 = cat2cat(data, mappings)

    o_2006_n = occup_back_2006_2008["old"]
    o_2008_n = occup_back_2006_2008["new"]  # or occup_back_2008_2010["old"]
    o_2010_n = occup_back_2008_2010["new"]
    o_2012_n = dummy_c2c(o_2012, "code")

    data_final = concat([o_2006_n, o_2008_n, o_2010_n, o_2012_n])

    assert int_round(data_final["wei_freq_c2c"].sum()) == occup.shape[0]
    assert [
        int_round(e) for e in data_final.groupby(["year"])["wei_freq_c2c"].sum()
    ] == [
        o_2006.shape[0],
        o_2008.shape[0],
        o_2010.shape[0],
        o_2012.shape[0],
    ]


def test_cat2cat_direct():
    vert_old = verticals.loc[verticals["v_date"] == "2020-04-01", :]
    vert_new = verticals.loc[verticals["v_date"] == "2020-05-01", :]

    ## extract mapping (transition) table from data using identifier
    trans_v = (
        vert_old.merge(vert_new, on="ean", how="inner")
        .loc[:, ["vertical_x", "vertical_y"]]
        .drop_duplicates()
    )

    data = cat2cat_data(
        old=vert_old,
        new=vert_new,
        id_var="ean",
        cat_var_old="vertical",
        cat_var_new="vertical",
        time_var="v_date",
    )
    mappings = cat2cat_mappings(trans_v, "backward")

    verts = cat2cat(data=data, mappings=mappings)

    assert int_round(verts["old"]["wei_freq_c2c"].sum()) == vert_old.shape[0]

    mappings = cat2cat_mappings(trans_v, "forward")

    verts = cat2cat(data=data, mappings=mappings)

    assert int_round(verts["new"]["wei_freq_c2c"].sum()) == vert_new.shape[0]

    ml = cat2cat_ml(
        vert_old.copy(),
        "vertical",
        ["sales"],
        [LinearDiscriminantAnalysis()],
    )

    verts = cat2cat(data=data, mappings=mappings, ml=ml)

    assert "wei_LinearDiscriminantAnalysis_c2c" in verts["new"].columns
    assert "wei_LinearDiscriminantAnalysis_c2c" in verts["old"].columns

    assert (
        int_round(verts["new"]["wei_LinearDiscriminantAnalysis_c2c"].sum())
        == vert_new.shape[0]
    )
    assert (
        int_round(verts["old"]["wei_LinearDiscriminantAnalysis_c2c"].sum())
        == vert_old.shape[0]
    )

    assert vert_old.equals(verticals.loc[verticals["v_date"] == "2020-04-01", :])
    assert vert_new.equals(verticals.loc[verticals["v_date"] == "2020-05-01", :])
