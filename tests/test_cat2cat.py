from cat2cat.datasets import load_trans, load_occup, load_verticals
from cat2cat import cat2cat
from cat2cat.dataclass import cat2cat_data, cat2cat_mappings, cat2cat_ml
from cat2cat.cat2cat_utils import dummy_c2c
from pandas import concat
from numpy import round

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


def test_cat2cat_base_back():
    data = cat2cat_data(o_old, o_new, "code", "code", "year")
    mappings = cat2cat_mappings(trans, "backward")
    c2c = cat2cat(data, mappings)

    assert isinstance(c2c, dict)
    assert sorted(list(c2c.keys())) == ["new", "old"]

    assert c2c["old"].shape == (227662, 17)
    assert c2c["new"].shape == (17323, 17)
    assert c2c["new"].shape[0] == o_new.shape[0]

    assert int_round(c2c["old"]["wei_freq_c2c"].sum()) == o_old.shape[0]
    assert int_round(c2c["new"]["wei_freq_c2c"].sum()) == o_new.shape[0]
    assert all(c2c["old"].groupby("index_c2c")["wei_freq_c2c"].sum().round() == 1)
    assert all(c2c["new"]["wei_freq_c2c"].values == 1)

    assert (
        int_round((c2c["old"]["rep_c2c"] * c2c["old"]["wei_naive_c2c"]).sum())
        == c2c["old"].shape[0]
    )


def test_cat2cat_base_for():
    data = cat2cat_data(o_old, o_new, "code", "code", "year")
    mappings = cat2cat_mappings(trans, "forward")
    c2c = cat2cat(data, mappings)

    assert isinstance(c2c, dict)
    assert sorted(list(c2c.keys())) == ["new", "old"]

    assert c2c["new"].shape == (18577, 17)
    assert c2c["old"].shape == (17223, 17)

    assert int_round(c2c["old"]["wei_freq_c2c"].sum()) == o_old.shape[0]
    assert int_round(c2c["new"]["wei_freq_c2c"].sum()) <= o_new.shape[0]
    assert all(c2c["old"].groupby("index_c2c")["wei_freq_c2c"].sum().round() == 1)
    assert all(c2c["old"]["wei_freq_c2c"].values == 1)

    assert (
        int_round((c2c["new"]["rep_c2c"] * c2c["new"]["wei_naive_c2c"]).sum())
        == c2c["new"].shape[0]
    )


def test_cat2cat_custom_freqs():
    data = cat2cat_data(o_old, o_new, "code", "code", "year")
    mappings_f = cat2cat_mappings(
        trans, "backward", o_new["code"].value_counts().to_dict()
    )
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
