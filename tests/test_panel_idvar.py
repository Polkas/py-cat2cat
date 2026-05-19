from cat2cat import cat2cat
from cat2cat.dataclass import cat2cat_data, cat2cat_mappings
from cat2cat.datasets import load_occup_panel, load_trans


def test_occup_panel_id_var_preserves_one_row_per_direct_match():
    occup_panel = load_occup_panel()
    trans = load_trans()
    old = occup_panel.loc[occup_panel.quarter == "2009Q4", :].copy()
    new = occup_panel.loc[occup_panel.quarter == "2010Q1", :].copy()
    shared_ids = set(old["panel_id"]).intersection(new["panel_id"])

    data_with_id = cat2cat_data(old, new, "code", "code", "quarter", id_var="panel_id")
    mappings = cat2cat_mappings(trans, "backward")

    result_with_id = cat2cat(data_with_id, mappings)

    assert result_with_id["old"].shape[0] == len(shared_ids)
    assert round(result_with_id["old"]["wei_freq_c2c"].sum()) == len(shared_ids)
    assert all(result_with_id["old"]["rep_c2c"] == 1)


def test_occup_panel_direct_matches_have_unit_weights():
    occup_panel = load_occup_panel()
    trans = load_trans()
    old = occup_panel.loc[occup_panel.quarter == "2009Q4", :].copy()
    new = occup_panel.loc[occup_panel.quarter == "2010Q1", :].copy()
    shared_ids = set(old["panel_id"]).intersection(new["panel_id"])

    data = cat2cat_data(old, new, "code", "code", "quarter", id_var="panel_id")
    mappings = cat2cat_mappings(trans, "backward")

    result = cat2cat(data, mappings)
    direct = result["old"].loc[result["old"]["panel_id"].isin(shared_ids), :]

    assert direct.shape[0] == len(shared_ids)
    assert all(direct["rep_c2c"] == 1)
    assert all(direct["wei_freq_c2c"] == 1)