from cat2cat.mappings import get_mappings, get_freqs, cat_apply_freq
from cat2cat.datasets import load_trans, load_occup
from numpy import array, concatenate, NaN
from numpy.random import choice, seed
from pandas import concat, DataFrame
import pytest

occup = load_occup()
trans = load_trans()
trans_small = [
    [1111, 111101],
    [1111, 111102],
    [1111, 111103],
    [1112, 111201],
    [1112, 111202],
    [1112, 111301],
    [1121, 111402],
    [1122, 111401],
    [1122, 111403],
    [1122, 111404],
    [1123, 111405],
    [1211, 112007],
    [1211, 112016],
    [1211, 112017],
    [1211, 112019],
    [1212, 112002],
    [1212, 112013],
    [1212, 112001],
    [1212, 112006],
    [1212, 112008],
    [1212, 112090],
]

# get_freqs


def test_get_freqs_range():
    actual = get_freqs(list(range(10)))
    expected = {0: 1, 1: 1, 2: 1, 3: 1, 4: 1, 5: 1, 6: 1, 7: 1, 8: 1, 9: 1}
    assert actual == expected


def test_get_freqs_random_array():
    seed(1234)
    actual = get_freqs(choice(5, 100, replace=True))
    expected = {0: 14, 1: 25, 2: 21, 3: 17, 4: 23}
    assert actual == expected


def test_get_freqs_random_list():
    seed(1234)
    actual = get_freqs(list(choice(5, 100, replace=True)))
    expected = {0: 14, 1: 25, 2: 21, 3: 17, 4: 23}
    assert actual == expected


def test_get_freqs_multiplier():
    seed(1234)
    actual = get_freqs(choice(5, 100, replace=True), choice(5, 100, replace=True))
    expected = {0: 25, 1: 60, 2: 40, 3: 27, 4: 43}
    assert actual == expected


def test_get_freqs_multiplier_len():
    seed(1234)
    with pytest.raises(ValueError):
        get_freqs(choice(5, 100, replace=True), choice(5, 90, replace=True))


# get_mappings


def test_get_mappings_array():
    actual = get_mappings(array(trans_small))
    expected = {
        "to_old": {
            112001: [1212],
            112002: [1212],
            112006: [1212],
            112007: [1211],
            112008: [1212],
            112013: [1212],
            112016: [1211],
            112017: [1211],
            112019: [1211],
            111401: [1122],
            111402: [1121],
            111403: [1122],
            111404: [1122],
            111405: [1123],
            111301: [1112],
            112090: [1212],
            111201: [1112],
            111202: [1112],
            111101: [1111],
            111102: [1111],
            111103: [1111],
        },
        "to_new": {
            1121: [111402],
            1122: [111401, 111403, 111404],
            1123: [111405],
            1111: [111101, 111102, 111103],
            1112: [111201, 111202, 111301],
            1211: [112007, 112016, 112017, 112019],
            1212: [112001, 112002, 112006, 112008, 112013, 112090],
        },
    }
    assert actual == expected


def test_get_mappings_DataFrame():
    actual = get_mappings(trans)["to_new"]["3417"]
    expected = [
        "331503",
        "331504",
        "331505",
        "331506",
        "331507",
        "331508",
        "333902",
    ]
    assert actual == expected


# test with NaNs
def test_get_mappings_nan_str():
    trans2 = trans.copy()
    trans2 = trans2.iloc[0:30, :]
    trans2 = concat(
        [trans2, DataFrame({"old": [None, "1111"], "new": ["111101", None]})], axis=0
    )
    actual = get_mappings(trans2)
    expected = {
        "to_old": {
            "111101": ["1111", "None"],
            "111102": ["1111"],
            "111103": ["1111"],
            "111201": ["1112"],
            "111202": ["1112"],
            "111301": ["1112"],
            "111401": ["1122"],
            "111402": ["1121"],
            "111403": ["1122"],
            "111404": ["1122"],
            "111405": ["1123"],
            "112001": ["1212"],
            "112002": ["1212"],
            "112003": ["1212"],
            "112004": ["1212"],
            "112005": ["1212"],
            "112006": ["1212"],
            "112007": ["1211"],
            "112008": ["1212"],
            "112009": ["1212"],
            "112010": ["1212"],
            "112011": ["1212"],
            "112012": ["1212"],
            "112013": ["1212"],
            "112014": ["1212"],
            "112015": ["1212"],
            "112016": ["1211"],
            "112017": ["1211"],
            "112019": ["1211"],
            "112090": ["1212"],
            "None": ["1111"],
        },
        "to_new": {
            "1111": ["111101", "111102", "111103", "None"],
            "1112": ["111201", "111202", "111301"],
            "1121": ["111402"],
            "1122": ["111401", "111403", "111404"],
            "1123": ["111405"],
            "1211": ["112007", "112016", "112017", "112019"],
            "1212": [
                "112001",
                "112002",
                "112003",
                "112004",
                "112005",
                "112006",
                "112008",
                "112009",
                "112010",
                "112011",
                "112012",
                "112013",
                "112014",
                "112015",
                "112090",
            ],
            "None": ["111101"],
        },
    }
    assert str(actual) == str(expected)


# test with NaNs
def test_get_mappings_nan_float():
    trans2 = trans_small.copy()
    trans2 = concatenate([trans2, [[NaN, 111101], [1111, NaN]]])
    actual = get_mappings(trans2)
    expected = {
        "to_old": {
            111101.0: [1111.0, NaN],
            111102.0: [1111.0],
            111103.0: [1111.0],
            111201.0: [1112.0],
            111202.0: [1112.0],
            111301.0: [1112.0],
            111401.0: [1122.0],
            111402.0: [1121.0],
            111403.0: [1122.0],
            111404.0: [1122.0],
            111405.0: [1123.0],
            112001.0: [1212.0],
            112002.0: [1212.0],
            112006.0: [1212.0],
            112007.0: [1211.0],
            112008.0: [1212.0],
            112013.0: [1212.0],
            112016.0: [1211.0],
            112017.0: [1211.0],
            112019.0: [1211.0],
            112090.0: [1212.0],
            NaN: [1111.0],
        },
        "to_new": {
            1111.0: [111101.0, 111102.0, 111103.0, NaN],
            1112.0: [111201.0, 111202.0, 111301.0],
            1121.0: [111402.0],
            1122.0: [111401.0, 111403.0, 111404.0],
            1123.0: [111405.0],
            1211.0: [112007.0, 112016.0, 112017.0, 112019.0],
            1212.0: [112001.0, 112002.0, 112006.0, 112008.0, 112013.0, 112090.0],
            NaN: [111101.0],
        },
    }
    assert str(actual) == str(expected)


@pytest.mark.parametrize("x", [1, "", [], {}])
def test_get_mappings_wrong(x):
    with pytest.raises(AttributeError):
        get_mappings(x)


class class_with_shape:
    shape = [1, 2]


def test_get_mappings_shape():
    with pytest.raises(TypeError):
        get_mappings(class_with_shape())


def test_get_mappings_different_types():
    trans2 = trans.copy()
    trans2["old"] = trans2["old"].astype(float)
    with pytest.raises(AssertionError):
        get_mappings(trans2)


# cat_apply_freq


def test_cat_apply_freq():
    actual = cat_apply_freq(
        get_mappings(trans)["to_new"],
        get_freqs(occup.code[occup.year == 2010].map(str).to_list()),
    )["3417"]
    expected = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    assert actual == expected
