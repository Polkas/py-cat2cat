# from cat2cat import cat2cat
import pytest
from cat2cat.mappings import get_mappings, get_freqs, cat_apply_freq
from cat2cat.datasets import load_trans, load_occup
from numpy import array
from numpy.random import choice, seed

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


@pytest.mark.parametrize("x", [1, "", [], {}])
def test_get_mappings_wrong(x):
    with pytest.raises(AttributeError):
        get_mappings(x)


class class_with_shape:
    shape = [1, 2]


def test_get_mappings_shape():
    with pytest.raises(TypeError):
        get_mappings(class_with_shape())


# cat_apply_freq


def test_cat_apply_freq():
    actual = cat_apply_freq(
        get_mappings(trans)["to_new"],
        get_freqs(occup.code[occup.year == 2010].map(str).to_list()),
    )["3417"]
    expected = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    assert actual == expected
