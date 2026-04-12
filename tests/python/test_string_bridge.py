"""Bridge regressions for string behavior.

`chararray` remains a legacy compatibility surface. It should stay supported as
a thin shim over the shared string behavior, not as a separate runtime path.
"""

import _numpy_native  # ensure tests run against the repo runtime
import numpy as np
import pytest
from numpy._core._exceptions import UFuncTypeError
from numpy._helpers import _ObjectArray


def test_char_upper_preserves_shape_for_ndarray():
    arr = np.array([["aa", "bb"], ["cc", "dd"]])
    out = np.char.upper(arr)
    assert out.shape == (2, 2)
    assert out.tolist() == [["AA", "BB"], ["CC", "DD"]]


def test_char_upper_preserves_shape_for_object_array_bridge():
    arr = _ObjectArray(["ab", "cd", "ef", "gh"], "str", shape=(2, 2))
    out = np.char.upper(arr)
    assert isinstance(out, np.ndarray)
    assert out.shape == (2, 2)
    assert out.tolist() == [["AB", "CD"], ["EF", "GH"]]


def test_char_strip_object_array_raises_type_error():
    arr = np.array([["  a  ", " b "], ["c ", "  d"]], dtype=object)
    with pytest.raises(UFuncTypeError):
        np.char.strip(arr)


def test_chararray_upper_matches_np_char_bridge():
    carr = np.char.asarray([["ab", "cd"], ["ef", "gh"]])
    expected = [["AB", "CD"], ["EF", "GH"]]
    np_char_out = np.char.upper(carr)
    carr_out = carr.upper()
    assert np_char_out.shape == (2, 2)
    assert carr_out.shape == (2, 2)
    assert type(np_char_out) is np.ndarray
    assert isinstance(carr_out, type(carr))
    assert np_char_out.tolist() == expected
    assert carr_out.tolist() == expected


def test_np_char_upper_and_chararray_upper_share_path():
    arr = np.array([["ab", "cd"]])
    carr = np.char.asarray(arr)
    arr_out = np.char.upper(arr)
    carr_np_out = np.char.upper(carr)
    carr_method_out = carr.upper()
    expected = [["AB", "CD"]]

    assert type(arr_out) is np.ndarray
    assert type(carr_np_out) is np.ndarray
    assert isinstance(carr_method_out, type(carr))
    assert arr_out.tolist() == expected
    assert carr_np_out.tolist() == expected
    assert carr_method_out.tolist() == expected


@pytest.mark.parametrize(
    ("method_name", "values", "expected"),
    [
        ("lower", [["AB", "CD"]], [["ab", "cd"]]),
        ("capitalize", [["ab", "cD"]], [["Ab", "Cd"]]),
    ],
)
def test_np_char_and_chararray_methods_keep_bridge_return_split(
    method_name, values, expected
):
    arr = np.array(values)
    carr = np.char.asarray(arr)
    np_char = getattr(np.char, method_name)
    arr_out = np_char(arr)
    carr_np_out = np_char(carr)
    carr_method_out = getattr(carr, method_name)()

    assert type(arr_out) is np.ndarray
    assert type(carr_np_out) is np.ndarray
    assert isinstance(carr_method_out, type(carr))
    assert arr_out.shape == arr.shape
    assert carr_np_out.shape == arr.shape
    assert carr_method_out.shape == arr.shape
    assert arr_out.tolist() == expected
    assert carr_np_out.tolist() == expected
    assert carr_method_out.tolist() == expected


def test_chararray_compare_keeps_trailing_whitespace_quirk():
    carr = np.char.asarray(["ab  ", "cd"])
    out = carr == np.array(["ab", "xx"])
    assert out.tolist() == [True, False]


def test_replace_shared_normalization_keeps_shape_and_return_split():
    arr = np.array([["alpha", "beta"]])
    carr = np.char.asarray(arr)
    arr_out = np.char.replace(arr, "a", "A")
    carr_np_out = np.char.replace(carr, "a", "A")
    carr_method_out = carr.replace("a", "A")
    expected = [["AlphA", "betA"]]

    assert type(arr_out) is np.ndarray
    assert type(carr_np_out) is np.ndarray
    assert isinstance(carr_method_out, type(carr))
    assert arr_out.shape == arr.shape
    assert carr_np_out.shape == arr.shape
    assert carr_method_out.shape == arr.shape
    assert arr_out.tolist() == expected
    assert carr_np_out.tolist() == expected
    assert carr_method_out.tolist() == expected


def test_startswith_shared_normalization_keeps_shape_and_return_kind():
    arr = np.array([["hello", "world"]])
    carr = np.char.asarray(arr)
    arr_out = np.char.startswith(arr, "he")
    carr_np_out = np.char.startswith(carr, "he")
    carr_method_out = carr.startswith("he")
    expected = [[True, False]]

    assert type(arr_out) is np.ndarray
    assert type(carr_np_out) is np.ndarray
    assert type(carr_method_out) is np.ndarray
    assert arr_out.shape == arr.shape
    assert carr_np_out.shape == arr.shape
    assert carr_method_out.shape == arr.shape
    assert arr_out.tolist() == expected
    assert carr_np_out.tolist() == expected
    assert carr_method_out.tolist() == expected


def test_char_join_preserves_shaped_inputs():
    arr = np.array([["ab", "cd"], ["ef", "gh"]])
    out = np.char.join("-", arr)
    assert out.shape == (2, 2)
    assert out.tolist() == [["a-b", "c-d"], ["e-f", "g-h"]]


def _to_python(value):
    return value.tolist() if hasattr(value, "tolist") else value


def test_char_join_matches_chararray_join_for_broadcast_separators():
    sep = np.char.asarray(["-", "="])
    seq = ["ab", "cd"]

    np_char_out = np.char.join(sep, seq)
    carr_out = sep.join(seq)

    assert type(np_char_out) is np.ndarray
    assert isinstance(carr_out, type(sep))
    assert np_char_out.tolist() == ["a-b", "c=d"]
    assert carr_out.tolist() == ["a-b", "c=d"]


@pytest.mark.parametrize(
    ("method_name", "value", "args", "expected"),
    [
        ("lstrip", ["  ab", " cd"], (), ["ab", "cd"]),
        ("rstrip", ["ab  ", "cd "], (), ["ab", "cd"]),
        ("isalnum", ["a1", "b!"], (), [True, False]),
        ("istitle", ["Hello World", "hello"], (), [True, False]),
        ("rsplit", ["a-b-c", "d-e"], ("-", 1), [["a-b", "c"], ["d", "e"]]),
        ("splitlines", ["a\nb", "c"], (), [["a", "b"], ["c"]]),
    ],
)
def test_np_char_exposes_missing_chararray_entrypoints(
    method_name, value, args, expected
):
    arr = np.array(value)
    carr = np.char.asarray(value)

    np_char = getattr(np.char, method_name)
    arr_out = np_char(arr, *args)
    carr_np_out = np_char(carr, *args)
    carr_method_out = getattr(carr, method_name)(*args)

    assert _to_python(arr_out) == expected
    assert _to_python(carr_np_out) == expected
    assert _to_python(carr_method_out) == expected


def test_char_find_preserves_shape():
    arr = np.array([["hello", "world"], ["color", "cold"]])
    out = np.char.find(arr, "lo")
    assert out.shape == (2, 2)
    assert out.tolist() == [[3, -1], [2, -1]]


def test_char_center_matches_chararray_method():
    arr = np.array(["hi", "yo"])
    assert np.char.center(arr, 4).tolist() == np.char.asarray(arr).center(4).tolist()


def test_char_isalpha_matches_chararray_method():
    arr = np.array([["abc", "123"], ["xy", "z9"]])
    assert np.char.isalpha(arr).tolist() == np.char.asarray(arr).isalpha().tolist()
