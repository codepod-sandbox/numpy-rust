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


def test_char_upper_handles_0d_object_array_bridge_input():
    arr = _ObjectArray(["abc"], "object", shape=())
    out = np.char.upper(arr)
    assert isinstance(out, np.ndarray)
    assert out.shape == ()
    assert out.tolist() == "ABC"


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


def test_shared_bridge_preserves_bytes_semantics():
    arr = np.array([b"ab", b"cab"])

    replaced = np.char.replace(arr, b"a", b"x")
    stripped = np.char.strip(np.array([b"--ab--", b"cab-"]), b"-")
    partitioned = np.char.partition(arr, b"a")

    assert replaced.tolist() == [b"xb", b"cxb"]
    assert stripped.tolist() == [b"ab", b"cab"]
    assert partitioned.tolist() == [
        [b"", b"a", b"b"],
        [b"c", b"a", b"b"],
    ]


def test_zfill_shared_bridge_preserves_bytes_semantics():
    arr = _ObjectArray([b"42", b"-7"], "object", shape=(2,))
    carr = np.char.asarray(arr)

    arr_out = np.char.zfill(arr, 4)
    carr_out = carr.zfill(4)

    assert arr_out.tolist() == [b"0042", b"-007"]
    assert carr_out.tolist() == [b"0042", b"-007"]


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


def test_startswith_and_replace_accept_array_like_secondary_operands():
    arr = np.array(["alpha", "beta"])
    carr = np.char.asarray(arr)
    prefixes = np.array(["al", "zz"])
    olds = np.array(["a", "b"])
    news = np.array(["A", "B"])

    starts_arr = np.char.startswith(arr, prefixes)
    starts_carr_np = np.char.startswith(carr, prefixes)
    starts_carr_method = carr.startswith(prefixes)
    replaced_arr = np.char.replace(arr, olds, news)
    replaced_carr_np = np.char.replace(carr, olds, news)
    replaced_carr_method = carr.replace(olds, news)

    assert starts_arr.tolist() == [True, False]
    assert starts_carr_np.tolist() == [True, False]
    assert starts_carr_method.tolist() == [True, False]
    assert replaced_arr.tolist() == ["AlphA", "Beta"]
    assert replaced_carr_np.tolist() == ["AlphA", "Beta"]
    assert replaced_carr_method.tolist() == ["AlphA", "Beta"]


def test_char_join_preserves_shaped_inputs():
    arr = np.array([["ab", "cd"], ["ef", "gh"]])
    out = np.char.join("-", arr)
    assert out.shape == (2, 2)
    assert out.tolist() == [["a-b", "c-d"], ["e-f", "g-h"]]


def test_char_join_matches_numpy_for_scalar_and_array_separators():
    seq = np.array(["hello", "world"])
    sep = np.array(["-", "="])
    carr_sep = np.char.asarray(["-", "="])

    scalar_np = np.char.join("-", seq)
    scalar_chararray = np.char.asarray(["-"]).join(seq)
    array_np = np.char.join(sep, np.array(["ab", "cd"]))
    array_chararray = carr_sep.join(np.array(["ab", "cd"]))

    assert scalar_np.tolist() == ["h-e-l-l-o", "w-o-r-l-d"]
    assert scalar_chararray.tolist() == ["h-e-l-l-o", "w-o-r-l-d"]
    assert array_np.tolist() == ["a-b", "c=d"]
    assert array_chararray.tolist() == ["a-b", "c=d"]


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


def test_split_families_preserve_public_return_type():
    arr = np.array(["a b", "c"])
    carr = np.char.asarray(["a b", "c\nd"])

    split_out = np.char.split(arr)
    split_carr_out = carr.split()
    splitlines_out = np.char.splitlines(np.array(["a\nb", "c"]))
    splitlines_carr_out = carr.splitlines()

    assert isinstance(split_out, _ObjectArray)
    assert isinstance(split_carr_out, _ObjectArray)
    assert isinstance(splitlines_out, _ObjectArray)
    assert isinstance(splitlines_carr_out, _ObjectArray)
    assert split_out.shape == (2,)
    assert split_carr_out.shape == (2,)
    assert splitlines_out.shape == (2,)
    assert splitlines_carr_out.shape == (2,)
    assert split_out.tolist() == [["a", "b"], ["c"]]
    assert split_carr_out.tolist() == [["a", "b"], ["c", "d"]]
    assert splitlines_out.tolist() == [["a", "b"], ["c"]]
    assert splitlines_carr_out.tolist() == [["a b"], ["c", "d"]]


def test_split_scalar_and_0d_preserve_numpy_shape_and_type():
    scalar_split = np.char.split("a b")
    zero_d_split = np.char.split(np.array("a b").reshape(()))
    scalar_splitlines = np.char.splitlines("a\nb")
    zero_d_splitlines = np.char.splitlines(np.array("a\nb").reshape(()))

    assert isinstance(scalar_split, _ObjectArray)
    assert isinstance(zero_d_split, _ObjectArray)
    assert isinstance(scalar_splitlines, _ObjectArray)
    assert isinstance(zero_d_splitlines, _ObjectArray)
    assert scalar_split.shape == ()
    assert zero_d_split.shape == ()
    assert scalar_splitlines.shape == ()
    assert zero_d_splitlines.shape == ()
    assert scalar_split.tolist() == ["a", "b"]
    assert zero_d_split.tolist() == ["a", "b"]
    assert scalar_splitlines.tolist() == ["a", "b"]
    assert zero_d_splitlines.tolist() == ["a", "b"]


def test_partition_families_preserve_public_return_shape_and_kind():
    arr = np.array(["a-b", "c-d"])
    carr = np.char.asarray(arr)

    arr_out = np.char.partition(arr, "-")
    carr_np_out = np.char.partition(carr, "-")
    carr_method_out = carr.partition("-")

    assert type(arr_out) is np.ndarray
    assert type(carr_np_out) is np.ndarray
    assert isinstance(carr_method_out, type(carr))
    assert arr_out.shape == (2, 3)
    assert carr_np_out.shape == (2, 3)
    assert carr_method_out.shape == (2, 3)
    assert arr_out.tolist() == [["a", "-", "b"], ["c", "-", "d"]]
    assert carr_np_out.tolist() == [["a", "-", "b"], ["c", "-", "d"]]
    assert carr_method_out.tolist() == [["a", "-", "b"], ["c", "-", "d"]]


def test_partition_scalar_and_0d_preserve_numpy_shape_and_type():
    scalar_out = np.char.partition("a-b", "-")
    zero_d = np.array("a-b").reshape(())
    zero_d_out = np.char.partition(zero_d, "-")
    zero_d_chararray_out = np.char.asarray(zero_d).partition("-")

    assert type(scalar_out) is np.ndarray
    assert type(zero_d_out) is np.ndarray
    assert isinstance(zero_d_chararray_out, type(np.char.asarray(["x"])))
    assert scalar_out.shape == (3,)
    assert zero_d_out.shape == (3,)
    assert zero_d_chararray_out.shape == (3,)
    assert scalar_out.tolist() == ["a", "-", "b"]
    assert zero_d_out.tolist() == ["a", "-", "b"]
    assert zero_d_chararray_out.tolist() == ["a", "-", "b"]


def test_bytes_partition_scalar_and_0d_preserve_public_shape():
    scalar_partition = np.char.partition(b"ab", b"a")
    zero_d_bytes = np.array([b"ab"], dtype="|S2").reshape(())
    zero_d_partition = np.char.partition(zero_d_bytes, b"a")
    scalar_rpartition = np.char.rpartition(b"ab", b"a")
    zero_d_rpartition = np.char.rpartition(zero_d_bytes, b"a")
    zero_d_chararray_partition = np.char.asarray(zero_d_bytes).partition(b"a")

    assert scalar_partition.shape == (3,)
    assert zero_d_partition.shape == (3,)
    assert scalar_rpartition.shape == (3,)
    assert zero_d_rpartition.shape == (3,)
    assert zero_d_chararray_partition.shape == (3,)
    assert scalar_partition.tolist() == [b"", b"a", b"b"]
    assert zero_d_partition.tolist() == [b"", b"a", b"b"]
    assert scalar_rpartition.tolist() == [b"", b"a", b"b"]
    assert zero_d_rpartition.tolist() == [b"", b"a", b"b"]
    assert zero_d_chararray_partition.tolist() == [b"", b"a", b"b"]


def test_encode_decode_share_bridge_behavior_and_public_return_kind():
    arr = np.array(["hi", "yo"])
    bytes_arr = np.array([b"hi", b"yo"])
    carr = np.char.asarray(arr)
    bytes_carr = np.char.asarray(bytes_arr)

    encoded = np.char.encode(arr, "utf-8")
    encoded_carr = carr.encode("utf-8")
    decoded = np.char.decode(bytes_arr, "utf-8")
    decoded_carr = bytes_carr.decode("utf-8")

    assert isinstance(encoded, _ObjectArray)
    assert not isinstance(encoded_carr, type(carr))
    assert type(decoded) is np.ndarray
    assert not isinstance(decoded_carr, type(bytes_carr))
    assert encoded.tolist() == [b"hi", b"yo"]
    assert encoded_carr.tolist() == [b"hi", b"yo"]
    assert decoded.tolist() == ["hi", "yo"]
    assert decoded_carr.tolist() == ["hi", "yo"]


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
