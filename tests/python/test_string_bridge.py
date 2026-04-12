"""Bridge regressions for string behavior.

`chararray` remains a legacy compatibility surface. It should stay supported as
a thin shim over the shared string behavior, not as a separate runtime path.
"""

import numpy as np


def test_char_upper_preserves_shape_for_ndarray():
    arr = np.array([["aa", "bb"], ["cc", "dd"]])
    out = np.char.upper(arr)
    assert out.shape == (2, 2)
    assert out.tolist() == [["AA", "BB"], ["CC", "DD"]]


def test_char_strip_preserves_object_array_shape():
    arr = np.array([["  a  ", " b "], ["c ", "  d"]], dtype=object)
    out = np.char.strip(arr)
    assert out.shape == (2, 2)
    assert out.tolist() == [["a", "b"], ["c", "d"]]


def test_chararray_upper_matches_np_char_bridge():
    carr = np.char.asarray([["ab", "cd"], ["ef", "gh"]])
    expected = [["AB", "CD"], ["EF", "GH"]]
    np_char_out = np.char.upper(carr)
    carr_out = carr.upper()
    assert np_char_out.shape == (2, 2)
    assert carr_out.shape == (2, 2)
    assert np_char_out.tolist() == expected
    assert carr_out.tolist() == expected


def test_chararray_compare_keeps_trailing_whitespace_quirk():
    carr = np.char.asarray(["ab  ", "cd"])
    out = carr == np.array(["ab", "xx"])
    assert out.tolist() == [True, False]


def test_char_join_preserves_shaped_inputs():
    arr = np.array([["ab", "cd"], ["ef", "gh"]])
    out = np.char.join("-", arr)
    assert out.shape == (2, 2)
    assert out.tolist() == [["a-b", "c-d"], ["e-f", "g-h"]]
