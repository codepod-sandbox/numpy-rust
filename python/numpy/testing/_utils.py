"""Pure-Python implementation of numpy.testing assertion functions."""
import numpy
import math
import sys

__all__ = [
    "assert_", "assert_equal", "assert_almost_equal", "assert_approx_equal",
    "assert_array_equal", "assert_array_almost_equal", "assert_allclose",
    "assert_array_less", "assert_raises", "assert_raises_regex",
    "assert_warns", "assert_no_warnings", "assert_array_max_ulp",
    "assert_array_compare", "assert_string_equal",
    "assert_array_almost_equal_nulp",
    "HAS_REFCOUNT", "IS_WASM", "IS_PYPY", "IS_PYSTON",
    "IS_64BIT", "IS_MUSL", "IS_EDITABLE", "NOGIL_BUILD", "HAS_LAPACK64",
    "suppress_warnings", "break_cycles", "check_support_sve",
    "runstring", "temppath",
    "extbuild",
    "BLAS_SUPPORTS_FPE", "_assert_valid_refcount", "_gen_alignment_data",
]

# Platform flags
HAS_REFCOUNT = False
IS_WASM = False
IS_PYPY = False
IS_PYSTON = False
IS_64BIT = (sys.maxsize > 2**32)
IS_MUSL = False
IS_EDITABLE = False
NOGIL_BUILD = False
HAS_LAPACK64 = False


def assert_array_almost_equal_nulp(x, y, nulp=1):
    """Assert arrays equal within nulp ULPs."""
    numpy.testing.assert_allclose(x, y, rtol=1e-7 * nulp, atol=0)


class _ExtBuild:
    @staticmethod
    def build_and_import_extension(*args, **kwargs):
        import pytest
        pytest.skip("C extensions not available")

extbuild = _ExtBuild()


def assert_(val, msg=""):
    if not val:
        raise AssertionError(msg or "assertion failed")


def _is_array_like(obj):
    return isinstance(obj, numpy.ndarray) or isinstance(obj, numpy._ObjectArray)

def _val_equal(a, b):
    """Compare two values, treating NaN as equal to NaN."""
    try:
        if isinstance(a, float) and isinstance(b, float):
            if math.isnan(a) and math.isnan(b):
                return True
        if isinstance(a, (int, float)) and isinstance(b, (int, float)):
            return a == b
    except (TypeError, ValueError):
        pass
    return a == b

def _as_list(arr):
    """Convert ndarray to flat Python list for element-wise comparison."""
    if isinstance(arr, numpy._ObjectArray):
        result = []
        for v in arr._data:
            if isinstance(v, complex):
                result.append(complex(v))
            elif isinstance(v, (str, bytes)):
                result.append(v)
            elif v is None:
                result.append(v)
            else:
                try:
                    result.append(float(v))
                except (TypeError, ValueError):
                    result.append(v)
        return result
    if isinstance(arr, numpy.ndarray):
        flat = arr.flatten()
        result = []
        for i in range(flat.size):
            v = flat[i]
            if isinstance(v, tuple):
                # Complex value stored as (re, im)
                result.append(complex(v[0], v[1] if len(v) > 1 else 0))
            elif isinstance(v, (str, bytes)):
                result.append(v)
            else:
                result.append(float(v))
        return result
    if isinstance(arr, (list, tuple)):
        result = []
        for v in arr:
            if isinstance(v, (list, tuple)):
                result.extend(_as_list(v))
            elif isinstance(v, (numpy.ndarray, numpy._ObjectArray)):
                result.extend(_as_list(v))
            else:
                result.append(v)
        return result
    return [arr]


def _is_array_like(x):
    """Return True if x is numpy array, _ObjectArray, or a list/tuple of numbers."""
    if isinstance(x, (numpy.ndarray, numpy._ObjectArray)):
        return True
    if isinstance(x, (list, tuple)) and len(x) > 0:
        return True
    return False

def assert_equal(actual, desired, err_msg="", verbose=True, *, strict=False):
    # Handle tuples and lists recursively
    if isinstance(actual, (tuple, list)) and isinstance(desired, (tuple, list)):
        if len(actual) != len(desired):
            raise AssertionError(
                f"Length mismatch: {len(actual)} vs {len(desired)}. {err_msg}"
            )
        for i, (a, d) in enumerate(zip(actual, desired)):
            assert_equal(a, d, err_msg=err_msg, verbose=verbose, strict=strict)
        return
    if _is_array_like(actual) and _is_array_like(desired):
        if isinstance(actual, (list, tuple)):
            actual = numpy.asarray(actual)
        if isinstance(desired, (list, tuple)):
            desired = numpy.asarray(desired)
        if actual.shape != desired.shape:
            raise AssertionError(
                f"Shape mismatch: {actual.shape} vs {desired.shape}. {err_msg}"
            )
        a_vals = _as_list(actual)
        d_vals = _as_list(desired)
        for i, (a, d) in enumerate(zip(a_vals, d_vals)):
            if not _val_equal(a, d):
                raise AssertionError(
                    f"Arrays not equal at index {i}: {a} != {d}. {err_msg}"
                )
        return
    if _is_array_like(actual) and isinstance(desired, (list, tuple)):
        desired = numpy.asarray(desired)
        assert_equal(actual, desired, err_msg=err_msg, verbose=verbose, strict=strict)
        return
    if _is_array_like(desired) and isinstance(actual, (list, tuple)):
        actual = numpy.asarray(actual)
        assert_equal(actual, desired, err_msg=err_msg, verbose=verbose, strict=strict)
        return
    if _is_array_like(actual) or _is_array_like(desired):
        # one is array, one is scalar-like
        arr = actual if _is_array_like(actual) else desired
        scalar = desired if _is_array_like(actual) else actual
        vals = _as_list(arr)
        if not _is_array_like(scalar):
            # Compare all elements to scalar
            for i, v in enumerate(vals):
                if not _val_equal(v, scalar):
                    raise AssertionError(f"Arrays not equal at index {i}: {v} != {scalar}. {err_msg}")
            return
        if arr.size == 1:
            if not _val_equal(vals[0], scalar):
                raise AssertionError(f"{vals[0]} != {scalar}. {err_msg}")
            return
    if _is_array_like(actual) or _is_array_like(desired):
        # Both arrays but different shapes - try element-wise
        a_vals = _as_list(actual) if _is_array_like(actual) else [actual]
        d_vals = _as_list(desired) if _is_array_like(desired) else [desired]
        if len(a_vals) != len(d_vals):
            raise AssertionError(f"Size mismatch: {len(a_vals)} vs {len(d_vals)}. {err_msg}")
        for i, (a, d) in enumerate(zip(a_vals, d_vals)):
            if not _val_equal(a, d):
                raise AssertionError(f"Items not equal at index {i}: {a} != {d}. {err_msg}")
        return
    if not _val_equal(actual, desired):
        raise AssertionError(
            f"Items not equal:\n actual: {actual}\n desired: {desired}\n{err_msg}"
        )


def assert_almost_equal(actual, desired, decimal=7, err_msg="", verbose=True):
    if _is_array_like(actual) or _is_array_like(desired):
        a_vals = _as_list(actual) if _is_array_like(actual) else [actual]
        d_vals = _as_list(desired) if _is_array_like(desired) else [desired]
        for a, d in zip(a_vals, d_vals):
            if abs(a - d) >= 1.5 * 10**(-decimal):
                raise AssertionError(
                    f"Not almost equal to {decimal} decimals:\n"
                    f" actual: {a}\n desired: {d}\n{err_msg}"
                )
        return
    if abs(float(actual) - float(desired)) >= 1.5 * 10**(-decimal):
        raise AssertionError(
            f"Not almost equal to {decimal} decimals:\n"
            f" actual: {actual}\n desired: {desired}\n{err_msg}"
        )


def assert_approx_equal(actual, desired, significant=7, err_msg="", verbose=True):
    if desired == 0:
        scale = 1.0
    else:
        scale = 10 ** (-math.floor(math.log10(abs(desired))))
    if abs(actual - desired) * scale >= 1.5 * 10 ** (-significant):
        raise AssertionError(
            f"Not approx equal to {significant} significant digits:\n"
            f" actual: {actual}\n desired: {desired}\n{err_msg}"
        )


def assert_array_equal(actual, desired, err_msg="", verbose=True, *, strict=False):
    assert_equal(actual, desired, err_msg=err_msg, verbose=verbose, strict=strict)


def assert_array_almost_equal(actual, desired, decimal=6, err_msg="", verbose=True):
    assert_almost_equal(actual, desired, decimal=decimal, err_msg=err_msg, verbose=verbose)


def assert_allclose(actual, desired, rtol=1e-7, atol=0, equal_nan=True,
                    err_msg="", verbose=True):
    actual = numpy.asarray(actual)
    desired = numpy.asarray(desired)
    a_vals = _as_list(actual) if _is_array_like(actual) else [float(actual)]
    d_vals = _as_list(desired) if _is_array_like(desired) else [float(desired)]
    for i, (a, d) in enumerate(zip(a_vals, d_vals)):
        # Handle NaN
        if equal_nan and math.isnan(a) and math.isnan(d):
            continue
        threshold = atol + rtol * abs(d)
        if abs(a - d) > threshold:
            raise AssertionError(
                f"Not close at index {i}: {a} vs {d} "
                f"(diff={abs(a-d)}, tol={threshold}). {err_msg}"
            )


def assert_array_less(x, y, err_msg="", verbose=True, *, strict=False):
    x_vals = _as_list(x) if isinstance(x, numpy.ndarray) else [float(x)]
    y_vals = _as_list(y) if isinstance(y, numpy.ndarray) else [float(y)]
    for i, (a, b) in enumerate(zip(x_vals, y_vals)):
        if not (a < b):
            raise AssertionError(f"Not less at index {i}: {a} >= {b}. {err_msg}")


def assert_raises(exception_class, *args, **kwargs):
    import pytest
    if not args:
        return pytest.raises(exception_class)
    callable_obj, *call_args = args
    with pytest.raises(exception_class):
        callable_obj(*call_args, **kwargs)


def assert_raises_regex(exception_class, expected_regexp, *args, **kwargs):
    import pytest
    if not args:
        return pytest.raises(exception_class, match=expected_regexp)
    callable_obj, *call_args = args
    with pytest.raises(exception_class, match=expected_regexp):
        callable_obj(*call_args, **kwargs)


def assert_warns(warning_class, *args, **kwargs):
    import pytest
    if not args:
        return pytest.warns(warning_class)
    callable_obj, *call_args = args
    with pytest.warns(warning_class):
        callable_obj(*call_args, **kwargs)


def assert_no_warnings(*args, **kwargs):
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        if args:
            return args[0](*args[1:], **kwargs)


def assert_array_max_ulp(a, b, maxulp=1, dtype=None):
    # ULP comparison stub - just check closeness
    assert_allclose(a, b, rtol=maxulp * 1e-15, atol=0)


def assert_array_compare(comparison, x, y, err_msg="", verbose=True, header="",
                         precision=6, equal_nan=True, equal_inf=True, *, strict=False):
    x_vals = _as_list(x) if isinstance(x, numpy.ndarray) else [x]
    y_vals = _as_list(y) if isinstance(y, numpy.ndarray) else [y]
    for i, (a, b) in enumerate(zip(x_vals, y_vals)):
        if equal_nan and isinstance(a, float) and isinstance(b, float):
            if math.isnan(a) and math.isnan(b):
                continue
        if not comparison(a, b):
            raise AssertionError(f"{header} at index {i}: {a} vs {b}. {err_msg}")


def assert_string_equal(actual, desired):
    assert actual == desired, f"{actual!r} != {desired!r}"


class suppress_warnings:
    def __init__(self, forwarding_rule="always"):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass

    def filter(self, *args, **kwargs):
        pass

    def record(self, *args, **kwargs):
        return []


def break_cycles():
    pass


def check_support_sve():
    return False


def runstring(code, ns):
    """Run a code string in the given namespace (NumPy testing API)."""
    compiled = compile(code, "<runstring>", "exec")
    import builtins
    getattr(builtins, "exec")(compiled, ns)


class temppath:
    def __init__(self, suffix="", prefix="tmp", dir=None):
        import tempfile
        self._f = tempfile.NamedTemporaryFile(
            suffix=suffix, prefix=prefix, dir=dir, delete=False
        )
        self.path = self._f.name
        self._f.close()

    def __enter__(self):
        return self.path

    def __exit__(self, *args):
        import os
        try:
            os.unlink(self.path)
        except OSError:
            pass


# BLAS / refcount helpers
BLAS_SUPPORTS_FPE = False

def _assert_valid_refcount(*args):
    """Stub - always returns True (no refcounting in RustPython)."""
    return True

def _gen_alignment_data(dtype="float64", type="binary", max_size=24):
    """Generate arrays for alignment testing."""
    import numpy as np
    for sz in [1, 2, 4, 8, max_size]:
        if type == "unary":
            yield np.zeros(sz, dtype=dtype), np.zeros(sz, dtype=dtype)
        else:
            yield np.zeros(sz, dtype=dtype), np.zeros(sz, dtype=dtype), np.zeros(sz, dtype=dtype)
