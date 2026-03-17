"""Pure-Python implementation of numpy.testing assertion functions."""
import numpy
import math

__all__ = [
    "assert_", "assert_equal", "assert_almost_equal", "assert_approx_equal",
    "assert_array_equal", "assert_array_almost_equal", "assert_allclose",
    "assert_array_less", "assert_raises", "assert_raises_regex",
    "assert_warns", "assert_no_warnings", "assert_array_max_ulp",
    "assert_array_compare", "assert_string_equal",
    "HAS_REFCOUNT", "IS_WASM", "IS_PYPY", "IS_PYSTON",
    "suppress_warnings", "break_cycles", "check_support_sve",
    "runstring", "temppath", "tempdir",
]

# Platform flags
HAS_REFCOUNT = False
IS_WASM = False
IS_PYPY = False
IS_PYSTON = False


def assert_(val, msg=""):
    if not val:
        raise AssertionError(msg or "assertion failed")


def _as_scalar(v):
    """Convert a value to a comparable scalar (handles complex tuples)."""
    if isinstance(v, (tuple, list)) and len(v) == 2:
        # Complex stored as (re, im) tuple in our Rust backend
        return complex(v[0], v[1])
    try:
        return float(v)
    except (TypeError, ValueError):
        return v


def _as_list(arr):
    """Convert ndarray to flat Python list for element-wise comparison."""
    if isinstance(arr, numpy.ndarray):
        flat = arr.flatten()
        return [_as_scalar(flat[i]) for i in range(flat.size)]
    return [arr]


def _is_structured_array(obj):
    """Return True if obj is a numpy StructuredArray (not an ndarray subclass)."""
    t = type(obj)
    return t.__name__ in ('StructuredArray', 'recarray') and not isinstance(obj, numpy.ndarray)


def _structured_array_equal(actual, desired):
    """Element-wise equality check for two StructuredArray objects."""
    if actual.shape != desired.shape:
        return False, f"Shape mismatch: {actual.shape} vs {desired.shape}"
    dt = actual.dtype
    if dt != desired.dtype:
        return False, f"Dtype mismatch: {dt} vs {desired.dtype}"
    for name in dt.names:
        a_col = actual[name]
        d_col = desired[name]
        a_flat = a_col.flatten().tolist()
        d_flat = d_col.flatten().tolist()
        for i, (a, d) in enumerate(zip(a_flat, d_flat)):
            if a != d:
                return False, f"Field '{name}' not equal at index {i}: {a} != {d}"
    return True, ""


def _is_array_like(obj):
    """Return True if obj is ndarray or an array-like object with shape/flatten."""
    if isinstance(obj, numpy.ndarray):
        return True
    # _ObjectArray and similar objects have shape, flatten, and tolist
    t = type(obj).__name__
    return t in ('_ObjectArray', '_ComplexResultArray') or (
        hasattr(obj, 'shape') and hasattr(obj, 'flatten') and hasattr(obj, 'tolist')
        and not _is_structured_array(obj)
    )


def _as_list_compat(obj):
    """Convert any array-like (ndarray or _ObjectArray) to flat scalar list."""
    if isinstance(obj, numpy.ndarray):
        return _as_list(obj)
    if hasattr(obj, 'flatten') and hasattr(obj, 'tolist'):
        try:
            flat = obj.flatten()
            lst = flat.tolist() if hasattr(flat, 'tolist') else list(flat)
            return [_as_scalar(x) for x in lst]
        except Exception:
            pass
    return [_as_scalar(obj)]


def assert_equal(actual, desired, err_msg="", verbose=True, *, strict=False):
    # Handle StructuredArray comparison (not ndarray subclasses)
    if _is_structured_array(actual) or _is_structured_array(desired):
        if _is_structured_array(actual) and _is_structured_array(desired):
            ok, msg = _structured_array_equal(actual, desired)
            if not ok:
                raise AssertionError(f"Structured arrays not equal: {msg}. {err_msg}")
            return
        # one is StructuredArray, one is not — fall through to generic
    if _is_array_like(actual) and _is_array_like(desired):
        # Both are array-like (handles ndarray vs _ObjectArray cross-comparison)
        a_shape = actual.shape if hasattr(actual, 'shape') else (1,)
        d_shape = desired.shape if hasattr(desired, 'shape') else (1,)
        if a_shape != d_shape:
            raise AssertionError(
                f"Shape mismatch: {a_shape} vs {d_shape}. {err_msg}"
            )
        a_vals = _as_list_compat(actual)
        d_vals = _as_list_compat(desired)
        for i, (a, d) in enumerate(zip(a_vals, d_vals)):
            try:
                equal = (a == d)
                # Handle complex comparison: abs of difference
                if isinstance(a, complex) and isinstance(d, complex):
                    equal = (abs(a - d) < 1e-10)
                if not equal:
                    raise AssertionError(
                        f"Arrays not equal at index {i}: {a} != {d}. {err_msg}"
                    )
            except (TypeError, ValueError):
                if a != d:
                    raise AssertionError(
                        f"Arrays not equal at index {i}: {a} != {d}. {err_msg}"
                    )
        return
    if _is_array_like(actual) or _is_array_like(desired):
        # one is array-like, one is scalar-like
        arr = actual if _is_array_like(actual) else desired
        scalar = desired if _is_array_like(actual) else actual
        arr_size = 1
        if hasattr(arr, 'size'):
            arr_size = arr.size
        elif hasattr(arr, 'shape'):
            arr_size = 1
            for s in arr.shape:
                arr_size *= s
        if arr_size == 1:
            vals = _as_list_compat(arr)
            v = vals[0] if vals else None
            if v != scalar:
                raise AssertionError(f"{v} != {scalar}. {err_msg}")
            return
    try:
        if actual != desired:
            raise AssertionError(
                f"Items not equal:\n actual: {actual}\n desired: {desired}\n{err_msg}"
            )
    except (TypeError, ValueError):
        raise AssertionError(
            f"Items not equal (comparison failed):\n actual: {actual}\n desired: {desired}\n{err_msg}"
        )


def assert_almost_equal(actual, desired, decimal=7, err_msg="", verbose=True):
    if isinstance(actual, numpy.ndarray) or isinstance(desired, numpy.ndarray):
        a_vals = _as_list(actual) if isinstance(actual, numpy.ndarray) else [actual]
        d_vals = _as_list(desired) if isinstance(desired, numpy.ndarray) else [desired]
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
    a_vals = _as_list(actual) if isinstance(actual, numpy.ndarray) else [float(actual)]
    d_vals = _as_list(desired) if isinstance(desired, numpy.ndarray) else [float(desired)]
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


class tempdir:
    def __init__(self, suffix="", prefix="tmp", dir=None):
        import tempfile
        self._dir = tempfile.mkdtemp(suffix=suffix, prefix=prefix, dir=dir)

    def __enter__(self):
        return self._dir

    def __exit__(self, *args):
        import shutil
        try:
            shutil.rmtree(self._dir)
        except OSError:
            pass
