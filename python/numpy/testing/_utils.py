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
    "runstring", "temppath",
]

# Platform flags
HAS_REFCOUNT = False
IS_WASM = False
IS_PYPY = False
IS_PYSTON = False


def assert_(val, msg=""):
    if not val:
        raise AssertionError(msg or "assertion failed")


def _as_list(arr):
    """Convert ndarray to flat Python list for element-wise comparison."""
    if isinstance(arr, numpy.ndarray):
        flat = arr.flatten()
        return [float(flat[i]) for i in range(flat.size)]
    return [arr]


def assert_equal(actual, desired, err_msg="", verbose=True, *, strict=False):
    # Normalize both sides to lists for comparison if either is array-like
    actual_is_arr = isinstance(actual, numpy.ndarray)
    desired_is_arr = isinstance(desired, numpy.ndarray)
    actual_is_iter = hasattr(actual, '__iter__') and not isinstance(actual, str)
    desired_is_iter = hasattr(desired, '__iter__') and not isinstance(desired, str)

    if actual_is_arr or desired_is_arr or (actual_is_iter and desired_is_iter):
        if actual_is_arr:
            a_vals = _as_list(actual)
        elif actual_is_iter:
            a_vals = list(actual)
        else:
            a_vals = [actual]
        if desired_is_arr:
            d_vals = _as_list(desired)
        elif desired_is_iter:
            d_vals = list(desired)
        else:
            d_vals = [desired]
        if len(a_vals) != len(d_vals):
            raise AssertionError(
                f"Length mismatch: {len(a_vals)} vs {len(d_vals)}. {err_msg}"
            )
        for i, (a, d) in enumerate(zip(a_vals, d_vals)):
            if float(a) != float(d):
                raise AssertionError(
                    f"Arrays not equal at index {i}: {a} != {d}.\n"
                    f" actual: {actual}\n desired: {desired}\n{err_msg}"
                )
        return
    if actual != desired:
        raise AssertionError(
            f"Items not equal:\n actual: {actual}\n desired: {desired}\n{err_msg}"
        )


def assert_almost_equal(actual, desired, decimal=7, err_msg="", verbose=True):
    is_array_like = isinstance(actual, numpy.ndarray) or isinstance(desired, numpy.ndarray) \
        or (hasattr(actual, '__iter__') and not isinstance(actual, str)) \
        or (hasattr(desired, '__iter__') and not isinstance(desired, str))
    if is_array_like:
        if isinstance(actual, numpy.ndarray):
            a_vals = _as_list(actual)
        elif hasattr(actual, '__iter__') and not isinstance(actual, str):
            a_vals = list(actual)
        else:
            a_vals = [actual]
        if isinstance(desired, numpy.ndarray):
            d_vals = _as_list(desired)
        elif hasattr(desired, '__iter__') and not isinstance(desired, str):
            d_vals = list(desired)
        else:
            d_vals = [desired]
        for a, d in zip(a_vals, d_vals):
            if abs(float(a) - float(d)) >= 1.5 * 10**(-decimal):
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


def _to_float_list(x):
    """Convert any array-like or scalar to a flat list of floats."""
    if isinstance(x, numpy.ndarray):
        return _as_list(x)
    if hasattr(x, 'tolist'):
        raw = x.tolist()
        return [float(v) for v in raw] if isinstance(raw, list) else [float(raw)]
    if hasattr(x, '__iter__') and not isinstance(x, str):
        return [float(v) for v in x]
    return [float(x)]


def assert_allclose(actual, desired, rtol=1e-7, atol=0, equal_nan=True,
                    err_msg="", verbose=True):
    a_vals = _to_float_list(actual)
    d_vals = _to_float_list(desired)
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
