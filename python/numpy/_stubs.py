"""Miscellaneous utilities, format functions, and stubs."""
import sys as _sys
import math as _math
import _numpy_native as _native
from _numpy_native import ndarray
from ._helpers import _ObjectArray, AxisError, _builtin_max, _copy_into, _ComplexResultArray
from ._core_types import (
    dtype, _normalize_dtype, StructuredDtype,
    Float64DType, Float32DType, Float16DType,
    Int8DType, Int16DType, Int32DType, Int64DType,
    UInt8DType, UInt16DType, UInt32DType, UInt64DType,
    Complex64DType, Complex128DType, BoolDType, StrDType,
    BytesDType, VoidDType, ObjectDType,
)
from ._creation import array, asarray, _make_complex_array

__all__ = [
    # error handling
    '_err_state', 'seterr', 'geterr', 'errstate',
    # print formatting
    'set_printoptions', 'get_printoptions', 'printoptions',
    # string/char module
    'char',
    # array representation
    'array_str', 'array_repr', 'array2string',
    # lib module
    'lib',
    # testing module
    'testing',
    # dtypes module
    'dtypes',
    # rec module
    'rec',
    # module stubs
    'core', 'compat', 'exceptions', 'matlib', 'ctypeslib',
    # exception classes
    'ComplexWarning', 'VisibleDeprecationWarning',
    # misc utilities
    'info', 'who', 'show_config', 'einsum_path', 'byte_bounds',
    # format functions
    'format_float_positional', 'format_float_scientific',
    # memmap stub
    'memmap',
    # misc stubs
    'seterrcall', 'geterrcall', 'add_newdoc', 'deprecate', 'get_include',
    # NumpyVersion
    'NumpyVersion',
    # constants
    'tracemalloc_domain', 'use_hugepage', 'nested_iters',
    # linear algebra / product functions
    'outer', 'cross', 'tensordot', 'inner', 'kron', 'matmul', 'vdot', 'einsum',
    # bit manipulation
    'packbits', 'unpackbits',
    # misc numeric
    'binary_repr', 'base_repr', 'frompyfunc',
    # indexing helpers
    'take_along_axis', 'put_along_axis',
    # matrix class
    'matrix',
    # internal helper
    '_has_complex',
]

# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------

_err_state = {"divide": "warn", "over": "warn", "under": "ignore", "invalid": "warn"}
_UNSET_CALL = object()  # sentinel for errstate call= parameter

def seterr(**kwargs):
    """Set floating point error handling."""
    global _err_state
    old = dict(_err_state)
    for k, v in kwargs.items():
        if k == "all":
            for key in _err_state:
                _err_state[key] = v
            continue
        if k not in _err_state:
            raise ValueError("invalid key: %r" % k)
        _err_state[k] = v
    return old

def geterr():
    return dict(_err_state)

class errstate:
    """Context manager for floating point error handling."""
    def __init__(self, *, call=_UNSET_CALL, **kwargs):
        self._kwargs = kwargs
        self._call = call
        self._old = None
        self._old_call = None
        self._entered = False
    def __enter__(self):
        if self._entered:
            raise TypeError("Cannot enter `np.errstate` twice")
        self._entered = True
        self._old = seterr(**self._kwargs)
        if self._call is not _UNSET_CALL:
            self._old_call = seterrcall(self._call)
        return self
    def __exit__(self, *args):
        seterr(**self._old)
        if self._call is not _UNSET_CALL:
            seterrcall(self._old_call)
    def __call__(self, func):
        import functools
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            with type(self)(**self._kwargs, call=self._call):
                return func(*args, **kwargs)
        return wrapper


# ---------------------------------------------------------------------------
# Print options
# ---------------------------------------------------------------------------

_print_options = {
    'precision': 8,
    'threshold': 1000,
    'edgeitems': 3,
    'linewidth': 75,
    'suppress': False,
    'nanstr': 'nan',
    'infstr': 'inf',
    'formatter': None,
    'sign': '-',
    'floatmode': 'maxprec',
    'legacy': False,
    'override_repr': None,
}

def set_printoptions(precision=None, threshold=None, edgeitems=None,
                     linewidth=None, suppress=None, nanstr=None, infstr=None,
                     formatter=None, sign=None, floatmode=None, legacy=None,
                     override_repr=None, **kwargs):
    """Set printing options."""
    if precision is not None: _print_options['precision'] = precision
    if threshold is not None: _print_options['threshold'] = threshold
    if edgeitems is not None: _print_options['edgeitems'] = edgeitems
    if linewidth is not None: _print_options['linewidth'] = linewidth
    if suppress is not None: _print_options['suppress'] = suppress
    if nanstr is not None: _print_options['nanstr'] = nanstr
    if infstr is not None: _print_options['infstr'] = infstr
    if formatter is not None: _print_options['formatter'] = formatter
    if sign is not None: _print_options['sign'] = sign
    if floatmode is not None: _print_options['floatmode'] = floatmode
    if legacy is not None: _print_options['legacy'] = legacy
    if override_repr is not None: _print_options['override_repr'] = override_repr

def get_printoptions():
    """Get current printing options."""
    return dict(_print_options)

class printoptions:
    """Context manager for print options."""
    def __init__(self, **kwargs):
        self._opts = kwargs
        self._old_opts = None
    def __enter__(self):
        self._old_opts = dict(_print_options)
        set_printoptions(**self._opts)
        return self._old_opts.copy()
    def __exit__(self, *args):
        _print_options.clear()
        _print_options.update(self._old_opts)


# ---------------------------------------------------------------------------
# char module
# ---------------------------------------------------------------------------

class _char_mod:
    @staticmethod
    def upper(a):
        return _native.char_upper(a)

    @staticmethod
    def lower(a):
        return _native.char_lower(a)

    @staticmethod
    def capitalize(a):
        return _native.char_capitalize(a)

    @staticmethod
    def strip(a):
        return _native.char_strip(a)

    @staticmethod
    def str_len(a):
        return _native.char_str_len(a)

    @staticmethod
    def startswith(a, prefix):
        return _native.char_startswith(a, prefix)

    @staticmethod
    def endswith(a, suffix):
        return _native.char_endswith(a, suffix)

    @staticmethod
    def replace(a, old, new):
        return _native.char_replace(a, old, new)

    @staticmethod
    def split(a, sep=None, maxsplit=-1):
        """Split each element in a around sep."""
        if isinstance(a, ndarray):
            items = a.tolist()
        elif isinstance(a, _ObjectArray):
            items = a._data
        elif isinstance(a, str):
            items = [a]
        else:
            items = list(a)
        result = []
        for s in items:
            result.append(str(s).split(sep, maxsplit))
        if len(result) == 1:
            return result[0]
        return result

    @staticmethod
    def join(sep, a):
        """Join strings in a with separator sep, element-wise."""
        if isinstance(a, ndarray):
            items = a.tolist()
        elif isinstance(a, _ObjectArray):
            items = a._data
        elif isinstance(a, (list, tuple)):
            items = a
        else:
            items = [a]
        # If items is a list of lists, join each sublist
        if len(items) > 0 and isinstance(items[0], (list, tuple)):
            result = [str(sep).join(str(x) for x in sub) for sub in items]
            return array(result)
        # Otherwise join all items into a single string
        return str(sep).join(str(x) for x in items)

    @staticmethod
    def find(a, sub, start=0, end=None):
        """Find first occurrence of sub in each element of a."""
        if isinstance(a, ndarray):
            items = a.tolist()
        elif isinstance(a, _ObjectArray):
            items = a._data
        elif isinstance(a, str):
            items = [a]
        else:
            items = list(a)
        result = []
        for s in items:
            s = str(s)
            if end is None:
                result.append(s.find(sub, start))
            else:
                result.append(s.find(sub, start, end))
        return array(result)

    @staticmethod
    def count(a, sub, start=0, end=None):
        """Count non-overlapping occurrences of sub in each element of a."""
        if isinstance(a, ndarray):
            items = a.tolist()
        elif isinstance(a, _ObjectArray):
            items = a._data
        elif isinstance(a, str):
            items = [a]
        else:
            items = list(a)
        result = []
        for s in items:
            s = str(s)
            if end is None:
                result.append(s.count(sub, start))
            else:
                result.append(s.count(sub, start, end))
        return array(result)

    @staticmethod
    def add(a, b):
        """Element-wise string concatenation."""
        if isinstance(a, ndarray):
            items_a = a.tolist()
        elif isinstance(a, _ObjectArray):
            items_a = a._data
        elif isinstance(a, str):
            items_a = [a]
        else:
            items_a = list(a)
        if isinstance(b, ndarray):
            items_b = b.tolist()
        elif isinstance(b, _ObjectArray):
            items_b = b._data
        elif isinstance(b, str):
            items_b = [b]
        else:
            items_b = list(b)
        # Broadcast if lengths differ
        if len(items_a) == 1 and len(items_b) > 1:
            items_a = items_a * len(items_b)
        elif len(items_b) == 1 and len(items_a) > 1:
            items_b = items_b * len(items_a)
        result = [str(x) + str(y) for x, y in zip(items_a, items_b)]
        return array(result)

    @staticmethod
    def multiply(a, i):
        """Element-wise string repetition."""
        if isinstance(a, ndarray):
            items = a.tolist()
        elif isinstance(a, _ObjectArray):
            items = a._data
        elif isinstance(a, str):
            items = [a]
        else:
            items = list(a)
        i = int(i)
        result = [str(s) * i for s in items]
        return array(result)

    @staticmethod
    def _to_str_list(a):
        """Convert input to a flat list of strings."""
        if isinstance(a, _ObjectArray):
            return [str(x) for x in a._data]
        if isinstance(a, ndarray):
            data = a.tolist()
            if isinstance(data, list):
                result = []
                for item in data:
                    if isinstance(item, list):
                        result.extend([str(x) for x in item])
                    else:
                        result.append(str(item))
                return result
            return [str(data)]
        if isinstance(a, str):
            return [a]
        if isinstance(a, (list, tuple)):
            result = []
            for item in a:
                if isinstance(item, (list, tuple)):
                    result.extend([str(x) for x in item])
                else:
                    result.append(str(item))
            return result
        return [str(a)]

    @staticmethod
    def center(a, width, fillchar=' '):
        """Pad each string element in a to width, centering the string."""
        data = _char_mod._to_str_list(a)
        return _ObjectArray([s.center(int(width), fillchar) for s in data])

    @staticmethod
    def ljust(a, width, fillchar=' '):
        """Left-justify each string element in a to width."""
        data = _char_mod._to_str_list(a)
        return _ObjectArray([s.ljust(int(width), fillchar) for s in data])

    @staticmethod
    def rjust(a, width, fillchar=' '):
        """Right-justify each string element in a to width."""
        data = _char_mod._to_str_list(a)
        return _ObjectArray([s.rjust(int(width), fillchar) for s in data])

    @staticmethod
    def zfill(a, width):
        """Pad each string element in a with zeros on the left to width."""
        data = _char_mod._to_str_list(a)
        return _ObjectArray([s.zfill(int(width)) for s in data])

    @staticmethod
    def title(a):
        """Return element-wise title cased version of string."""
        data = _char_mod._to_str_list(a)
        return _ObjectArray([s.title() for s in data])

    @staticmethod
    def swapcase(a):
        """Return element-wise with uppercase converted to lowercase and vice versa."""
        data = _char_mod._to_str_list(a)
        return _ObjectArray([s.swapcase() for s in data])

    @staticmethod
    def isalpha(a):
        """Return true for each element if all characters are alphabetic."""
        data = _char_mod._to_str_list(a)
        return array([1.0 if s.isalpha() else 0.0 for s in data]).astype("bool")

    @staticmethod
    def isdigit(a):
        """Return true for each element if all characters are digits."""
        data = _char_mod._to_str_list(a)
        return array([1.0 if s.isdigit() else 0.0 for s in data]).astype("bool")

    @staticmethod
    def isnumeric(a):
        """Return true for each element if all characters are numeric."""
        data = _char_mod._to_str_list(a)
        return array([1.0 if (s.isnumeric() if hasattr(s, 'isnumeric') else s.isdigit()) else 0.0 for s in data]).astype("bool")

    @staticmethod
    def isupper(a):
        """Return true for each element if all cased characters are uppercase."""
        data = _char_mod._to_str_list(a)
        return array([1.0 if s.isupper() else 0.0 for s in data]).astype("bool")

    @staticmethod
    def islower(a):
        """Return true for each element if all cased characters are lowercase."""
        data = _char_mod._to_str_list(a)
        return array([1.0 if s.islower() else 0.0 for s in data]).astype("bool")

    @staticmethod
    def isspace(a):
        """Return true for each element if all characters are whitespace."""
        data = _char_mod._to_str_list(a)
        return array([1.0 if s.isspace() else 0.0 for s in data]).astype("bool")

    @staticmethod
    def isdecimal(a):
        """Return true for each element if all characters are decimal."""
        a = asarray(a)
        return array([1.0 if str(s).isdecimal() else 0.0 for s in a.flatten().tolist()]).reshape(a.shape).astype("bool")

    @staticmethod
    def encode(a, encoding='utf-8', errors='strict'):
        """Encode each string element to bytes."""
        data = _char_mod._to_str_list(a)
        return _ObjectArray([s.encode(encoding, errors) for s in data])

    @staticmethod
    def decode(a, encoding='utf-8', errors='strict'):
        """Decode each bytes element to string."""
        data = _char_mod._to_str_list(a)
        return _ObjectArray([s.decode(encoding, errors) if isinstance(s, bytes) else s for s in data])

char = _char_mod()


# ---------------------------------------------------------------------------
# Array representation
# ---------------------------------------------------------------------------

def array_str(a, max_line_width=None, precision=None, suppress_small=None):
    """Return a string representation of the data in an array."""
    a = asarray(a)
    return str(a)

def array_repr(a, max_line_width=None, precision=None, suppress_small=None):
    """Return the string representation of an array."""
    a = asarray(a)
    return repr(a)

def array2string(a, max_line_width=None, precision=None, suppress_small=None,
                 separator=' ', prefix='', style=None, formatter=None,
                 threshold=None, edgeitems=None, sign=None, floatmode=None,
                 suffix='', legacy=None):
    """Return a string representation of an array."""
    a = asarray(a)
    return repr(a)


# ---------------------------------------------------------------------------
# info / who
# ---------------------------------------------------------------------------

def info(object=None, maxwidth=76, output=None, toplevel='numpy'):
    """Display documentation for numpy objects."""
    if object is not None:
        doc = getattr(object, '__doc__', None)
        if doc:
            print(doc)
        else:
            print("No documentation available for {}".format(object))


def who(vardict=None):
    """Print info about variables in the given dictionary."""
    if vardict is None:
        return
    for name, val in vardict.items():
        if hasattr(val, 'shape'):
            print("{}: shape={}, dtype={}".format(name, val.shape, val.dtype))


# ---------------------------------------------------------------------------
# lib module
# ---------------------------------------------------------------------------

class _LibModule:
    class stride_tricks:
        @staticmethod
        def as_strided(x, shape=None, strides=None, subok=False, writeable=True):
            """Simplified as_strided - creates a new array with the given shape.
            WARNING: This does NOT share memory with the original array.
            It creates a view-like result by repeating/tiling data."""
            x = asarray(x)
            if shape is None:
                return x.copy()
            # Best effort: reshape or tile to match requested shape
            flat = x.flatten().tolist()
            total = 1
            for s in shape:
                total *= s
            # Repeat flat data to fill the requested size
            result = []
            for i in range(total):
                result.append(flat[i % len(flat)])
            return array(result).reshape(shape)

        @staticmethod
        def sliding_window_view(x, window_shape, axis=None):
            """Create a sliding window view of the array."""
            x = asarray(x)
            if isinstance(window_shape, int):
                window_shape = (window_shape,)
            if x.ndim == 1 and len(window_shape) == 1:
                w = window_shape[0]
                data = x.tolist()
                n = len(data) - w + 1
                if n <= 0:
                    return array([]).reshape((0, w))
                rows = []
                for i in range(n):
                    rows.append(data[i:i+w])
                return array(rows)
            raise NotImplementedError("sliding_window_view only supports 1D")

lib = _LibModule()

def _has_complex(result):
    """Check if any element in result is complex (avoids shadowed builtin any)."""
    for r in result:
        if isinstance(r, complex):
            return True
    return False


def _scimath_wrap(result):
    """Wrap complex128 ndarray results in _ComplexResultArray for compat.
    RustPython returns complex scalars as (re, im) tuples; _ComplexResultArray
    converts them to proper Python complex objects on element access."""
    if hasattr(result, 'dtype') and result.dtype == dtype('complex128'):
        flat = result.flatten().tolist()
        flat_complex = [complex(v[0], v[1]) if isinstance(v, tuple) else complex(v) for v in flat]
        return _ComplexResultArray(flat_complex, result.shape)
    return result


class _ScimathModule:
    """Complex-safe math functions (numpy.lib.scimath)."""

    @staticmethod
    def sqrt(x):
        from ._creation import asarray
        return _scimath_wrap(_native.scimath_sqrt(asarray(x)))

    @staticmethod
    def log(x):
        from ._creation import asarray
        return _scimath_wrap(_native.scimath_log(asarray(x)))

    @staticmethod
    def log2(x):
        from ._creation import asarray
        return _scimath_wrap(_native.scimath_log2(asarray(x)))

    @staticmethod
    def log10(x):
        from ._creation import asarray
        return _scimath_wrap(_native.scimath_log10(asarray(x)))

    @staticmethod
    def arcsin(x):
        from ._creation import asarray
        return _scimath_wrap(_native.scimath_arcsin(asarray(x)))

    @staticmethod
    def arccos(x):
        from ._creation import asarray
        return _scimath_wrap(_native.scimath_arccos(asarray(x)))

    @staticmethod
    def arctanh(x):
        from ._creation import asarray
        return _scimath_wrap(_native.scimath_arctanh(asarray(x)))

    @staticmethod
    def power(x, p):
        from ._creation import asarray
        return _scimath_wrap(_native.scimath_power(asarray(x), asarray(p)))

lib.scimath = _ScimathModule()


# ---------------------------------------------------------------------------
# NumpyVersion
# ---------------------------------------------------------------------------

class NumpyVersion:
    """Minimal numpy version comparison class."""
    def __init__(self, vstring):
        self.vstring = vstring
        parts = vstring.split('.')
        self.major = int(parts[0]) if len(parts) > 0 else 0
        self.minor = int(parts[1]) if len(parts) > 1 else 0
        self.bugfix = int(parts[2].split('rc')[0].split('a')[0].split('b')[0]) if len(parts) > 2 else 0
    def __repr__(self):
        return f"NumpyVersion('{self.vstring}')"
    def __str__(self):
        return self.vstring
    def __lt__(self, other):
        if isinstance(other, str): other = NumpyVersion(other)
        return (self.major, self.minor, self.bugfix) < (other.major, other.minor, other.bugfix)
    def __le__(self, other):
        if isinstance(other, str): other = NumpyVersion(other)
        return (self.major, self.minor, self.bugfix) <= (other.major, other.minor, other.bugfix)
    def __gt__(self, other):
        if isinstance(other, str): other = NumpyVersion(other)
        return (self.major, self.minor, self.bugfix) > (other.major, other.minor, other.bugfix)
    def __ge__(self, other):
        if isinstance(other, str): other = NumpyVersion(other)
        return (self.major, self.minor, self.bugfix) >= (other.major, other.minor, other.bugfix)
    def __eq__(self, other):
        if isinstance(other, str): other = NumpyVersion(other)
        return (self.major, self.minor, self.bugfix) == (other.major, other.minor, other.bugfix)

lib.NumpyVersion = NumpyVersion


# ---------------------------------------------------------------------------
# testing module
# ---------------------------------------------------------------------------

class _AssertRaisesRegexContext:
    def __init__(self, exc_class, pattern):
        self.exc_class = exc_class
        self.pattern = pattern
    def __enter__(self):
        return self
    def __exit__(self, exc_type, exc_val, exc_tb):
        import re
        if exc_type is None:
            raise AssertionError("{} not raised".format(self.exc_class.__name__))
        if not issubclass(exc_type, self.exc_class):
            return False
        if not re.search(self.pattern, str(exc_val)):
            raise AssertionError("'{}' did not match '{}'".format(str(exc_val), self.pattern))
        return True


class _TestingModule:
    def assert_allclose(self, actual, desired, rtol=1e-7, atol=0, equal_nan=True, err_msg='', verbose=True):
        from . import allclose
        actual = asarray(actual)
        desired = asarray(desired)
        if not allclose(actual, desired, rtol=rtol, atol=atol, equal_nan=equal_nan):
            actual_list = actual.tolist()
            desired_list = desired.tolist()
            raise AssertionError(err_msg or "Not equal to tolerance rtol={}, atol={}\n Actual: {}\n Desired: {}".format(rtol, atol, actual_list, desired_list))

    def assert_array_equal(self, x, y, err_msg='', verbose=True, strict=False):
        from . import array_equal, broadcast_to
        x = asarray(x)
        y = asarray(y)
        # Handle scalar vs array comparison (NumPy broadcasts)
        if x.shape != y.shape:
            # 0-D vs 1-element: equivalent for comparison purposes
            if x.ndim == 0 and y.size == 1:
                y = y.reshape(())
            elif y.ndim == 0 and x.size == 1:
                x = x.reshape(())
            elif y.size == 1:
                y = broadcast_to(y.flatten(), x.shape)
            elif x.size == 1:
                x = broadcast_to(x.flatten(), y.shape)
        if not array_equal(x, y, equal_nan=True):
            raise AssertionError(err_msg or "Arrays are not equal\n x: {}\n y: {}".format(x.tolist(), y.tolist()))

    def assert_array_almost_equal(self, x, y, decimal=6, err_msg='', verbose=True):
        from . import allclose
        x = asarray(x)
        y = asarray(y)
        if not allclose(x, y, rtol=0, atol=1.5 * 10**(-decimal)):
            raise AssertionError(err_msg or "Arrays are not almost equal to {} decimals".format(decimal))

    def assert_equal(self, actual, desired, err_msg='', verbose=True):
        from . import array_equal, broadcast_to
        # Handle tuples/lists recursively
        if isinstance(actual, (tuple, list)) and isinstance(desired, (tuple, list)):
            if len(actual) != len(desired):
                raise AssertionError(err_msg or "Length mismatch: {} vs {}".format(len(actual), len(desired)))
            for i, (a, d) in enumerate(zip(actual, desired)):
                self.assert_equal(a, d, err_msg=err_msg, verbose=verbose)
            return
        actual_a = asarray(actual)
        desired_a = asarray(desired)
        # Empty array is vacuously equal to any scalar
        if actual_a.size == 0 and desired_a.size <= 1:
            return
        if desired_a.size == 0 and actual_a.size <= 1:
            return
        # Handle 0-d scalar comparison: extract element and compare directly
        if actual_a.shape == () and desired_a.size == 1:
            a_val = actual_a.flatten()[0]
            d_val = desired_a.flatten()[0]
            if a_val != d_val:
                raise AssertionError(err_msg or "Items are not equal:\n actual: {}\n desired: {}".format(actual, desired))
            return
        if desired_a.shape == () and actual_a.size == 1:
            a_val = actual_a.flatten()[0]
            d_val = desired_a.flatten()[0]
            if a_val != d_val:
                raise AssertionError(err_msg or "Items are not equal:\n actual: {}\n desired: {}".format(actual, desired))
            return
        # Handle scalar vs array comparison
        if actual_a.shape != desired_a.shape:
            if desired_a.size == 1:
                desired_a = broadcast_to(desired_a.flatten(), actual_a.shape)
            elif actual_a.size == 1:
                actual_a = broadcast_to(actual_a.flatten(), desired_a.shape)
        if not array_equal(actual_a, desired_a, equal_nan=True):
            raise AssertionError(err_msg or "Items are not equal:\n actual: {}\n desired: {}".format(actual, desired))

    def assert_raises(self, exception_class, *args, **kwargs):
        """Simple assert_raises - returns a context manager."""
        class _AssertRaisesCtx:
            def __init__(self, exc_cls):
                self.exc_cls = exc_cls
            def __enter__(self):
                return self
            def __exit__(self, exc_type, exc_val, exc_tb):
                if exc_type is None:
                    raise AssertionError("Expected {} but no exception raised".format(self.exc_cls.__name__))
                if not issubclass(exc_type, self.exc_cls):
                    return False  # re-raise
                return True  # suppress exception
        if args:
            # Called as assert_raises(Error, func, *args)
            callable_obj = args[0]
            rest = args[1:]
            try:
                callable_obj(*rest, **kwargs)
            except exception_class:
                return
            raise AssertionError("Expected {}".format(exception_class.__name__))
        return _AssertRaisesCtx(exception_class)

    def assert_raises_regex(self, exception_class, expected_regex, *args, **kwargs):
        """Assert that an exception is raised matching a regex."""
        import re
        callable_obj = kwargs.pop('callable', None)
        if callable_obj is None and len(args) >= 1:
            callable_obj = args[0]
            args = args[1:]
        if callable_obj is not None:
            try:
                callable_obj(*args, **kwargs)
            except exception_class as e:
                if not re.search(expected_regex, str(e)):
                    raise AssertionError(
                        "Exception message '{}' did not match '{}'".format(str(e), expected_regex))
                return
            except Exception as e:
                raise AssertionError(
                    "Expected {}, got {}".format(exception_class.__name__, type(e).__name__))
            raise AssertionError("{} not raised".format(exception_class.__name__))
        else:
            # Context manager mode
            return _AssertRaisesRegexContext(exception_class, expected_regex)

    def assert_warns(self, warning_class, *args, **kwargs):
        """Assert that a warning is raised. Since we don't have warnings module, just run the callable."""
        callable_obj = kwargs.pop('callable', None)
        if callable_obj is None and len(args) >= 1:
            callable_obj = args[0]
            args = args[1:]
        if callable_obj is not None:
            return callable_obj(*args, **kwargs)
        # Return a context manager that suppresses warnings
        class _WarnCtx:
            def __enter__(self_ctx):
                return self_ctx
            def __exit__(self_ctx, *exc):
                return False
        return _WarnCtx()

    def assert_approx_equal(self, actual, desired, significant=7, err_msg='', verbose=True):
        """Assert approximately equal to given number of significant digits."""
        if desired == 0:
            if _math.fabs(actual) > 1.5 * 10**(-significant):
                raise AssertionError("{} != {} to {} significant digits".format(actual, desired, significant))
        else:
            rel = _math.fabs((actual - desired) / desired)
            if rel > 1.5 * 10**(-significant):
                raise AssertionError("{} != {} to {} significant digits".format(actual, desired, significant))

    def assert_array_less(self, x, y, err_msg='', verbose=True):
        """Assert array_like x is less than array_like y, element-wise."""
        x = asarray(x)
        y = asarray(y)
        if not all((x < y).flatten().tolist()):
            raise AssertionError("Arrays are not less-ordered\nx: {}\ny: {}".format(x.tolist(), y.tolist()))


# testing module is now properly defined in numpy/testing/_utils.py
# Import the real module instead of using the stub class
import numpy.testing as testing


# ---------------------------------------------------------------------------
# dtypes module
# ---------------------------------------------------------------------------

class _dtypes_mod:
    Float64DType = Float64DType
    Float32DType = Float32DType
    Float16DType = Float16DType
    Int8DType = Int8DType
    Int16DType = Int16DType
    Int32DType = Int32DType
    Int64DType = Int64DType
    UInt8DType = UInt8DType
    UInt16DType = UInt16DType
    UInt32DType = UInt32DType
    UInt64DType = UInt64DType
    Complex64DType = Complex64DType
    Complex128DType = Complex128DType
    BoolDType = BoolDType
    StrDType = StrDType
    BytesDType = BytesDType
    VoidDType = VoidDType
    ObjectDType = ObjectDType

dtypes = _dtypes_mod()


# ---------------------------------------------------------------------------
# rec module
# ---------------------------------------------------------------------------

class _RecModule:
    """Minimal np.rec namespace."""
    def __init__(self):
        self.recarray = None  # placeholder

    def array(self, data, dtype=None, formats=None, names=None, shape=None, byteorder=None):
        """Create a record array (falls back to regular array)."""
        import numpy as np
        # Build dtype from formats if provided
        if dtype is None and formats is not None:
            if isinstance(formats, str):
                fmt_list = [f.strip() for f in formats.split(',')]
            else:
                fmt_list = list(formats)
            if names is None:
                names = ['f{}'.format(i) for i in range(len(fmt_list))]
            elif isinstance(names, str):
                names = [n.strip() for n in names.split(',')]
            fields = list(zip(names, fmt_list))
            dtype = StructuredDtype(fields)
        if isinstance(data, (list, tuple)):
            if dtype is not None:
                if isinstance(dtype, list):
                    dtype = StructuredDtype(dtype)
                return np.array(data, dtype=dtype)
        arr = asarray(data)
        if dtype is not None:
            dt = dtype if isinstance(dtype, StructuredDtype) else StructuredDtype(dtype) if isinstance(dtype, list) else dtype
            # Try to attach structured dtype metadata; silently skip if type doesn't allow it
            try:
                arr._structured_dtype = dt
            except (TypeError, AttributeError):
                pass
        return arr

    def fromarrays(self, arrays, dtype=None, names=None):
        """Create a record array from separate arrays."""
        import numpy as np
        if isinstance(names, str):
            names = [n.strip() for n in names.split(',')]
        if dtype is None and names is not None:
            # Infer dtype from each array
            fields = []
            for i, (name, arr) in enumerate(zip(names, arrays)):
                a = np.asarray(arr)
                fields.append((name, str(a.dtype)))
            dtype = StructuredDtype(fields)
        if dtype is not None:
            if not isinstance(dtype, StructuredDtype):
                if isinstance(dtype, list):
                    dtype = StructuredDtype(dtype)
            # Build structured array from column arrays
            col_arrays = [np.asarray(a) for a in arrays]
            n = len(col_arrays[0]) if len(col_arrays) > 0 else 0
            records = []
            for i in range(n):
                rec = tuple(a[i] if hasattr(a, '__getitem__') else a for a in col_arrays)
                records.append(rec)
            return np.array(records, dtype=dtype)
        return np.array(arrays)

    def fromrecords(self, reclist, dtype=None, names=None, formats=None):
        """Create a record array from a list of records (tuples)."""
        import numpy as np
        if isinstance(names, str):
            names = [n.strip() for n in names.split(',')]
        if dtype is not None:
            if isinstance(dtype, list):
                dtype = StructuredDtype(dtype)
            return np.array(reclist, dtype=dtype)
        if names is not None and formats is not None:
            if isinstance(formats, str):
                formats = [f.strip() for f in formats.split(',')]
            fields = list(zip(names, formats))
            dt = StructuredDtype(fields)
            return np.array(reclist, dtype=dt)
        if names is not None:
            # Infer types from first record
            if len(reclist) > 0:
                first = reclist[0]
                fields = []
                for i, name in enumerate(names):
                    val = first[i] if isinstance(first, (list, tuple)) else first
                    if isinstance(val, int):
                        fields.append((name, 'int64'))
                    elif isinstance(val, float):
                        fields.append((name, 'float64'))
                    elif isinstance(val, str):
                        # Find max string length
                        max_len = max(len(str(r[i] if isinstance(r, (list, tuple)) else r)) for r in reclist)
                        fields.append((name, 'U' + str(max(max_len, 1))))
                    elif isinstance(val, bytes):
                        max_len = max(len(r[i] if isinstance(r, (list, tuple)) else r) for r in reclist)
                        fields.append((name, 'S' + str(max(max_len, 1))))
                    else:
                        fields.append((name, 'float64'))
                dt = StructuredDtype(fields)
                return np.array(reclist, dtype=dt)
            return np.array(reclist)
        return np.array(reclist)

rec = _RecModule()


# ---------------------------------------------------------------------------
# show_config, einsum_path, byte_bounds
# ---------------------------------------------------------------------------

def show_config():
    """Show numpy-rust build configuration."""
    print("numpy-rust (codepod)")
    print("  backend: Rust + RustPython")

def einsum_path(*operands, optimize='greedy'):
    """Evaluate optimal contraction order (stub returns naive path).

    Performs the same input validation as einsum so that bad calls
    raise the same errors (ValueError / TypeError).
    """
    if len(operands) == 0:
        raise ValueError("No input operands")
    # Parse subscripts / arrays just like einsum does
    if not isinstance(operands[0], str):
        # Interleaved format – first operand must be array-like
        raise TypeError("subscripts must be a string")
    subscripts = operands[0]
    if not isinstance(subscripts, str):
        raise TypeError("subscripts must be a string")
    arrays = list(operands[1:])
    if len(arrays) == 0 and subscripts == '':
        raise ValueError("No input operands")
    # Convert to ndarrays
    for i in range(len(arrays)):
        arrays[i] = asarray(arrays[i])
    # Check operand count vs subscripts
    if '->' in subscripts:
        input_part = subscripts.split('->')[0]
    else:
        input_part = subscripts
    n_subs = len(input_part.split(','))
    if n_subs != len(arrays):
        raise ValueError(
            "einsum: {} operands but subscripts specify {}".format(
                len(arrays), n_subs))
    # Check dims vs subscript indices
    input_terms = input_part.split(',')
    for idx, term in enumerate(input_terms):
        if idx < len(arrays):
            clean = term.replace('...', '')
            n_explicit = len(clean)
            arr_ndim = arrays[idx].ndim
            has_ellipsis = '...' in term
            if has_ellipsis:
                if arr_ndim < n_explicit:
                    raise ValueError(
                        "einsum: operand has {} dims but subscript "
                        "has {} indices".format(arr_ndim, n_explicit))
            else:
                if n_explicit != arr_ndim:
                    raise ValueError(
                        "einsum: operand has {} dims but subscript "
                        "has {} indices".format(arr_ndim, n_explicit))
    # Return naive path
    n = len(arrays)
    path = [(0, 1)] * _builtin_max(1, n - 1)
    return path, ""

def byte_bounds(a):
    """Return low and high byte pointers (stub returns (0, nbytes))."""
    arr = asarray(a)
    return (0, arr.nbytes)


# ---------------------------------------------------------------------------
# Module stubs
# ---------------------------------------------------------------------------

# np.core module stub
class _CoreModule:
    """Minimal np.core namespace."""
    pass

core = _CoreModule()
core.numeric = core  # self-reference for np.core.numeric compatibility
core.multiarray = core  # np.core.multiarray compatibility
core.fromnumeric = core  # np.core.fromnumeric compatibility

# np.compat module stub
class _CompatModule:
    pass
compat = _CompatModule()

# np.exceptions module
class _ExceptionsModule:
    AxisError = AxisError  # already defined
    ComplexWarning = type('ComplexWarning', (UserWarning,), {})
    DTypePromotionError = type('DTypePromotionError', (TypeError,), {})
    VisibleDeprecationWarning = type('VisibleDeprecationWarning', (UserWarning,), {})
    ModuleDeprecationWarning = type('ModuleDeprecationWarning', (DeprecationWarning,), {})
    RankWarning = type('RankWarning', (UserWarning,), {})
    TooHardError = type('TooHardError', (RuntimeError,), {})

exceptions = _ExceptionsModule()
exceptions.__name__ = 'numpy.exceptions'

# Expose exception classes at top level (sklearn fallback path)
ComplexWarning = exceptions.ComplexWarning
VisibleDeprecationWarning = exceptions.VisibleDeprecationWarning

# np.matlib stub
class _MatlibModule:
    """Minimal np.matlib namespace."""
    pass
matlib = _MatlibModule()

# np.ctypeslib stub
class _CtypeslibModule:
    pass
ctypeslib = _CtypeslibModule()


# ---------------------------------------------------------------------------
# format_float functions
# ---------------------------------------------------------------------------

def format_float_positional(x, precision=None, unique=True, fractional=True, trim='k', sign=False, pad_left=None, pad_right=None, min_digits=None):
    """Format a float in positional notation."""
    from numpy._core.arrayprint import format_float_positional as _ffp
    return _ffp(x, precision=precision, unique=unique, fractional=fractional,
                trim=trim, sign=sign, pad_left=pad_left, pad_right=pad_right,
                min_digits=min_digits)

def format_float_scientific(x, precision=None, unique=True, trim='k', sign=False, pad_left=None, exp_digits=None, min_digits=None):
    """Format a float in scientific notation."""
    from numpy._core.arrayprint import format_float_scientific as _ffs
    return _ffs(x, precision=precision, unique=unique, trim=trim, sign=sign,
                pad_left=pad_left, exp_digits=exp_digits, min_digits=min_digits)


# ---------------------------------------------------------------------------
# memmap stub
# ---------------------------------------------------------------------------

class memmap:
    """Memory-mapped file stub (not supported in sandboxed environment)."""
    def __new__(cls, filename, dtype=None, mode='r+', offset=0, shape=None, order='C'):
        raise NotImplementedError("memmap not supported in sandboxed environment")


# ---------------------------------------------------------------------------
# Misc stubs
# ---------------------------------------------------------------------------

_errcall_func = None

def seterrcall(func):
    """Set callback for floating-point error handler."""
    global _errcall_func
    old = _errcall_func
    _errcall_func = func
    return old

def geterrcall():
    """Get callback for floating-point error handler."""
    return _errcall_func

def add_newdoc(place, obj, doc):
    """Add documentation (no-op in our implementation)."""
    pass

def deprecate(func=None, oldname=None, newname=None, message=None):
    """Deprecation decorator (no-op)."""
    if func is not None:
        return func
    def decorator(f):
        return f
    return decorator

def get_include():
    """Return the numpy include directory."""
    import os
    return os.path.dirname(os.path.abspath(__file__))


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

tracemalloc_domain = 0
use_hugepage = 0
nested_iters = None  # Not supported


# ---------------------------------------------------------------------------
# Linear algebra / product functions
# ---------------------------------------------------------------------------

def outer(a, b, out=None):
    """Compute outer product."""
    a = asarray(a).flatten()
    b = asarray(b).flatten()
    result = _native.outer(a, b)
    if out is not None:
        _copy_into(out, result)
        return out
    return result


def cross(a, b, axisa=-1, axisb=-1, axisc=-1, axis=None):
    """Cross product of two arrays."""
    from ._manipulation import moveaxis, broadcast_shapes, broadcast_to
    _a_scalar = not isinstance(a, (ndarray, list, tuple))
    _b_scalar = not isinstance(b, (ndarray, list, tuple))
    a = asarray(a) if not isinstance(a, ndarray) else a
    b = asarray(b) if not isinstance(b, ndarray) else b
    if a.ndim == 0 or b.ndim == 0 or _a_scalar or _b_scalar:
        raise ValueError("At least one array has zero dimension")
    if axis is not None:
        axisa = axisb = axisc = axis
    if axisa < -a.ndim or axisa >= a.ndim:
        raise AxisError(axisa, a.ndim, "axisa")
    if axisb < -b.ndim or axisb >= b.ndim:
        raise AxisError(axisb, b.ndim, "axisb")
    if a.ndim > 1 and axisa != -1 and axisa != a.ndim - 1:
        a = moveaxis(a, axisa, -1)
    if b.ndim > 1 and axisb != -1 and axisb != b.ndim - 1:
        b = moveaxis(b, axisb, -1)
    if a.ndim >= 2 and b.ndim == 1:
        b = b.reshape((1,) * (a.ndim - 1) + (b.shape[0],))
        b_shape = list(a.shape[:-1]) + [b.shape[-1]]
        b_flat = b.flatten().tolist()
        b_new = []
        batch = 1
        for s in a.shape[:-1]:
            batch *= s
        vec_len = b.shape[-1]
        for i in range(batch):
            b_new.extend(b_flat[:vec_len])
        b = array(b_new).reshape(b_shape)
    elif b.ndim >= 2 and a.ndim == 1:
        a = a.reshape((1,) * (b.ndim - 1) + (a.shape[0],))
        a_shape = list(b.shape[:-1]) + [a.shape[-1]]
        a_flat = a.flatten().tolist()
        a_new = []
        batch = 1
        for s in b.shape[:-1]:
            batch *= s
        vec_len = a.shape[-1]
        for i in range(batch):
            a_new.extend(a_flat[:vec_len])
        a = array(a_new).reshape(a_shape)
    if a.ndim == 1 and b.ndim == 1:
        af = a.flatten().tolist()
        bf = b.flatten().tolist()
        la, lb = len(af), len(bf)
        if la not in (2, 3) or lb not in (2, 3):
            raise ValueError("incompatible vector sizes for cross product")
        if la == 2:
            af = [af[0], af[1], 0.0]
        if lb == 2:
            bf = [bf[0], bf[1], 0.0]
        cx = af[1]*bf[2] - af[2]*bf[1]
        cy = af[2]*bf[0] - af[0]*bf[2]
        cz = af[0]*bf[1] - af[1]*bf[0]
        if la == 2 and lb == 2:
            return array(cz)
        return array([cx, cy, cz])
    if a.ndim >= 2 and b.ndim >= 2:
        a_batch = a.shape[:-1]
        b_batch = b.shape[:-1]
        try:
            out_batch = broadcast_shapes(a_batch, b_batch)
        except Exception:
            out_batch = a_batch
        a_bc = broadcast_to(a, tuple(out_batch) + (a.shape[-1],))
        b_bc = broadcast_to(b, tuple(out_batch) + (b.shape[-1],))
        la = a_bc.shape[-1]
        lb = b_bc.shape[-1]
        batch_size = 1
        for s in out_batch:
            batch_size *= s
        af = a_bc.flatten().tolist()
        bf = b_bc.flatten().tolist()
        results = []
        for i in range(batch_size):
            ai = af[i * la:(i + 1) * la]
            bi = bf[i * lb:(i + 1) * lb]
            if la == 2:
                ai = [ai[0], ai[1], 0.0]
            if lb == 2:
                bi = [bi[0], bi[1], 0.0]
            cx = ai[1]*bi[2] - ai[2]*bi[1]
            cy = ai[2]*bi[0] - ai[0]*bi[2]
            cz = ai[0]*bi[1] - ai[1]*bi[0]
            if la == 2 and lb == 2:
                results.append(cz)
            else:
                results.extend([cx, cy, cz])
        if la == 2 and lb == 2:
            result = array(results).reshape(out_batch)
        else:
            result = array(results).reshape(list(out_batch) + [3])
        if axisc != -1 and axisc != result.ndim - 1 and result.ndim > 1 and not (la == 2 and lb == 2):
            result = moveaxis(result, -1, axisc)
        return result


def tensordot(a, b, axes=2):
    """Compute tensor dot product along specified axes."""
    from ._manipulation import _transpose_with_axes
    from _numpy_native import dot
    a = asarray(a) if not isinstance(a, ndarray) else a
    b = asarray(b) if not isinstance(b, ndarray) else b
    if isinstance(axes, int):
        axes_a = list(range(a.ndim - axes, a.ndim))
        axes_b = list(range(0, axes))
    else:
        axes_a = list(axes[0]) if not isinstance(axes[0], int) else [axes[0]]
        axes_b = list(axes[1]) if not isinstance(axes[1], int) else [axes[1]]
    na = a.ndim
    nb = b.ndim
    axes_a = [ax if ax >= 0 else ax + na for ax in axes_a]
    axes_b = [ax if ax >= 0 else ax + nb for ax in axes_b]
    free_a = [i for i in range(na) if i not in axes_a]
    free_b = [i for i in range(nb) if i not in axes_b]
    perm_a = free_a + axes_a
    perm_b = axes_b + free_b
    at = _transpose_with_axes(a, perm_a)
    bt = _transpose_with_axes(b, perm_b)
    free_a_shape = [a.shape[i] for i in free_a]
    free_b_shape = [b.shape[i] for i in free_b]
    contract_size = 1
    for ax in axes_a:
        contract_size *= a.shape[ax]
    rows = 1
    for s in free_a_shape:
        rows *= s
    cols = 1
    for s in free_b_shape:
        cols *= s
    at2 = at.reshape([rows, contract_size])
    bt2 = bt.reshape([contract_size, cols])
    result = dot(at2, bt2)
    out_shape = free_a_shape + free_b_shape
    if len(out_shape) == 0:
        # 0-d result — return scalar with shape ()
        val = float(result.flatten()[0])
        return array(val)
    return result.reshape(out_shape)


def inner(a, b):
    """Inner product of two arrays."""
    from _numpy_native import dot
    a = asarray(a)
    b = asarray(b)
    if a.ndim <= 1 and b.ndim <= 1:
        return dot(a, b)
    if a.ndim == 2 and b.ndim == 2:
        return dot(a, b.T)
    return tensordot(a, b, axes=([-1], [-1]))


def kron(a, b):
    """Kronecker product of two arrays."""
    a = asarray(a)
    b = asarray(b)
    if a.ndim == 1:
        a = a.reshape((1, a.size))
    if b.ndim == 1:
        b = b.reshape((1, b.size))
    ar, ac = a.shape[0], a.shape[1]
    br, bc = b.shape[0], b.shape[1]
    rows = []
    for i in range(ar):
        for bi in range(br):
            row = []
            for j in range(ac):
                for bj in range(bc):
                    row.append(a[i][j] * b[bi][bj])
            rows.append(row)
    return array(rows)


def matmul(x1, x2):
    """Matrix product of two arrays (same as the @ operator)."""
    # Check for __array_ufunc__ dispatch on inputs
    for arg in (x1, x2):
        if not isinstance(arg, ndarray) and not isinstance(arg, (int, float, bool, complex)):
            au = getattr(type(arg), '__array_ufunc__', NotImplemented)
            if au is not NotImplemented and au is not None:
                result = arg.__array_ufunc__(matmul, '__call__', x1, x2)
                if result is not NotImplemented:
                    return result
    from _numpy_native import dot
    x1 = asarray(x1)
    x2 = asarray(x2)
    return dot(x1, x2)


def vdot(a, b):
    """Conjugate dot product of two arrays (flattened)."""
    from _numpy_native import dot
    a = asarray(a).flatten()
    b = asarray(b).flatten()
    return dot(a, b)


def einsum(*operands, **kwargs):
    """Evaluates the Einstein summation convention on the operands."""
    out = kwargs.pop('out', None)
    dtype = kwargs.pop('dtype', None)
    order = kwargs.pop('order', 'K')
    casting = kwargs.pop('casting', 'safe')
    optimize = kwargs.pop('optimize', False)
    if len(operands) == 0:
        raise ValueError("No input operands")
    # Handle interleaved subscript format: einsum(op0, subs0, op1, subs1, ..., [output_subs])
    if not isinstance(operands[0], str):
        # Interleaved format
        arrays = []
        sub_parts = []
        # Map integer subscript labels to letters
        label_map = {}
        label_counter = [0]
        def _int_to_letter(n):
            if n == Ellipsis:
                return '...'
            if n not in label_map:
                label_map[n] = chr(ord('a') + label_counter[0])
                label_counter[0] += 1
            return label_map[n]
        i = 0
        while i < len(operands):
            arr = operands[i]
            i += 1
            if i >= len(operands):
                # Last element is output subscripts
                if isinstance(arr, list):
                    out_sub = ''.join(_int_to_letter(x) for x in arr)
                    sub_parts.append('->' + out_sub)
                break
            subs = operands[i]
            i += 1
            # Convert ndarray subscripts to a list
            if isinstance(subs, ndarray):
                subs = subs.tolist()
                if not isinstance(subs, list):
                    subs = [subs]
            if isinstance(subs, list):
                sub_str = ''.join(_int_to_letter(x) for x in subs)
            else:
                sub_str = str(subs)
            arrays.append(asarray(arr))
            sub_parts.append(sub_str)
            # Check if next element is a list (output subscripts) with no following operand
            if i < len(operands) and isinstance(operands[i], list) and (i + 1 >= len(operands) or isinstance(operands[i + 1], list)):
                out_sub = ''.join(_int_to_letter(x) for x in operands[i])
                sub_parts.append('->' + out_sub)
                i += 1
        subscripts = ','.join(p for p in sub_parts if not p.startswith('->'))
        out_part = [p for p in sub_parts if p.startswith('->')]
        if out_part:
            subscripts += out_part[0]
    else:
        if len(operands) < 1:
            raise ValueError("No input operands")
        subscripts = operands[0]
        arrays = list(operands[1:])
    if not isinstance(subscripts, str):
        raise TypeError("subscripts must be a string")
    if len(arrays) == 0 and subscripts == '':
        raise ValueError("No input operands")
    # Validate and convert all operands to arrays
    for i in range(len(arrays)):
        if not isinstance(arrays[i], ndarray):
            arrays[i] = asarray(arrays[i])
    # When dtype is specified, cast all inputs to that dtype for computation
    if dtype is not None:
        import numpy as _np
        _compute_dt = str(_np.dtype(dtype))
        for i in range(len(arrays)):
            if str(arrays[i].dtype) != _compute_dt:
                arrays[i] = arrays[i].astype(_compute_dt)
    # Upcast all arrays to a common dtype to avoid Rust type mismatch errors
    elif len(arrays) > 1:
        import numpy as _np
        common_dt = str(arrays[0].dtype)
        for a in arrays[1:]:
            common_dt = str(_np.promote_types(common_dt, str(a.dtype)))
        for i in range(len(arrays)):
            if str(arrays[i].dtype) != common_dt:
                arrays[i] = arrays[i].astype(common_dt)
    # Handle implicit output subscripts
    if '->' not in subscripts:
        input_subs = subscripts.replace(' ', '')
        # Handle ellipsis
        parts = input_subs.split(',')
        from collections import Counter
        counts = Counter()
        has_ellipsis = '...' in input_subs
        for p in parts:
            p_clean = p.replace('...', '')
            counts.update(p_clean)
        output = ''.join(sorted(c for c, n in counts.items() if n == 1))
        if has_ellipsis:
            output = '...' + output
        subscripts = input_subs + '->' + output
    # Expand ellipsis to explicit indices before calling native einsum
    if '...' in subscripts:
        parts = subscripts.split('->')
        input_part = parts[0]
        output_part = parts[1] if len(parts) > 1 else None
        input_terms = input_part.split(',')
        # Find used indices
        used_indices = set()
        for t in input_terms:
            used_indices.update(c for c in t if c.isalpha())
        if output_part:
            used_indices.update(c for c in output_part if c.isalpha())
        # Find available indices for ellipsis expansion
        all_letters = [chr(c) for c in range(ord('A'), ord('Z')+1)] + [chr(c) for c in range(ord('a'), ord('z')+1)]
        avail = [c for c in all_letters if c not in used_indices]
        # Determine ellipsis dimensions for each operand
        expanded_terms = []
        ellipsis_ndim = 0
        for idx_t, t in enumerate(input_terms):
            if '...' in t:
                explicit_count = len(t.replace('...', ''))
                if idx_t < len(arrays):
                    arr_ndim = arrays[idx_t].ndim
                    this_ellipsis = arr_ndim - explicit_count
                    if this_ellipsis < 0:
                        this_ellipsis = 0
                    ellipsis_ndim = _builtin_max(ellipsis_ndim, this_ellipsis)
        # Now expand
        ellipsis_labels = avail[:ellipsis_ndim]
        new_input_terms = []
        for idx_t, t in enumerate(input_terms):
            if '...' in t:
                explicit_count = len(t.replace('...', ''))
                if idx_t < len(arrays):
                    arr_ndim = arrays[idx_t].ndim
                    this_ellipsis = arr_ndim - explicit_count
                else:
                    this_ellipsis = ellipsis_ndim
                if this_ellipsis < 0:
                    this_ellipsis = 0
                # Use right-aligned ellipsis labels for broadcasting
                labels = ellipsis_labels[ellipsis_ndim - this_ellipsis:]
                new_input_terms.append(t.replace('...', ''.join(labels)))
            else:
                new_input_terms.append(t)
        new_input = ','.join(new_input_terms)
        if output_part is not None:
            if '...' in output_part:
                new_output = output_part.replace('...', ''.join(ellipsis_labels))
            else:
                new_output = output_part
            subscripts = new_input + '->' + new_output
        else:
            subscripts = new_input
    try:
        result = _native.einsum(subscripts, *arrays)
    except TypeError as e:
        # Handle scalar inputs / type mismatches that native einsum can't handle
        err_msg = str(e)
        if "Expected type" in err_msg:
            # Upcast everything to float64 as a safe fallback
            new_arrays = [asarray(a).astype('float64') if not isinstance(a, ndarray) else a.astype('float64') for a in arrays]
            result = _native.einsum(subscripts, *new_arrays)
        else:
            raise
    if dtype is not None and isinstance(result, ndarray):
        import numpy as _np
        _out_dt = str(_np.dtype(dtype))
        if str(result.dtype) != _out_dt:
            result = result.astype(_out_dt)
    if out is not None:
        if isinstance(out, ndarray):
            flat_r = result.flatten()
            for i in range(flat_r.size):
                out.flat[i] = flat_r[i]
            return out
    return result


# ---------------------------------------------------------------------------
# Bit manipulation
# ---------------------------------------------------------------------------

def packbits(a, axis=None, bitorder='big'):
    """Pack a binary-valued array into uint8."""
    if not isinstance(a, ndarray):
        a = asarray(a)
    vals = a.flatten().tolist()
    if bitorder == 'little':
        result = []
        for i in range(0, len(vals), 8):
            chunk = vals[i:i+8]
            byte = 0
            for j in range(len(chunk)):
                if int(chunk[j]):
                    byte |= (1 << j)
            result.append(byte)
        return array(result)
    else:
        result = []
        for i in range(0, len(vals), 8):
            chunk = vals[i:i+8]
            byte = 0
            for j in range(len(chunk)):
                if int(chunk[j]):
                    byte |= (1 << (7 - j))
            result.append(byte)
        return array(result)


def unpackbits(a, axis=None, count=None, bitorder='big'):
    """Unpack elements of a uint8 array into a binary-valued output array."""
    if not isinstance(a, ndarray):
        a = asarray(a)
    vals = a.flatten().tolist()
    result = []
    for v in vals:
        byte = int(v)
        if bitorder == 'little':
            for j in range(8):
                result.append((byte >> j) & 1)
        else:
            for j in range(7, -1, -1):
                result.append((byte >> j) & 1)
    if count is not None:
        count = int(count)
        if count < len(result):
            result = result[:count]
        else:
            result = result + [0] * (count - len(result))
    return array(result)


# ---------------------------------------------------------------------------
# Misc numeric
# ---------------------------------------------------------------------------

def binary_repr(num, width=None):
    num = int(num)  # Convert numpy scalars to plain Python int to avoid overflow
    if num >= 0:
        s = bin(num)[2:]
        if width is not None:
            s = s.zfill(width)
        return s
    else:
        if width is not None:
            s = bin(2**width + num)[2:]
            return s.zfill(width)
        else:
            return '-' + bin(-num)[2:]


def base_repr(number, base=2, padding=0):
    if base < 2 or base > 36:
        raise ValueError("Bases greater than 36 not handled in base_repr.")
    if number == 0:
        return "0" * (padding + 1)
    digits = []
    n = __import__("builtins").abs(number)
    while n:
        digits.append(str(n % base) if n % base < 10 else chr(ord('A') + n % base - 10))
        n //= base
    s = "".join(reversed(digits))
    s = "0" * padding + s
    if number < 0:
        s = "-" + s
    return s


def frompyfunc(func, nin, nout):
    """Takes an arbitrary Python function and returns a NumPy ufunc-like object."""
    from ._manipulation import vectorize
    return vectorize(func)


# ---------------------------------------------------------------------------
# Indexing helpers
# ---------------------------------------------------------------------------

def take_along_axis(arr, indices, axis):
    """Take values from the input array by matching 1-d index and data slices along the given axis."""
    from ._manipulation import moveaxis
    arr = asarray(arr)
    indices = asarray(indices)
    if arr.ndim == 1:
        result = []
        for i in range(indices.size):
            result.append(arr[int(indices[i])])
        return array(result)
    if arr.ndim == 2:
        if axis == 0:
            rows = []
            for j in range(arr.shape[1]):
                col = []
                for i in range(indices.shape[0]):
                    col.append(arr[int(indices[i][j])][j])
                rows.append(col)
            result = []
            for i in range(indices.shape[0]):
                row = [rows[j][i] for j in range(arr.shape[1])]
                result.append(row)
            return array(result)
        else:
            rows = []
            for i in range(arr.shape[0]):
                row = []
                for j in range(indices.shape[1]):
                    row.append(arr[i][int(indices[i][j])])
                rows.append(row)
            return array(rows)
    if axis < 0:
        axis = arr.ndim + axis
    arr_m = moveaxis(arr, axis, -1)
    ind_m = moveaxis(indices, axis, -1)
    out_shape = ind_m.shape
    n_axis = arr_m.shape[-1]
    lead = 1
    for s in arr_m.shape[:-1]:
        lead *= s
    arr_flat = arr_m.reshape((lead, n_axis))
    ind_flat = ind_m.reshape((lead, ind_m.shape[-1]))
    arr_list = arr_flat.tolist()
    ind_list = ind_flat.tolist()
    result = []
    for i in range(lead):
        row = arr_list[i]
        idxs = ind_list[i]
        result.append([row[int(j)] for j in idxs])
    result_arr = array(result).reshape(out_shape)
    return moveaxis(result_arr, -1, axis)


def put_along_axis(arr, indices, values, axis):
    """Put values into the destination array by matching 1-d index and data slices along the given axis."""
    from ._manipulation import moveaxis
    arr = asarray(arr)
    indices = asarray(indices)
    values = asarray(values)
    if arr.ndim == 1:
        result = [arr[i] for i in range(arr.size)]
        vals_flat = values.flatten()
        for i in range(indices.size):
            result[int(indices[i])] = vals_flat[i % vals_flat.size]
        return array(result)
    if arr.ndim == 2 and axis == 1:
        rows = []
        for i in range(arr.shape[0]):
            row = [arr[i][j] for j in range(arr.shape[1])]
            for j in range(indices.shape[1]):
                idx = int(indices[i][j])
                row[idx] = values[i][j] if values.ndim == 2 else values[j]
            rows.append(row)
        return array(rows)
    if axis < 0:
        axis = arr.ndim + axis
    arr_m = moveaxis(arr, axis, -1)
    ind_m = moveaxis(indices, axis, -1)
    val_m = moveaxis(values, axis, -1)
    out_shape = arr_m.shape
    lead = 1
    for s in arr_m.shape[:-1]:
        lead *= s
    n_axis = arr_m.shape[-1]
    arr_flat = arr_m.reshape((lead, n_axis)).tolist()
    ind_flat = ind_m.reshape((lead, ind_m.shape[-1])).tolist()
    val_flat = val_m.reshape((lead, val_m.shape[-1])).tolist()
    for i in range(lead):
        for j in range(len(ind_flat[i])):
            arr_flat[i][int(ind_flat[i][j])] = val_flat[i][j]
    result = array(arr_flat).reshape(out_shape)
    return moveaxis(result, -1, axis)


# ---------------------------------------------------------------------------
# matrix class
# ---------------------------------------------------------------------------

def _has_complex(result):
    """Check if any element in result is complex."""
    for r in result:
        if isinstance(r, complex):
            return True
    return False


class matrix:
    """Simplified matrix class (deprecated in NumPy, but still used)."""
    def __init__(self, data, dtype=None, copy=True):
        from ._manipulation import atleast_2d
        if isinstance(data, str):
            rows = data.split(";")
            parsed = []
            for row in rows:
                parsed.append([float(x) for x in row.strip().split()])
            self.A = array(parsed)
        else:
            self.A = atleast_2d(asarray(data))
        if dtype is not None:
            self.A = self.A.astype(str(dtype))

    @property
    def T(self):
        return matrix(self.A.T)

    @property
    def I(self):
        from _numpy_native import linalg as _linalg
        return matrix(_linalg.inv(self.A))

    @property
    def shape(self):
        return self.A.shape

    @property
    def ndim(self):
        return self.A.ndim

    def __mul__(self, other):
        from _numpy_native import dot
        if isinstance(other, matrix):
            return matrix(dot(self.A, other.A))
        return matrix(self.A * asarray(other))

    def __add__(self, other):
        if isinstance(other, matrix):
            return matrix(self.A + other.A)
        return matrix(self.A + asarray(other))

    def __sub__(self, other):
        if isinstance(other, matrix):
            return matrix(self.A - other.A)
        return matrix(self.A - asarray(other))

    def __getitem__(self, key):
        return self.A[key]

    def tolist(self):
        return self.A.tolist()

    def __repr__(self):
        return "matrix({})".format(self.A.tolist())


