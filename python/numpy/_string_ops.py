"""String and char array operations."""
import _numpy_native as _native
from _numpy_native import ndarray
from ._helpers import _ObjectArray
from ._creation import array, asarray
from ._string_bridge import (
    python_string_add,
    python_string_compare,
    native_string_unary,
    normalize_native_string_input,
    python_string_join,
    python_string_items,
    python_string_map,
    python_string_encode,
    python_string_decode,
    python_string_pad,
    python_string_partition,
    python_string_predicate,
    python_string_replace,
    python_string_rsplit,
    python_string_search,
    python_string_split,
    python_string_splitlines,
    python_string_strip,
    python_string_mod,
    python_string_repeat,
    python_string_transform,
    python_string_unicode_predicate,
    python_string_zfill,
)

__all__ = ['char']


def _strip_whitespace(s):
    """Strip trailing whitespace/nulls from a string for chararray comparisons."""
    if isinstance(s, bytes):
        return s.rstrip(b'\x00 ')
    return s.rstrip('\x00 ')


def _to_items(a):
    """Convert array-like to flat list of items."""
    if isinstance(a, chararray):
        return a._arr.flatten().tolist()
    if isinstance(a, ndarray):
        return a.flatten().tolist()
    if isinstance(a, _ObjectArray):
        return a._data
    if isinstance(a, (str, bytes)):
        return [a]
    if isinstance(a, (list, tuple)):
        result = []
        for item in a:
            if isinstance(item, (list, tuple)):
                result.extend(item)
            else:
                result.append(item)
        return result
    return [a]


def _to_str(v):
    """Convert a value to string, decoding bytes if needed."""
    if isinstance(v, bytes):
        return v.decode('latin-1')
    return str(v)


def _coerce_native_string_array(a):
    """Delegate native string input normalization to the shared bridge."""
    return normalize_native_string_input(a)[0]


def _native_string_output(a, native_op, *args):
    arr = _coerce_native_string_array(a)
    return native_op(arr, *args)


def _native_bool_output(a, native_op, *args):
    arr = _coerce_native_string_array(a)
    return native_op(arr, *args)


def _native_int_output(a, native_op, *args):
    arr = _coerce_native_string_array(a)
    return native_op(arr, *args)


class chararray:
    """Character array — a wrapper around ndarray that provides string methods
    and strips trailing whitespace in comparisons."""

    def __init__(self, shape, itemsize=None, unicode=None, buffer=None,
                 offset=0, strides=None, order='C'):
        import numpy as _np
        if isinstance(shape, int):
            shape = (shape,)
        if unicode:
            dt = 'U' + str(itemsize or 1)
        else:
            dt = 'S' + str(itemsize or 1)
        self._arr = _np.zeros(shape, dtype=dt)

    @classmethod
    def _from_array(cls, arr):
        """Wrap an existing ndarray as a chararray (no copy)."""
        obj = object.__new__(cls)
        obj._arr = arr
        return obj

    # --- ndarray-like properties ---
    @property
    def dtype(self):
        return self._arr.dtype

    @property
    def shape(self):
        return self._arr.shape

    @property
    def ndim(self):
        return self._arr.ndim

    @property
    def size(self):
        return self._arr.size

    @property
    def itemsize(self):
        return self._arr.itemsize

    @property
    def base(self):
        return self._arr

    @property
    def flat(self):
        return self._arr.flat

    @property
    def flags(self):
        return self._arr.flags

    @property
    def strides(self):
        return self._arr.strides

    @property
    def T(self):
        return chararray._from_array(self._arr.T)

    def __len__(self):
        return len(self._arr)

    def __iter__(self):
        for i in range(len(self._arr)):
            v = self._arr[i]
            if isinstance(v, ndarray):
                yield chararray._from_array(v)
            else:
                yield _strip_whitespace(v)

    def __getitem__(self, key):
        result = self._arr[key]
        if isinstance(result, ndarray):
            return chararray._from_array(result)
        if isinstance(result, _ObjectArray):
            return chararray._from_array(result)
        if isinstance(result, (list, tuple)):
            # 2D ObjectArray returns a list for row access
            import numpy as _np
            arr = _np.array(result)
            if isinstance(arr, ndarray):
                return chararray._from_array(arr)
            return chararray._from_array(arr)
        # Scalar: strip trailing whitespace
        return _strip_whitespace(result)

    def __setitem__(self, key, value):
        self._arr[key] = value

    def __repr__(self):
        return "chararray(" + repr(self._arr) + ")"

    def __str__(self):
        return str(self._arr)

    def __bool__(self):
        return bool(self._arr)

    # --- Comparison operators (strip trailing whitespace) ---
    def _compare(self, other, op):
        return python_string_compare(self, other, op, strip=True)

    def __eq__(self, other):
        return self._compare(other, lambda a, b: a == b)

    def __ne__(self, other):
        return self._compare(other, lambda a, b: a != b)

    def __ge__(self, other):
        return self._compare(other, lambda a, b: a >= b)

    def __le__(self, other):
        return self._compare(other, lambda a, b: a <= b)

    def __gt__(self, other):
        return self._compare(other, lambda a, b: a > b)

    def __lt__(self, other):
        return self._compare(other, lambda a, b: a < b)

    # --- Arithmetic ---
    def __add__(self, other):
        return python_string_add(self, other, wrap_chararray=True)

    def __radd__(self, other):
        return python_string_add(other, self, wrap_chararray=True)

    def __mul__(self, other):
        return python_string_repeat(self, other, wrap_chararray=True)

    def __rmul__(self, other):
        return self.__mul__(other)

    def __mod__(self, other):
        return python_string_mod(self, other, wrap_chararray=True)

    def __rmod__(self, other):
        raise TypeError("unsupported operand type(s) for %: '{}' and 'chararray'".format(
            type(other).__name__))

    def __format__(self, fmt):
        if fmt == '':
            return str(self)
        if fmt == 'r':
            return repr(self)
        return str(self)

    # --- Array methods ---
    def flatten(self):
        return chararray._from_array(self._arr.flatten())

    def ravel(self):
        return chararray._from_array(self._arr.ravel())

    def reshape(self, *shape):
        if len(shape) == 1 and isinstance(shape[0], (list, tuple)):
            shape = tuple(shape[0])
        return chararray._from_array(self._arr.reshape(*shape))

    def copy(self):
        return chararray._from_array(self._arr.copy())

    def tolist(self):
        return self._arr.tolist()

    def tobytes(self):
        return self._arr.tobytes()

    def astype(self, dtype):
        return chararray._from_array(self._arr.astype(dtype))

    def view(self, dtype=None):
        if dtype is chararray or dtype is type(self):
            return self
        return self._arr.view(dtype)

    def argsort(self, axis=-1, kind=None, order=None, stable=False):
        return self._arr.argsort(axis=axis, stable=stable)

    # --- String methods ---
    def upper(self):
        return native_string_unary(self, _native.char_upper, wrap_chararray=True)

    def lower(self):
        return native_string_unary(self, _native.char_lower, wrap_chararray=True)

    def capitalize(self):
        return native_string_unary(self, _native.char_capitalize, wrap_chararray=True)

    def strip(self, chars=None):
        return python_string_strip(self, chars, wrap_chararray=True)

    def lstrip(self, chars=None):
        return python_string_strip(
            self,
            chars,
            method_name="lstrip",
            wrap_chararray=True,
        )

    def rstrip(self, chars=None):
        return python_string_strip(
            self,
            chars,
            method_name="rstrip",
            wrap_chararray=True,
        )

    def title(self):
        return python_string_transform(self, "title", wrap_chararray=True)

    def swapcase(self):
        return python_string_transform(self, "swapcase", wrap_chararray=True)

    def center(self, width, fillchar=None):
        return python_string_pad(
            self,
            width,
            "center",
            fillchar=fillchar,
            wrap_chararray=True,
        )

    def ljust(self, width, fillchar=None):
        return python_string_pad(
            self,
            width,
            "ljust",
            fillchar=fillchar,
            wrap_chararray=True,
        )

    def rjust(self, width, fillchar=None):
        return python_string_pad(
            self,
            width,
            "rjust",
            fillchar=fillchar,
            wrap_chararray=True,
        )

    def zfill(self, width):
        return python_string_zfill(self, width, wrap_chararray=True)

    def replace(self, old, new, count=None):
        return python_string_replace(
            self,
            old,
            new,
            count=count,
            wrap_chararray=True,
        )

    def startswith(self, prefix, start=0, end=None):
        return python_string_search(self, prefix, "startswith", start=start, end=end).astype(
            "bool"
        )

    def endswith(self, suffix, start=0, end=None):
        return python_string_search(self, suffix, "endswith", start=start, end=end).astype(
            "bool"
        )

    def find(self, sub, start=0, end=None):
        return python_string_search(self, sub, "find", start=start, end=end)

    def rfind(self, sub, start=0, end=None):
        return python_string_search(self, sub, "rfind", start=start, end=end)

    def index(self, sub, start=0, end=None):
        return python_string_search(self, sub, "index", start=start, end=end)

    def rindex(self, sub, start=0, end=None):
        return python_string_search(self, sub, "rindex", start=start, end=end)

    def count(self, sub, start=0, end=None):
        return python_string_search(self, sub, "count", start=start, end=end)

    def expandtabs(self, tabsize=8):
        return python_string_transform(
            self,
            "expandtabs",
            tabsize,
            wrap_chararray=True,
        )

    def isalnum(self):
        return python_string_predicate(self, lambda item: item.isalnum())

    def isalpha(self):
        return python_string_predicate(self, lambda item: item.isalpha())

    def isdigit(self):
        return python_string_predicate(self, lambda item: item.isdigit())

    def islower(self):
        return python_string_predicate(self, lambda item: item.islower())

    def isupper(self):
        return python_string_predicate(self, lambda item: item.isupper())

    def isspace(self):
        return python_string_predicate(self, lambda item: item.isspace())

    def istitle(self):
        return python_string_predicate(self, lambda item: item.istitle())

    def isnumeric(self):
        return python_string_unicode_predicate(self, "isnumeric")

    def isdecimal(self):
        return python_string_unicode_predicate(self, "isdecimal")

    def split(self, sep=None, maxsplit=-1):
        return python_string_split(self, sep=sep, maxsplit=maxsplit)

    def rsplit(self, sep=None, maxsplit=-1):
        return python_string_rsplit(self, sep=sep, maxsplit=maxsplit)

    def splitlines(self):
        return python_string_splitlines(self)

    def partition(self, sep):
        return python_string_partition(self, sep, wrap_chararray=True)

    def rpartition(self, sep):
        return python_string_partition(
            self,
            sep,
            method_name="rpartition",
            wrap_chararray=True,
        )

    def encode(self, encoding='utf-8', errors='strict'):
        return python_string_encode(self, encoding=encoding, errors=errors)

    def decode(self, encoding='utf-8', errors='strict'):
        return python_string_decode(self, encoding=encoding, errors=errors)

    def join(self, seq):
        return python_string_join(seq, self, wrap_chararray=True)


def _is_shared_bridge_array_like(value):
    return isinstance(value, (ndarray, _ObjectArray, list, tuple, chararray))


def _contains_bytes_value(value):
    if isinstance(value, bytes):
        return True
    if _is_shared_bridge_array_like(value):
        items, _ = python_string_items(value)
        return any(isinstance(item, bytes) for item in items)
    return False


def _should_use_shared_string_bridge(a, *operands):
    if isinstance(a, chararray) or _contains_bytes_value(a):
        return True
    return any(
        _is_shared_bridge_array_like(operand) or _contains_bytes_value(operand)
        for operand in operands
    )


class _char_mod:
    # Expose chararray class
    chararray = chararray

    @staticmethod
    def array(obj, itemsize=None, copy=True, unicode=None, order=None):
        """Create a chararray from obj."""
        import numpy as _np
        if isinstance(obj, chararray):
            if copy:
                return obj.copy()
            return obj
        if isinstance(obj, ndarray):
            arr = obj.copy() if copy else obj
            return chararray._from_array(arr)
        if isinstance(obj, (str, bytes)):
            arr = _np.array([obj])
            return chararray._from_array(arr)
        # List/tuple/other
        arr = _np.array(obj)
        if copy:
            arr = arr.copy()
        return chararray._from_array(arr)

    @staticmethod
    def asarray(obj, itemsize=None, unicode=None, order=None):
        """Create a chararray from obj (no copy if possible)."""
        return _char_mod.array(obj, itemsize=itemsize, copy=False, unicode=unicode, order=order)

    @staticmethod
    def compare_chararrays(a1, a2, cmp, rstrip):
        """Compare two string arrays element-wise using the given comparison operator."""
        ops = {'<': lambda x, y: x < y, '<=': lambda x, y: x <= y,
               '==': lambda x, y: x == y, '>=': lambda x, y: x >= y,
               '>': lambda x, y: x > y, '!=': lambda x, y: x != y}
        op = ops.get(cmp)
        if op is None:
            raise ValueError(f"Invalid comparison: {cmp!r}")
        return python_string_compare(a1, a2, op, strip=rstrip)

    @staticmethod
    def equal(a, b):
        """Element-wise comparison for equality, stripping trailing whitespace."""
        return python_string_compare(a, b, lambda x, y: x == y, strip=True)

    @staticmethod
    def not_equal(a, b):
        return python_string_compare(a, b, lambda x, y: x != y, strip=True)

    @staticmethod
    def greater(a, b):
        return python_string_compare(a, b, lambda x, y: x > y, strip=True)

    @staticmethod
    def greater_equal(a, b):
        return python_string_compare(a, b, lambda x, y: x >= y, strip=True)

    @staticmethod
    def less(a, b):
        return python_string_compare(a, b, lambda x, y: x < y, strip=True)

    @staticmethod
    def less_equal(a, b):
        return python_string_compare(a, b, lambda x, y: x <= y, strip=True)

    @staticmethod
    def upper(a):
        return native_string_unary(a, _native.char_upper)

    @staticmethod
    def lower(a):
        return native_string_unary(a, _native.char_lower)

    @staticmethod
    def capitalize(a):
        return native_string_unary(a, _native.char_capitalize)

    @staticmethod
    def strip(a, chars=None):
        if (
            chars is None
            and isinstance(a, (ndarray, _ObjectArray))
            and str(getattr(a, "dtype", "")) == "object"
        ):
            from ._core._exceptions import UFuncTypeError

            raise UFuncTypeError("strip", "object arrays are not supported")
        if chars is None and not isinstance(a, chararray):
            return _native_string_output(a, _native.char_strip)
        return python_string_strip(a, chars)

    @staticmethod
    def lstrip(a, chars=None):
        return python_string_strip(a, chars, method_name="lstrip")

    @staticmethod
    def rstrip(a, chars=None):
        return python_string_strip(a, chars, method_name="rstrip")

    @staticmethod
    def str_len(a):
        return _native_int_output(a, _native.char_str_len)

    @staticmethod
    def startswith(a, prefix, start=0, end=None):
        if start == 0 and end is None and not _should_use_shared_string_bridge(a, prefix):
            return _native_bool_output(a, _native.char_startswith, prefix)
        return python_string_search(a, prefix, "startswith", start=start, end=end).astype("bool")

    @staticmethod
    def endswith(a, suffix, start=0, end=None):
        if start == 0 and end is None and not _should_use_shared_string_bridge(a, suffix):
            return _native_bool_output(a, _native.char_endswith, suffix)
        return python_string_search(a, suffix, "endswith", start=start, end=end).astype("bool")

    @staticmethod
    def replace(a, old, new, count=None):
        if count is None and not _should_use_shared_string_bridge(a, old, new):
            return _native_string_output(a, _native.char_replace, old, new)
        return python_string_replace(a, old, new, count=count)

    @staticmethod
    def split(a, sep=None, maxsplit=-1):
        """Split each element in a around sep."""
        return python_string_split(a, sep=sep, maxsplit=maxsplit)

    @staticmethod
    def rsplit(a, sep=None, maxsplit=-1):
        return python_string_rsplit(a, sep=sep, maxsplit=maxsplit)

    @staticmethod
    def splitlines(a):
        return python_string_splitlines(a)

    @staticmethod
    def join(sep, a):
        """Join strings in a with separator sep, element-wise."""
        return python_string_join(a, sep)

    @staticmethod
    def find(a, sub, start=0, end=None):
        """Find first occurrence of sub in each element of a."""
        return python_string_search(a, sub, "find", start=start, end=end)

    @staticmethod
    def rfind(a, sub, start=0, end=None):
        return python_string_search(a, sub, "rfind", start=start, end=end)

    @staticmethod
    def index(a, sub, start=0, end=None):
        if isinstance(a, (str, bytes)):
            import numpy as _np
            if end is None:
                return _np.array(a.index(sub, start))
            return _np.array(a.index(sub, start, end))
        return python_string_search(a, sub, "index", start=start, end=end)

    @staticmethod
    def rindex(a, sub, start=0, end=None):
        if isinstance(a, (str, bytes)):
            import numpy as _np
            if end is None:
                return _np.array(a.rindex(sub, start))
            return _np.array(a.rindex(sub, start, end))
        return python_string_search(a, sub, "rindex", start=start, end=end)

    @staticmethod
    def count(a, sub, start=0, end=None):
        """Count non-overlapping occurrences of sub in each element of a."""
        return python_string_search(a, sub, "count", start=start, end=end)

    @staticmethod
    def add(a, b):
        """Element-wise string concatenation."""
        if isinstance(a, chararray) or isinstance(b, chararray):
            if not isinstance(a, chararray):
                a = _char_mod.array(a)
            return a + b
        return python_string_add(a, b)

    @staticmethod
    def multiply(a, i):
        """Element-wise string repetition."""
        if isinstance(a, chararray):
            return a * i
        return python_string_repeat(a, i)

    @staticmethod
    def mod(format_str, values):
        """Element-wise string formatting."""
        import numpy as _np
        if isinstance(format_str, chararray):
            return format_str % values
        if isinstance(format_str, (str, bytes)):
            if isinstance(values, ndarray):
                items = values.flatten().tolist()
                result = [format_str % v for v in items]
                out = _np.array(result)
                if len(values.shape) > 1:
                    out = out.reshape(values.shape)
                return chararray._from_array(out)
            return format_str % values
        return python_string_mod(format_str, values)

    @staticmethod
    def _to_str_list(a):
        """Convert input to a flat list of strings."""
        if isinstance(a, chararray):
            return [str(x) for x in a._arr.flatten().tolist()]
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
        return python_string_pad(a, width, "center", fillchar=fillchar)

    @staticmethod
    def ljust(a, width, fillchar=' '):
        """Left-justify each string element in a to width."""
        return python_string_pad(a, width, "ljust", fillchar=fillchar)

    @staticmethod
    def rjust(a, width, fillchar=' '):
        """Right-justify each string element in a to width."""
        return python_string_pad(a, width, "rjust", fillchar=fillchar)

    @staticmethod
    def zfill(a, width):
        return python_string_zfill(a, width)

    @staticmethod
    def title(a):
        return python_string_transform(a, "title")

    @staticmethod
    def swapcase(a):
        return python_string_transform(a, "swapcase")

    @staticmethod
    def isalpha(a):
        return python_string_predicate(a, lambda item: item.isalpha())

    @staticmethod
    def isdigit(a):
        return python_string_predicate(a, lambda item: item.isdigit())

    @staticmethod
    def isalnum(a):
        return python_string_predicate(a, lambda item: item.isalnum())

    @staticmethod
    def isnumeric(a):
        return python_string_unicode_predicate(a, "isnumeric")

    @staticmethod
    def isupper(a):
        return python_string_predicate(a, lambda item: item.isupper())

    @staticmethod
    def islower(a):
        return python_string_predicate(a, lambda item: item.islower())

    @staticmethod
    def isspace(a):
        return python_string_predicate(a, lambda item: item.isspace())

    @staticmethod
    def istitle(a):
        return python_string_predicate(a, lambda item: item.istitle())

    @staticmethod
    def isdecimal(a):
        return python_string_unicode_predicate(a, "isdecimal")

    @staticmethod
    def expandtabs(a, tabsize=8):
        return python_string_transform(a, "expandtabs", tabsize)

    @staticmethod
    def partition(a, sep):
        return python_string_partition(a, sep)

    @staticmethod
    def rpartition(a, sep):
        return python_string_partition(a, sep, method_name="rpartition")

    @staticmethod
    def encode(a, encoding='utf-8', errors='strict'):
        return python_string_encode(a, encoding=encoding, errors=errors)

    @staticmethod
    def decode(a, encoding='utf-8', errors='strict'):
        return python_string_decode(a, encoding=encoding, errors=errors)


char = _char_mod()
