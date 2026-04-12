"""String and char array operations."""
import _numpy_native as _native
from _numpy_native import ndarray
from ._helpers import _ObjectArray
from ._creation import array, asarray
from ._string_bridge import (
    native_string_unary,
    normalize_native_string_input,
    python_string_broadcast,
    python_string_items,
    python_string_map,
    python_string_predicate,
    python_string_search,
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


def _python_string_strip(value, chars=None, *, method_name="strip", wrap_chararray=False):
    if chars is None:
        return python_string_map(
            value,
            lambda item: getattr(str(item), method_name)(),
            wrap_chararray=wrap_chararray,
        )
    items, _ = python_string_items(value)
    chars_values = python_string_broadcast(chars, len(items))
    chars_iter = iter(chars_values)
    return python_string_map(
        value,
        lambda item: getattr(str(item), method_name)(next(chars_iter)),
        wrap_chararray=wrap_chararray,
    )


def _python_string_pad(value, width, method_name, fillchar=" ", *, wrap_chararray=False):
    items, _ = python_string_items(value)
    widths = python_string_broadcast(width, len(items))
    width_iter = iter(widths)
    fill = " " if fillchar is None else fillchar
    return python_string_map(
        value,
        lambda item: getattr(str(item), method_name)(int(next(width_iter)), fill),
        wrap_chararray=wrap_chararray,
    )


def _python_string_replace(value, old, new, count=None, *, wrap_chararray=False):
    items, _ = python_string_items(value)
    olds = python_string_broadcast(old, len(items))
    news = python_string_broadcast(new, len(items))
    old_iter = iter(olds)
    new_iter = iter(news)
    if count is None:
        return python_string_map(
            value,
            lambda item: str(item).replace(next(old_iter), next(new_iter)),
            wrap_chararray=wrap_chararray,
        )
    counts = python_string_broadcast(count, len(items))
    count_iter = iter(counts)
    return python_string_map(
        value,
        lambda item: str(item).replace(
            next(old_iter),
            next(new_iter),
            int(next(count_iter)),
        ),
        wrap_chararray=wrap_chararray,
    )


def _python_string_split(value, sep=None, maxsplit=-1):
    return python_string_map(
        value,
        lambda item: str(item).split(sep, maxsplit),
        result_kind="object",
    )


def _python_string_rsplit(value, sep=None, maxsplit=-1):
    return python_string_map(
        value,
        lambda item: str(item).rsplit(sep, maxsplit),
        result_kind="object",
    )


def _python_string_splitlines(value):
    return python_string_map(
        value,
        lambda item: str(item).splitlines(),
        result_kind="object",
    )


def _python_string_partition(value, sep, *, method_name="partition", wrap_chararray=False):
    items, _ = python_string_items(value)
    seps = python_string_broadcast(sep, len(items))
    sep_iter = iter(seps)
    return python_string_map(
        value,
        lambda item: list(getattr(str(item), method_name)(next(sep_iter))),
        result_kind="object",
        wrap_chararray=wrap_chararray,
        extra_shape=(3,),
    )


def _python_string_transform(value, method_name, *args, wrap_chararray=False):
    return python_string_map(
        value,
        lambda item: getattr(str(item), method_name)(*args),
        wrap_chararray=wrap_chararray,
    )


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
        import numpy as _np

        def _flat_items(x):
            """Recursively flatten to a list of scalar items."""
            if isinstance(x, chararray):
                return _flat_items(x._arr)
            if isinstance(x, _ObjectArray):
                result = []
                for item in x._data:
                    if isinstance(item, (list, tuple)):
                        result.extend(item)
                    else:
                        result.append(item)
                return result
            if isinstance(x, ndarray):
                raw = x.flatten().tolist()
                out = []
                for item in raw:
                    if isinstance(item, (list, tuple)):
                        out.extend(item)
                    else:
                        out.append(item)
                return out
            if isinstance(x, (list, tuple)):
                out = []
                for item in x:
                    if isinstance(item, (list, tuple)):
                        out.extend(item)
                    else:
                        out.append(item)
                return out
            return [x]

        items_a = _flat_items(self)
        if isinstance(other, (str, bytes)):
            items_b = [other] * len(items_a)
        else:
            items_b = _flat_items(other)
        # Handle broadcasting: if sizes don't match, broadcast
        if len(items_b) == 1 and len(items_a) > 1:
            items_b = items_b * len(items_a)
        elif len(items_a) == 1 and len(items_b) > 1:
            items_a = items_a * len(items_b)
        # Determine output shape from broadcasting
        shape_a = self._arr.shape if hasattr(self._arr, 'shape') else (len(items_a),)
        if isinstance(other, chararray):
            shape_b = other._arr.shape if hasattr(other._arr, 'shape') else (len(items_b),)
        elif isinstance(other, ndarray):
            shape_b = other.shape
        elif isinstance(other, _ObjectArray):
            shape_b = other.shape
        else:
            shape_b = shape_a
        result = []
        for a, b in zip(items_a, items_b):
            sa = _strip_whitespace(a) if isinstance(a, (str, bytes)) else a
            sb = _strip_whitespace(b) if isinstance(b, (str, bytes)) else b
            # Convert between bytes/str for comparison
            if isinstance(sa, bytes) and isinstance(sb, str):
                try:
                    sa = sa.decode('latin-1')
                except Exception:
                    pass
            elif isinstance(sa, str) and isinstance(sb, bytes):
                try:
                    sb = sb.decode('latin-1')
                except Exception:
                    pass
            result.append(op(sa, sb))
        out = _np.array([1.0 if x else 0.0 for x in result]).astype('bool')
        # Use the larger shape for output
        out_shape = shape_a if len(shape_a) >= len(shape_b) else shape_b
        if len(out_shape) > 1 and out.size == 1:
            # scalar result broadcasted
            pass
        elif len(out_shape) > 1:
            try:
                out = out.reshape(out_shape)
            except Exception:
                pass
        return out

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
        import numpy as _np
        items_a = _to_items(self)
        items_b = _to_items(other)
        if len(items_a) == 1 and len(items_b) > 1:
            items_a = items_a * len(items_b)
        elif len(items_b) == 1 and len(items_a) > 1:
            items_b = items_b * len(items_a)
        result = []
        for a, b in zip(items_a, items_b):
            if isinstance(a, bytes) and not isinstance(b, bytes):
                b = str(b).encode('latin-1')
            elif isinstance(b, bytes) and not isinstance(a, bytes):
                a = str(a).encode('latin-1')
            result.append(a + b)
        arr = _np.array(result)
        if len(self._arr.shape) > 1:
            arr = arr.reshape(self._arr.shape)
        return chararray._from_array(arr)

    def __radd__(self, other):
        import numpy as _np
        items_a = _to_items(self)
        if isinstance(other, (str, bytes)):
            items_b = [other] * len(items_a)
        else:
            items_b = _to_items(other)
        if len(items_b) == 1 and len(items_a) > 1:
            items_b = items_b * len(items_a)
        result = []
        for b, a in zip(items_b, items_a):
            if isinstance(a, bytes) and not isinstance(b, bytes):
                b = str(b).encode('latin-1')
            elif isinstance(b, bytes) and not isinstance(a, bytes):
                a = str(a).encode('latin-1')
            result.append(b + a)
        arr = _np.array(result)
        if len(self._arr.shape) > 1:
            arr = arr.reshape(self._arr.shape)
        return chararray._from_array(arr)

    def __mul__(self, other):
        import numpy as _np
        if isinstance(other, (str,)):
            raise ValueError("Can only multiply by integers")
        if not isinstance(other, int):
            if hasattr(other, '__index__'):
                other = other.__index__()
            else:
                raise ValueError("Can only multiply by integers")
        items = _to_items(self)
        result = [s * other for s in items]
        arr = _np.array(result)
        if len(self._arr.shape) > 1:
            arr = arr.reshape(self._arr.shape)
        return chararray._from_array(arr)

    def __rmul__(self, other):
        return self.__mul__(other)

    def __mod__(self, other):
        import numpy as _np
        items = _to_items(self)
        if isinstance(other, ndarray):
            other_items = other.flatten().tolist()
        elif isinstance(other, (list, tuple)):
            # Flatten nested
            other_items = []
            for item in other:
                if isinstance(item, (list, tuple)):
                    other_items.extend(item)
                else:
                    other_items.append(item)
        else:
            other_items = [other] * len(items)
        if len(other_items) == 1 and len(items) > 1:
            other_items = other_items * len(items)
        result = [str(s) % v for s, v in zip(items, other_items)]
        arr = _np.array(result)
        if len(self._arr.shape) > 1:
            arr = arr.reshape(self._arr.shape)
        return chararray._from_array(arr)

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
        return _python_string_strip(self, chars, wrap_chararray=True)

    def lstrip(self, chars=None):
        return _python_string_strip(
            self,
            chars,
            method_name="lstrip",
            wrap_chararray=True,
        )

    def rstrip(self, chars=None):
        return _python_string_strip(
            self,
            chars,
            method_name="rstrip",
            wrap_chararray=True,
        )

    def title(self):
        return _python_string_transform(self, "title", wrap_chararray=True)

    def swapcase(self):
        return _python_string_transform(self, "swapcase", wrap_chararray=True)

    def center(self, width, fillchar=None):
        return _python_string_pad(
            self,
            width,
            "center",
            fillchar=fillchar,
            wrap_chararray=True,
        )

    def ljust(self, width, fillchar=None):
        return _python_string_pad(
            self,
            width,
            "ljust",
            fillchar=fillchar,
            wrap_chararray=True,
        )

    def rjust(self, width, fillchar=None):
        return _python_string_pad(
            self,
            width,
            "rjust",
            fillchar=fillchar,
            wrap_chararray=True,
        )

    def zfill(self, width):
        return python_string_map(
            self,
            lambda item: str(item).zfill(int(width)),
            wrap_chararray=True,
        )

    def replace(self, old, new, count=None):
        return _python_string_replace(
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
        return _python_string_transform(
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
        items, _ = python_string_items(self)
        for item in items:
            if isinstance(item, bytes):
                raise TypeError("isnumeric is only available for unicode strings")
        return python_string_predicate(self, lambda item: item.isnumeric())

    def isdecimal(self):
        items, _ = python_string_items(self)
        for item in items:
            if isinstance(item, bytes):
                raise TypeError("isdecimal is only available for unicode strings")
        return python_string_predicate(self, lambda item: item.isdecimal())

    def split(self, sep=None, maxsplit=-1):
        return _python_string_split(self, sep=sep, maxsplit=maxsplit)

    def rsplit(self, sep=None, maxsplit=-1):
        return _python_string_rsplit(self, sep=sep, maxsplit=maxsplit)

    def splitlines(self):
        return _python_string_splitlines(self)

    def partition(self, sep):
        return _python_string_partition(self, sep, wrap_chararray=True)

    def rpartition(self, sep):
        return _python_string_partition(
            self,
            sep,
            method_name="rpartition",
            wrap_chararray=True,
        )

    def encode(self, encoding='utf-8', errors='strict'):
        return python_string_map(
            self,
            lambda item: item if isinstance(item, bytes) else str(item).encode(encoding, errors),
            wrap_chararray=True,
        )

    def decode(self, encoding='utf-8', errors='strict'):
        return python_string_map(
            self,
            lambda item: item.decode(encoding, errors) if isinstance(item, bytes) else item,
            wrap_chararray=True,
        )

    def join(self, seq):
        seqs = _to_items(seq)
        seps, shape = python_string_items(self)
        if len(seps) == 1 and len(seqs) > 1:
            seps = seps * len(seqs)
            shape = (len(seqs),)
        result = []
        for idx, sep in enumerate(seps):
            current = seqs[idx] if idx < len(seqs) else seqs[-1]
            if isinstance(current, (list, tuple)):
                result.append(str(sep).join(str(part) for part in current))
            else:
                result.append(str(sep).join(str(current)))
        return chararray._from_array(array(result).reshape(shape))


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
        import numpy as _np
        ops = {'<': lambda x, y: x < y, '<=': lambda x, y: x <= y,
               '==': lambda x, y: x == y, '>=': lambda x, y: x >= y,
               '>': lambda x, y: x > y, '!=': lambda x, y: x != y}
        op = ops.get(cmp)
        if op is None:
            raise ValueError(f"Invalid comparison: {cmp!r}")
        def _norm(s):
            if rstrip:
                if isinstance(s, bytes):
                    return s.rstrip()
                return s.rstrip()
            return s
        a1_items = _to_items(a1)
        a2_items = _to_items(a2)
        result = [op(_norm(x), _norm(y)) for x, y in zip(a1_items, a2_items)]
        return _np.array(result, dtype=bool)

    @staticmethod
    def equal(a, b):
        """Element-wise comparison for equality, stripping trailing whitespace."""
        import numpy as _np
        if isinstance(a, (str, bytes)) and isinstance(b, (str, bytes)):
            result = _strip_whitespace(a) == _strip_whitespace(b)
            return _np.array(result)
        items_a = _to_items(a)
        items_b = _to_items(b)
        if len(items_a) == 1 and len(items_b) > 1:
            items_a = items_a * len(items_b)
        elif len(items_b) == 1 and len(items_a) > 1:
            items_b = items_b * len(items_a)
        result = [_strip_whitespace(x) == _strip_whitespace(y) for x, y in zip(items_a, items_b)]
        return _np.array([1.0 if x else 0.0 for x in result]).astype('bool')

    @staticmethod
    def not_equal(a, b):
        import numpy as _np
        if isinstance(a, (str, bytes)) and isinstance(b, (str, bytes)):
            result = _strip_whitespace(a) != _strip_whitespace(b)
            return _np.array(result)
        items_a = _to_items(a)
        items_b = _to_items(b)
        if len(items_a) == 1 and len(items_b) > 1:
            items_a = items_a * len(items_b)
        elif len(items_b) == 1 and len(items_a) > 1:
            items_b = items_b * len(items_a)
        result = [_strip_whitespace(x) != _strip_whitespace(y) for x, y in zip(items_a, items_b)]
        return _np.array([1.0 if x else 0.0 for x in result]).astype('bool')

    @staticmethod
    def greater(a, b):
        import numpy as _np
        items_a = _to_items(a)
        items_b = _to_items(b)
        if len(items_a) == 1 and len(items_b) > 1:
            items_a = items_a * len(items_b)
        elif len(items_b) == 1 and len(items_a) > 1:
            items_b = items_b * len(items_a)
        result = [_strip_whitespace(x) > _strip_whitespace(y) for x, y in zip(items_a, items_b)]
        return _np.array([1.0 if x else 0.0 for x in result]).astype('bool')

    @staticmethod
    def greater_equal(a, b):
        import numpy as _np
        items_a = _to_items(a)
        items_b = _to_items(b)
        if len(items_a) == 1 and len(items_b) > 1:
            items_a = items_a * len(items_b)
        elif len(items_b) == 1 and len(items_a) > 1:
            items_b = items_b * len(items_a)
        result = [_strip_whitespace(x) >= _strip_whitespace(y) for x, y in zip(items_a, items_b)]
        return _np.array([1.0 if x else 0.0 for x in result]).astype('bool')

    @staticmethod
    def less(a, b):
        import numpy as _np
        items_a = _to_items(a)
        items_b = _to_items(b)
        if len(items_a) == 1 and len(items_b) > 1:
            items_a = items_a * len(items_b)
        elif len(items_b) == 1 and len(items_a) > 1:
            items_b = items_b * len(items_a)
        result = [_strip_whitespace(x) < _strip_whitespace(y) for x, y in zip(items_a, items_b)]
        return _np.array([1.0 if x else 0.0 for x in result]).astype('bool')

    @staticmethod
    def less_equal(a, b):
        import numpy as _np
        items_a = _to_items(a)
        items_b = _to_items(b)
        if len(items_a) == 1 and len(items_b) > 1:
            items_a = items_a * len(items_b)
        elif len(items_b) == 1 and len(items_a) > 1:
            items_b = items_b * len(items_a)
        result = [_strip_whitespace(x) <= _strip_whitespace(y) for x, y in zip(items_a, items_b)]
        return _np.array([1.0 if x else 0.0 for x in result]).astype('bool')

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
        return _python_string_strip(a, chars)

    @staticmethod
    def str_len(a):
        return _native_int_output(a, _native.char_str_len)

    @staticmethod
    def startswith(a, prefix, start=0, end=None):
        if start == 0 and end is None and not isinstance(a, chararray):
            return _native_bool_output(a, _native.char_startswith, prefix)
        return python_string_map(
            a,
            lambda item: str(item).startswith(prefix, start)
            if end is None
            else str(item).startswith(prefix, start, end),
            result_kind="bool",
        )

    @staticmethod
    def endswith(a, suffix, start=0, end=None):
        if start == 0 and end is None and not isinstance(a, chararray):
            return _native_bool_output(a, _native.char_endswith, suffix)
        return python_string_map(
            a,
            lambda item: str(item).endswith(suffix, start)
            if end is None
            else str(item).endswith(suffix, start, end),
            result_kind="bool",
        )

    @staticmethod
    def replace(a, old, new, count=None):
        if count is None and not isinstance(a, chararray):
            return _native_string_output(a, _native.char_replace, old, new)
        return _python_string_replace(a, old, new, count=count)

    @staticmethod
    def split(a, sep=None, maxsplit=-1):
        """Split each element in a around sep."""
        out = _python_string_split(a, sep=sep, maxsplit=maxsplit)
        values = out.tolist()
        return values[0] if len(values) == 1 else values

    @staticmethod
    def join(sep, a):
        """Join strings in a with separator sep, element-wise."""
        if isinstance(a, (str, bytes)):
            return str(sep).join(str(a))
        if isinstance(a, (list, tuple)) and a and not isinstance(a[0], (list, tuple)):
            return str(sep).join(str(item) for item in a)
        seps = python_string_broadcast(sep, len(python_string_items(a)[0]))
        sep_iter = iter(seps)
        return python_string_map(
            a,
            lambda item: str(next(sep_iter)).join(str(item)),
        )

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
        if isinstance(a, chararray):
            return a * i
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
        return format_str % values

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
        return _python_string_pad(a, width, "center", fillchar=fillchar)

    @staticmethod
    def ljust(a, width, fillchar=' '):
        """Left-justify each string element in a to width."""
        return _python_string_pad(a, width, "ljust", fillchar=fillchar)

    @staticmethod
    def rjust(a, width, fillchar=' '):
        """Right-justify each string element in a to width."""
        return _python_string_pad(a, width, "rjust", fillchar=fillchar)

    @staticmethod
    def zfill(a, width):
        return python_string_map(a, lambda item: str(item).zfill(int(width)))

    @staticmethod
    def title(a):
        return _python_string_transform(a, "title")

    @staticmethod
    def swapcase(a):
        return _python_string_transform(a, "swapcase")

    @staticmethod
    def isalpha(a):
        return python_string_predicate(a, lambda item: item.isalpha())

    @staticmethod
    def isdigit(a):
        return python_string_predicate(a, lambda item: item.isdigit())

    @staticmethod
    def isnumeric(a):
        items, _ = python_string_items(a)
        for item in items:
            if isinstance(item, bytes):
                raise TypeError("isnumeric is only available for unicode strings")
        return python_string_predicate(a, lambda item: item.isnumeric())

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
    def isdecimal(a):
        items, _ = python_string_items(a)
        for item in items:
            if isinstance(item, bytes):
                raise TypeError("isdecimal is only available for unicode strings")
        return python_string_predicate(a, lambda item: item.isdecimal())

    @staticmethod
    def expandtabs(a, tabsize=8):
        if isinstance(a, (str, bytes)):
            return a.expandtabs(tabsize)
        return _python_string_transform(a, "expandtabs", tabsize)

    @staticmethod
    def partition(a, sep):
        if isinstance(a, (str, bytes)):
            return list(a.partition(sep))
        return _python_string_partition(a, sep)

    @staticmethod
    def rpartition(a, sep):
        if isinstance(a, (str, bytes)):
            return list(a.rpartition(sep))
        return _python_string_partition(a, sep, method_name="rpartition")

    @staticmethod
    def encode(a, encoding='utf-8', errors='strict'):
        """Encode each string element to bytes."""
        if isinstance(a, (str, bytes)):
            if isinstance(a, bytes):
                return a
            return a.encode(encoding, errors)
        return python_string_map(
            a,
            lambda item: item if isinstance(item, bytes) else str(item).encode(encoding, errors),
        )

    @staticmethod
    def decode(a, encoding='utf-8', errors='strict'):
        """Decode each bytes element to string."""
        if isinstance(a, (str, bytes)):
            if isinstance(a, bytes):
                return a.decode(encoding, errors)
            return a
        return python_string_map(
            a,
            lambda item: item.decode(encoding, errors) if isinstance(item, bytes) else item,
        )


char = _char_mod()
