"""String and char array operations."""
import _numpy_native as _native
from _numpy_native import ndarray
from ._helpers import _ObjectArray
from ._creation import array, asarray

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
    """Normalize string-op inputs to a native ndarray boundary once."""
    if isinstance(a, chararray):
        return a._arr, True
    if isinstance(a, ndarray):
        return a, False
    if isinstance(a, _ObjectArray):
        return asarray(a._data), False
    return asarray(a), False


def _native_string_output(a, native_op, *args):
    arr, wrap_chararray = _coerce_native_string_array(a)
    result = native_op(arr, *args)
    if wrap_chararray:
        return chararray._from_array(result)
    return result


def _native_bool_output(a, native_op, *args):
    arr, _ = _coerce_native_string_array(a)
    return native_op(arr, *args)


def _native_int_output(a, native_op, *args):
    arr, _ = _coerce_native_string_array(a)
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
        return chararray._from_array(_native.char_upper(self._arr))

    def lower(self):
        return chararray._from_array(_native.char_lower(self._arr))

    def capitalize(self):
        return chararray._from_array(_native.char_capitalize(self._arr))

    def strip(self, chars=None):
        import numpy as _np
        items = self._arr.flatten().tolist()
        if chars is not None:
            if isinstance(chars, (list, tuple)):
                if len(chars) == 1:
                    chars = chars * len(items)
                result = []
                for s, c in zip(items, chars):
                    if isinstance(s, bytes):
                        result.append(s.strip(c if isinstance(c, bytes) else c.encode('latin-1')))
                    else:
                        result.append(s.strip(c))
            else:
                result = []
                for s in items:
                    if isinstance(s, bytes):
                        result.append(s.strip(chars if isinstance(chars, bytes) else chars.encode('latin-1')))
                    else:
                        result.append(s.strip(chars))
        else:
            result = [s.strip() if isinstance(s, (str, bytes)) else str(s).strip() for s in items]
        arr = _np.array(result)
        if len(self._arr.shape) > 1:
            arr = arr.reshape(self._arr.shape)
        return chararray._from_array(arr)

    def lstrip(self, chars=None):
        import numpy as _np
        items = self._arr.flatten().tolist()
        if chars is not None:
            if isinstance(chars, (list, tuple)):
                if len(chars) == 1:
                    chars = chars * len(items)
                result = []
                for s, c in zip(items, chars):
                    if isinstance(s, bytes):
                        result.append(s.lstrip(c if isinstance(c, bytes) else c.encode('latin-1')))
                    else:
                        result.append(s.lstrip(c))
            else:
                result = [s.lstrip(chars) for s in items]
        else:
            result = [s.lstrip() if isinstance(s, (str, bytes)) else str(s).lstrip() for s in items]
        arr = _np.array(result)
        if len(self._arr.shape) > 1:
            arr = arr.reshape(self._arr.shape)
        return chararray._from_array(arr)

    def rstrip(self, chars=None):
        import numpy as _np
        items = self._arr.flatten().tolist()
        if chars is not None:
            if isinstance(chars, (list, tuple)):
                if len(chars) == 1:
                    chars = chars * len(items)
                result = []
                for s, c in zip(items, chars):
                    if isinstance(s, bytes):
                        result.append(s.rstrip(c if isinstance(c, bytes) else c.encode('latin-1')))
                    else:
                        result.append(s.rstrip(c))
            else:
                result = [s.rstrip(chars) for s in items]
        else:
            result = [s.rstrip() if isinstance(s, (str, bytes)) else str(s).rstrip() for s in items]
        arr = _np.array(result)
        if len(self._arr.shape) > 1:
            arr = arr.reshape(self._arr.shape)
        return chararray._from_array(arr)

    def title(self):
        import numpy as _np
        items = self._arr.flatten().tolist()
        result = [s.title() if isinstance(s, (str, bytes)) else str(s).title() for s in items]
        arr = _np.array(result)
        if len(self._arr.shape) > 1:
            arr = arr.reshape(self._arr.shape)
        return chararray._from_array(arr)

    def swapcase(self):
        import numpy as _np
        items = self._arr.flatten().tolist()
        result = [s.swapcase() if isinstance(s, (str, bytes)) else str(s).swapcase() for s in items]
        arr = _np.array(result)
        if len(self._arr.shape) > 1:
            arr = arr.reshape(self._arr.shape)
        return chararray._from_array(arr)

    def center(self, width, fillchar=None):
        import numpy as _np
        items = self._arr.flatten().tolist()
        widths = width if isinstance(width, (list, tuple)) else [width]
        if len(widths) == 1:
            widths = widths * len(items)
        result = []
        for i, s in enumerate(items):
            w = int(widths[i % len(widths)])
            fc = fillchar if fillchar is not None else (b' ' if isinstance(s, bytes) else ' ')
            if isinstance(s, bytes) and isinstance(fc, bytes):
                result.append(s.center(w, fc))
            else:
                result.append(str(s).center(w, str(fc) if not isinstance(fc, str) else fc))
        arr = _np.array(result)
        if len(self._arr.shape) > 1:
            arr = arr.reshape(self._arr.shape)
        return chararray._from_array(arr)

    def ljust(self, width, fillchar=None):
        import numpy as _np
        items = self._arr.flatten().tolist()
        widths = width if isinstance(width, (list, tuple)) else [width]
        if len(widths) == 1:
            widths = widths * len(items)
        result = []
        for i, s in enumerate(items):
            w = int(widths[i % len(widths)])
            fc = fillchar if fillchar is not None else (b' ' if isinstance(s, bytes) else ' ')
            if isinstance(s, bytes) and isinstance(fc, bytes):
                result.append(s.ljust(w, fc))
            else:
                result.append(str(s).ljust(w, str(fc) if not isinstance(fc, str) else fc))
        arr = _np.array(result)
        if len(self._arr.shape) > 1:
            arr = arr.reshape(self._arr.shape)
        return chararray._from_array(arr)

    def rjust(self, width, fillchar=None):
        import numpy as _np
        items = self._arr.flatten().tolist()
        widths = width if isinstance(width, (list, tuple)) else [width]
        if len(widths) == 1:
            widths = widths * len(items)
        result = []
        for i, s in enumerate(items):
            w = int(widths[i % len(widths)])
            fc = fillchar if fillchar is not None else (b' ' if isinstance(s, bytes) else ' ')
            if isinstance(s, bytes) and isinstance(fc, bytes):
                result.append(s.rjust(w, fc))
            else:
                result.append(str(s).rjust(w, str(fc) if not isinstance(fc, str) else fc))
        arr = _np.array(result)
        if len(self._arr.shape) > 1:
            arr = arr.reshape(self._arr.shape)
        return chararray._from_array(arr)

    def zfill(self, width):
        import numpy as _np
        items = self._arr.flatten().tolist()
        result = [s.zfill(int(width)) if isinstance(s, (str, bytes)) else str(s).zfill(int(width)) for s in items]
        arr = _np.array(result)
        if len(self._arr.shape) > 1:
            arr = arr.reshape(self._arr.shape)
        return chararray._from_array(arr)

    def replace(self, old, new, count=None):
        import numpy as _np
        items = self._arr.flatten().tolist()
        olds = old if isinstance(old, (list, tuple)) else [old]
        news = new if isinstance(new, (list, tuple)) else [new]
        counts = count if isinstance(count, (list, tuple)) else ([count] if count is not None else None)
        if len(olds) == 1:
            olds = olds * len(items)
        if len(news) == 1:
            news = news * len(items)
        if counts is not None and len(counts) == 1:
            counts = counts * len(items)
        result = []
        for i, s in enumerate(items):
            o = olds[i % len(olds)]
            n = news[i % len(news)]
            if counts is not None:
                c = int(counts[i % len(counts)])
                result.append(s.replace(o, n, c))
            else:
                result.append(s.replace(o, n))
        arr = _np.array(result)
        if len(self._arr.shape) > 1:
            arr = arr.reshape(self._arr.shape)
        return chararray._from_array(arr)

    def startswith(self, prefix, start=0, end=None):
        import numpy as _np
        items = self._arr.flatten().tolist()
        result = []
        for s in items:
            if end is None:
                result.append(s.startswith(prefix, start))
            else:
                result.append(s.startswith(prefix, start, end))
        out = _np.array([1.0 if x else 0.0 for x in result]).astype('bool')
        if len(self._arr.shape) > 1:
            out = out.reshape(self._arr.shape)
        return out

    def endswith(self, suffix, start=0, end=None):
        import numpy as _np
        items = self._arr.flatten().tolist()
        result = []
        for s in items:
            if end is None:
                result.append(s.endswith(suffix, start))
            else:
                result.append(s.endswith(suffix, start, end))
        out = _np.array([1.0 if x else 0.0 for x in result]).astype('bool')
        if len(self._arr.shape) > 1:
            out = out.reshape(self._arr.shape)
        return out

    def find(self, sub, start=0, end=None):
        import numpy as _np
        items = self._arr.flatten().tolist()
        subs = sub if isinstance(sub, (list, tuple)) else [sub]
        if len(subs) == 1:
            subs = subs * len(items)
        result = []
        for i, s in enumerate(items):
            sb = subs[i % len(subs)]
            if end is None:
                result.append(s.find(sb, start))
            else:
                result.append(s.find(sb, start, end))
        out = _np.array(result)
        if len(self._arr.shape) > 1:
            out = out.reshape(self._arr.shape)
        return out

    def rfind(self, sub, start=0, end=None):
        import numpy as _np
        items = self._arr.flatten().tolist()
        subs = sub if isinstance(sub, (list, tuple)) else [sub]
        if len(subs) == 1:
            subs = subs * len(items)
        result = []
        for i, s in enumerate(items):
            sb = subs[i % len(subs)]
            if end is None:
                result.append(s.rfind(sb, start))
            else:
                result.append(s.rfind(sb, start, end))
        out = _np.array(result)
        if len(self._arr.shape) > 1:
            out = out.reshape(self._arr.shape)
        return out

    def index(self, sub, start=0, end=None):
        import numpy as _np
        items = self._arr.flatten().tolist()
        result = []
        for s in items:
            if end is None:
                result.append(s.index(sub, start))
            else:
                result.append(s.index(sub, start, end))
        out = _np.array(result)
        if len(self._arr.shape) > 1:
            out = out.reshape(self._arr.shape)
        return out

    def rindex(self, sub, start=0, end=None):
        import numpy as _np
        items = self._arr.flatten().tolist()
        result = []
        for s in items:
            if end is None:
                result.append(s.rindex(sub, start))
            else:
                result.append(s.rindex(sub, start, end))
        out = _np.array(result)
        if len(self._arr.shape) > 1:
            out = out.reshape(self._arr.shape)
        return out

    def count(self, sub, start=0, end=None):
        import numpy as _np
        items = self._arr.flatten().tolist()
        result = []
        for s in items:
            if end is None:
                result.append(s.count(sub, start))
            else:
                result.append(s.count(sub, start, end))
        out = _np.array(result)
        if len(self._arr.shape) > 1:
            out = out.reshape(self._arr.shape)
        return out

    def expandtabs(self, tabsize=8):
        import numpy as _np
        items = self._arr.flatten().tolist()
        result = [s.expandtabs(tabsize) for s in items]
        arr = _np.array(result)
        if len(self._arr.shape) > 1:
            arr = arr.reshape(self._arr.shape)
        return chararray._from_array(arr)

    def isalnum(self):
        import numpy as _np
        items = self._arr.flatten().tolist()
        out = _np.array([1.0 if s.isalnum() else 0.0 for s in items]).astype('bool')
        if len(self._arr.shape) > 1:
            out = out.reshape(self._arr.shape)
        return out

    def isalpha(self):
        import numpy as _np
        items = self._arr.flatten().tolist()
        out = _np.array([1.0 if s.isalpha() else 0.0 for s in items]).astype('bool')
        if len(self._arr.shape) > 1:
            out = out.reshape(self._arr.shape)
        return out

    def isdigit(self):
        import numpy as _np
        items = self._arr.flatten().tolist()
        out = _np.array([1.0 if s.isdigit() else 0.0 for s in items]).astype('bool')
        if len(self._arr.shape) > 1:
            out = out.reshape(self._arr.shape)
        return out

    def islower(self):
        import numpy as _np
        items = self._arr.flatten().tolist()
        out = _np.array([1.0 if s.islower() else 0.0 for s in items]).astype('bool')
        if len(self._arr.shape) > 1:
            out = out.reshape(self._arr.shape)
        return out

    def isupper(self):
        import numpy as _np
        items = self._arr.flatten().tolist()
        out = _np.array([1.0 if s.isupper() else 0.0 for s in items]).astype('bool')
        if len(self._arr.shape) > 1:
            out = out.reshape(self._arr.shape)
        return out

    def isspace(self):
        import numpy as _np
        items = self._arr.flatten().tolist()
        out = _np.array([1.0 if s.isspace() else 0.0 for s in items]).astype('bool')
        if len(self._arr.shape) > 1:
            out = out.reshape(self._arr.shape)
        return out

    def istitle(self):
        import numpy as _np
        items = self._arr.flatten().tolist()
        out = _np.array([1.0 if s.istitle() else 0.0 for s in items]).astype('bool')
        if len(self._arr.shape) > 1:
            out = out.reshape(self._arr.shape)
        return out

    def isnumeric(self):
        import numpy as _np
        items = self._arr.flatten().tolist()
        # isnumeric only works on unicode, not bytes
        for s in items:
            if isinstance(s, bytes):
                raise TypeError("isnumeric is only available for unicode strings")
        out = _np.array([1.0 if s.isnumeric() else 0.0 for s in items]).astype('bool')
        if len(self._arr.shape) > 1:
            out = out.reshape(self._arr.shape)
        return out

    def isdecimal(self):
        import numpy as _np
        items = self._arr.flatten().tolist()
        for s in items:
            if isinstance(s, bytes):
                raise TypeError("isdecimal is only available for unicode strings")
        out = _np.array([1.0 if s.isdecimal() else 0.0 for s in items]).astype('bool')
        if len(self._arr.shape) > 1:
            out = out.reshape(self._arr.shape)
        return out

    def split(self, sep=None, maxsplit=-1):
        import numpy as _np
        items = self._arr.flatten().tolist()
        result = [s.split(sep, maxsplit) for s in items]
        out = _np.array(result, dtype=object)
        if len(self._arr.shape) > 1:
            out = out.reshape(self._arr.shape)
        return out

    def rsplit(self, sep=None, maxsplit=-1):
        import numpy as _np
        items = self._arr.flatten().tolist()
        result = [s.rsplit(sep, maxsplit) for s in items]
        out = _np.array(result, dtype=object)
        if len(self._arr.shape) > 1:
            out = out.reshape(self._arr.shape)
        return out

    def splitlines(self):
        import numpy as _np
        items = self._arr.flatten().tolist()
        result = [s.splitlines() for s in items]
        out = _np.array(result, dtype=object)
        if len(self._arr.shape) > 1:
            out = out.reshape(self._arr.shape)
        return out

    def partition(self, sep):
        import numpy as _np
        items = self._arr.flatten().tolist()
        seps = sep if isinstance(sep, (list, tuple)) else [sep]
        if len(seps) == 1:
            seps = seps * len(items)
        result = []
        for i, s in enumerate(items):
            sp = seps[i % len(seps)]
            result.append(s.partition(sp))
        # Build array of tuples
        arr = _np.array(result)
        if len(self._arr.shape) > 1:
            new_shape = list(self._arr.shape) + [3]
            arr = arr.reshape(new_shape)
        return chararray._from_array(arr)

    def rpartition(self, sep):
        import numpy as _np
        items = self._arr.flatten().tolist()
        seps = sep if isinstance(sep, (list, tuple)) else [sep]
        if len(seps) == 1:
            seps = seps * len(items)
        result = []
        for i, s in enumerate(items):
            sp = seps[i % len(seps)]
            result.append(s.rpartition(sp))
        arr = _np.array(result)
        if len(self._arr.shape) > 1:
            new_shape = list(self._arr.shape) + [3]
            arr = arr.reshape(new_shape)
        return chararray._from_array(arr)

    def encode(self, encoding='utf-8', errors='strict'):
        import numpy as _np
        items = self._arr.flatten().tolist()
        result = []
        for s in items:
            if isinstance(s, bytes):
                result.append(s)
            else:
                result.append(s.encode(encoding, errors))
        arr = _np.array(result)
        if len(self._arr.shape) > 1:
            arr = arr.reshape(self._arr.shape)
        return chararray._from_array(arr)

    def decode(self, encoding='utf-8', errors='strict'):
        import numpy as _np
        items = self._arr.flatten().tolist()
        result = []
        for s in items:
            if isinstance(s, bytes):
                result.append(s.decode(encoding, errors))
            else:
                result.append(s)
        arr = _np.array(result)
        if len(self._arr.shape) > 1:
            arr = arr.reshape(self._arr.shape)
        return chararray._from_array(arr)

    def join(self, seq):
        import numpy as _np
        items = self._arr.flatten().tolist()
        seqs = _to_items(seq)
        if len(items) == 1 and len(seqs) > 1:
            items = items * len(seqs)
        result = []
        for i, sep in enumerate(items):
            s = seqs[i % len(seqs)] if i < len(seqs) else seqs[-1]
            if isinstance(s, (list, tuple)):
                result.append(sep.join(s))
            else:
                result.append(sep.join(str(s)))
        arr = _np.array(result)
        if len(self._arr.shape) > 1:
            arr = arr.reshape(self._arr.shape)
        return chararray._from_array(arr)


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
        return _native_string_output(a, _native.char_upper)

    @staticmethod
    def lower(a):
        return _native_string_output(a, _native.char_lower)

    @staticmethod
    def capitalize(a):
        if isinstance(a, chararray):
            return a.capitalize()
        return _native.char_capitalize(a)

    @staticmethod
    def strip(a, chars=None):
        if chars is not None and isinstance(a, chararray):
            return a.strip(chars)
        return _native_string_output(a, _native.char_strip)

    @staticmethod
    def str_len(a):
        return _native_int_output(a, _native.char_str_len)

    @staticmethod
    def startswith(a, prefix, start=0, end=None):
        if (start != 0 or end is not None) and isinstance(a, chararray):
            return a.startswith(prefix, start, end)
        return _native_bool_output(a, _native.char_startswith, prefix)

    @staticmethod
    def endswith(a, suffix, start=0, end=None):
        if (start != 0 or end is not None) and isinstance(a, chararray):
            return a.endswith(suffix, start, end)
        return _native_bool_output(a, _native.char_endswith, suffix)

    @staticmethod
    def replace(a, old, new, count=None):
        if count is not None and isinstance(a, chararray):
            return a.replace(old, new, count=count)
        return _native_string_output(a, _native.char_replace, old, new)

    @staticmethod
    def split(a, sep=None, maxsplit=-1):
        """Split each element in a around sep."""
        if isinstance(a, chararray):
            return a.split(sep, maxsplit)
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
        if isinstance(a, chararray):
            items_a = a._arr.flatten().tolist()
        elif isinstance(a, ndarray):
            items_a = a.tolist()
        elif isinstance(a, _ObjectArray):
            items_a = a._data
        elif isinstance(a, (list, tuple)):
            items_a = a
        else:
            items_a = [a]

        seps = _to_items(sep)

        # If single string, join each char
        if isinstance(a, (str, bytes)):
            if len(seps) == 1:
                return seps[0].join(a)
            result = [s.join(a) for s in seps]
            return array(result)

        # If items is a list of lists, join each sublist
        if len(items_a) > 0 and isinstance(items_a[0], (list, tuple)):
            if len(seps) == 1:
                seps = seps * len(items_a)
            result = [str(seps[i]).join(str(x) for x in sub) for i, sub in enumerate(items_a)]
            return array(result)
        # Otherwise join all items into a single string
        if len(seps) == 1:
            return str(seps[0]).join(str(x) for x in items_a)
        result = [str(s).join(str(x) for x in items_a) for s in seps]
        return array(result)

    @staticmethod
    def find(a, sub, start=0, end=None):
        """Find first occurrence of sub in each element of a."""
        if isinstance(a, chararray):
            return a.find(sub, start, end)
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
    def rfind(a, sub, start=0, end=None):
        if isinstance(a, chararray):
            return a.rfind(sub, start, end)
        items = _to_items(a)
        result = []
        for s in items:
            s = str(s)
            if end is None:
                result.append(s.rfind(sub, start))
            else:
                result.append(s.rfind(sub, start, end))
        return array(result)

    @staticmethod
    def index(a, sub, start=0, end=None):
        if isinstance(a, chararray):
            return a.index(sub, start, end)
        if isinstance(a, (str, bytes)):
            import numpy as _np
            if end is None:
                return _np.array(a.index(sub, start))
            return _np.array(a.index(sub, start, end))
        items = _to_items(a)
        result = []
        for s in items:
            s = str(s)
            if end is None:
                result.append(s.index(sub, start))
            else:
                result.append(s.index(sub, start, end))
        return array(result)

    @staticmethod
    def rindex(a, sub, start=0, end=None):
        if isinstance(a, chararray):
            return a.rindex(sub, start, end)
        if isinstance(a, (str, bytes)):
            import numpy as _np
            if end is None:
                return _np.array(a.rindex(sub, start))
            return _np.array(a.rindex(sub, start, end))
        items = _to_items(a)
        result = []
        for s in items:
            s = str(s)
            if end is None:
                result.append(s.rindex(sub, start))
            else:
                result.append(s.rindex(sub, start, end))
        return array(result)

    @staticmethod
    def count(a, sub, start=0, end=None):
        """Count non-overlapping occurrences of sub in each element of a."""
        if isinstance(a, chararray):
            return a.count(sub, start, end)
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
        if isinstance(a, chararray):
            return a.center(width, fillchar)
        import numpy as _np
        items = _char_mod._to_str_list(a)
        widths = width if isinstance(width, (list, tuple)) else [width]
        if len(widths) == 1:
            widths = widths * len(items)
        result = []
        for i, s in enumerate(items):
            w = int(widths[i % len(widths)])
            result.append(s.center(w, fillchar))
        # For bytes input, return chararray; for plain list/str input, return _ObjectArray
        if isinstance(a, (ndarray, chararray)):
            arr = _np.array(result)
            if isinstance(a, ndarray) and len(a.shape) > 1:
                arr = arr.reshape(a.shape)
            return chararray._from_array(arr)
        return _ObjectArray(result)

    @staticmethod
    def ljust(a, width, fillchar=' '):
        """Left-justify each string element in a to width."""
        if isinstance(a, chararray):
            return a.ljust(width, fillchar)
        import numpy as _np
        items = _char_mod._to_str_list(a)
        widths = width if isinstance(width, (list, tuple)) else [width]
        if len(widths) == 1:
            widths = widths * len(items)
        result = []
        for i, s in enumerate(items):
            w = int(widths[i % len(widths)])
            result.append(s.ljust(w, fillchar))
        if isinstance(a, (ndarray, chararray)):
            arr = _np.array(result)
            if isinstance(a, ndarray) and len(a.shape) > 1:
                arr = arr.reshape(a.shape)
            return chararray._from_array(arr)
        return _ObjectArray(result)

    @staticmethod
    def rjust(a, width, fillchar=' '):
        """Right-justify each string element in a to width."""
        if isinstance(a, chararray):
            return a.rjust(width, fillchar)
        import numpy as _np
        items = _char_mod._to_str_list(a)
        widths = width if isinstance(width, (list, tuple)) else [width]
        if len(widths) == 1:
            widths = widths * len(items)
        result = []
        for i, s in enumerate(items):
            w = int(widths[i % len(widths)])
            result.append(s.rjust(w, fillchar))
        if isinstance(a, (ndarray, chararray)):
            arr = _np.array(result)
            if isinstance(a, ndarray) and len(a.shape) > 1:
                arr = arr.reshape(a.shape)
            return chararray._from_array(arr)
        return _ObjectArray(result)

    @staticmethod
    def zfill(a, width):
        if isinstance(a, chararray):
            return a.zfill(width)
        data = _char_mod._to_str_list(a)
        return _ObjectArray([s.zfill(int(width)) for s in data])

    @staticmethod
    def title(a):
        if isinstance(a, chararray):
            return a.title()
        data = _char_mod._to_str_list(a)
        return _ObjectArray([s.title() for s in data])

    @staticmethod
    def swapcase(a):
        if isinstance(a, chararray):
            return a.swapcase()
        data = _char_mod._to_str_list(a)
        return _ObjectArray([s.swapcase() for s in data])

    @staticmethod
    def isalpha(a):
        if isinstance(a, chararray):
            return a.isalpha()
        data = _char_mod._to_str_list(a)
        return array([1.0 if s.isalpha() else 0.0 for s in data]).astype("bool")

    @staticmethod
    def isdigit(a):
        if isinstance(a, chararray):
            return a.isdigit()
        data = _char_mod._to_str_list(a)
        return array([1.0 if s.isdigit() else 0.0 for s in data]).astype("bool")

    @staticmethod
    def isnumeric(a):
        if isinstance(a, chararray):
            return a.isnumeric()
        data = _char_mod._to_str_list(a)
        return array([1.0 if (s.isnumeric() if hasattr(s, 'isnumeric') else s.isdigit()) else 0.0 for s in data]).astype("bool")

    @staticmethod
    def isupper(a):
        if isinstance(a, chararray):
            return a.isupper()
        data = _char_mod._to_str_list(a)
        return array([1.0 if s.isupper() else 0.0 for s in data]).astype("bool")

    @staticmethod
    def islower(a):
        if isinstance(a, chararray):
            return a.islower()
        data = _char_mod._to_str_list(a)
        return array([1.0 if s.islower() else 0.0 for s in data]).astype("bool")

    @staticmethod
    def isspace(a):
        if isinstance(a, chararray):
            return a.isspace()
        data = _char_mod._to_str_list(a)
        return array([1.0 if s.isspace() else 0.0 for s in data]).astype("bool")

    @staticmethod
    def isdecimal(a):
        if isinstance(a, chararray):
            return a.isdecimal()
        a = asarray(a)
        return array([1.0 if str(s).isdecimal() else 0.0 for s in a.flatten().tolist()]).reshape(a.shape).astype("bool")

    @staticmethod
    def expandtabs(a, tabsize=8):
        if isinstance(a, chararray):
            return a.expandtabs(tabsize)
        import numpy as _np
        if isinstance(a, (str, bytes)):
            return a.expandtabs(tabsize)
        items = _to_items(a)
        result = [s.expandtabs(tabsize) for s in items]
        arr = _np.array(result)
        return chararray._from_array(arr)

    @staticmethod
    def partition(a, sep):
        if isinstance(a, chararray):
            return a.partition(sep)
        if isinstance(a, (str, bytes)):
            return list(a.partition(sep))
        items = _to_items(a)
        result = [list(s.partition(sep)) for s in items]
        return array(result)

    @staticmethod
    def rpartition(a, sep):
        if isinstance(a, chararray):
            return a.rpartition(sep)
        if isinstance(a, (str, bytes)):
            return list(a.rpartition(sep))
        items = _to_items(a)
        result = [list(s.rpartition(sep)) for s in items]
        return array(result)

    @staticmethod
    def encode(a, encoding='utf-8', errors='strict'):
        """Encode each string element to bytes."""
        if isinstance(a, chararray):
            return a.encode(encoding, errors)
        if isinstance(a, (str, bytes)):
            if isinstance(a, bytes):
                return a
            return a.encode(encoding, errors)
        data = _char_mod._to_str_list(a)
        import numpy as _np
        result = [s.encode(encoding, errors) for s in data]
        arr = _np.array(result)
        if isinstance(a, ndarray) and len(a.shape) > 1:
            arr = arr.reshape(a.shape)
        return chararray._from_array(arr)

    @staticmethod
    def decode(a, encoding='utf-8', errors='strict'):
        """Decode each bytes element to string."""
        if isinstance(a, chararray):
            return a.decode(encoding, errors)
        if isinstance(a, (str, bytes)):
            if isinstance(a, bytes):
                return a.decode(encoding, errors)
            return a
        data = _char_mod._to_str_list(a)
        import numpy as _np
        result = [s.decode(encoding, errors) if isinstance(s, bytes) else s for s in data]
        arr = _np.array(result)
        if isinstance(a, ndarray) and len(a.shape) > 1:
            arr = arr.reshape(a.shape)
        return chararray._from_array(arr)


char = _char_mod()
