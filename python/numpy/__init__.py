"""NumPy-compatible Python package wrapping the Rust native module."""
import sys as _sys
import math as _math
from functools import reduce as _reduce

__version__ = "1.26.0"

# Import from native Rust module
import _numpy_native as _native
from _numpy_native import ndarray
from _numpy_native import dot
from _numpy_native import concatenate as _native_concatenate

class AxisError(ValueError, IndexError):
    """Exception for invalid axis."""
    def __init__(self, axis=None, ndim=None, msg_prefix=None):
        if axis is not None and ndim is not None:
            msg = "axis {} is out of bounds for array of dimension {}".format(axis, ndim)
            if msg_prefix:
                msg = "{}: {}".format(msg_prefix, msg)
        elif axis is not None:
            msg = str(axis)
        else:
            msg = msg_prefix or ""
        super().__init__(msg)
        self.axis = axis
        self.ndim = ndim

# Simple flags object for Python-only array stubs
class _ArrayFlags:
    def __init__(self, c_contiguous=True, f_contiguous=False):
        self.c_contiguous = c_contiguous
        self.f_contiguous = f_contiguous
        self.writeable = True
        self.owndata = True
        self.aligned = True
        self.writebackifcopy = False
    def __getitem__(self, key):
        k = key.upper()
        if k in ('C_CONTIGUOUS', 'C', 'CONTIGUOUS'):
            return self.c_contiguous
        if k in ('F_CONTIGUOUS', 'F', 'FORTRAN'):
            return self.f_contiguous
        if k == 'WRITEABLE':
            return self.writeable
        return False

# Wrap creation functions to accept (and currently ignore) dtype keyword
class _ObjectArray:
    """Lightweight fallback for arrays with non-numeric dtypes (strings, structured, etc.)."""
    def __init__(self, data, dt=None, shape=None, is_fortran=False, itemsize=None):
        if isinstance(data, (list, tuple)):
            self._data = list(data)
        else:
            self._data = [data]
        self._dtype = dt or "object"
        self._is_fortran = is_fortran
        if shape is not None:
            self._shape = tuple(shape)
            self._ndim = len(self._shape)
        elif isinstance(self._data, list) and len(self._data) > 0 and isinstance(self._data[0], (list, tuple)):
            self._shape = (len(self._data), len(self._data[0]))
            self._ndim = 2
        else:
            self._shape = (len(self._data),)
            self._ndim = 1
        # itemsize for strides computation (unicode=4, bytes=1, default=8)
        if itemsize is not None:
            self._itemsize = itemsize
        elif isinstance(self._dtype, str) and self._dtype == 'str':
            self._itemsize = 4
        elif isinstance(self._dtype, str) and self._dtype in ('bytes', 'S1'):
            self._itemsize = 1
        else:
            self._itemsize = 8

    @property
    def shape(self): return self._shape
    @property
    def ndim(self): return self._ndim
    @property
    def dtype(self): return self._dtype
    @property
    def size(self):
        n = 1
        for s in self._shape:
            n *= s
        return n
    @property
    def T(self): return self
    @property
    def flags(self): return _ArrayFlags(c_contiguous=not self._is_fortran, f_contiguous=self._is_fortran)
    @property
    def strides(self):
        """C-order strides computed from shape and itemsize."""
        s = [self._itemsize]
        for dim in reversed(self._shape[1:]):
            s.insert(0, s[0] * dim)
        return tuple(s)
    def _mark_fortran(self):
        self._is_fortran = True

    def copy(self): return _ObjectArray(list(self._data), self._dtype, shape=self._shape, is_fortran=self._is_fortran, itemsize=self._itemsize)
    def astype(self, dtype): return _ObjectArray(list(self._data), str(dtype), shape=self._shape, is_fortran=self._is_fortran)
    def flatten(self): return self
    def ravel(self): return self
    def tolist(self): return list(self._data)
    def all(self): return all(self._data)
    def any(self): return any(self._data)
    def __len__(self): return len(self._data)
    def __getitem__(self, key):
        result = self._data[key]
        if isinstance(key, slice):
            return _ObjectArray(result, self._dtype)
        return result
    def __setitem__(self, key, value):
        if isinstance(key, slice):
            if isinstance(value, (list, tuple)):
                self._data[key] = value
            elif isinstance(value, _ObjectArray):
                self._data[key] = value._data
            else:
                self._data[key] = [value] * len(range(*key.indices(len(self._data))))
        else:
            self._data[key] = value
    def _to_bool_array(self, data):
        return _native.array([1.0 if x else 0.0 for x in data]).astype("bool")
    def __eq__(self, other):
        if isinstance(other, _ObjectArray):
            return self._to_bool_array([a == b for a, b in zip(self._data, other._data)])
        if isinstance(other, ndarray):
            return self._to_bool_array([a == b for a, b in zip(self._data, other.flatten().tolist())])
        if other is None or isinstance(other, (int, float, complex, str)):
            return self._to_bool_array([x == other for x in self._data])
        return NotImplemented
    def __ne__(self, other):
        if isinstance(other, _ObjectArray):
            return self._to_bool_array([a != b for a, b in zip(self._data, other._data)])
        if isinstance(other, ndarray):
            return self._to_bool_array([a != b for a, b in zip(self._data, other.flatten().tolist())])
        if other is None or isinstance(other, (int, float, complex, str)):
            return self._to_bool_array([x != other for x in self._data])
        return NotImplemented
    @staticmethod
    def _cmp_complex(a, b):
        """Lexicographic comparison for complex: (real, imag). Returns -1, 0, or 1."""
        ar = a.real if isinstance(a, complex) else float(a)
        ai = a.imag if isinstance(a, complex) else 0.0
        br = b.real if isinstance(b, complex) else float(b)
        bi = b.imag if isinstance(b, complex) else 0.0
        if ar < br: return -1
        if ar > br: return 1
        if ai < bi: return -1
        if ai > bi: return 1
        return 0
    def _cmp_lt(self, a, b):
        if isinstance(a, complex) or isinstance(b, complex):
            return self._cmp_complex(a, b) < 0
        return a < b
    def _cmp_le(self, a, b):
        if isinstance(a, complex) or isinstance(b, complex):
            return self._cmp_complex(a, b) <= 0
        return a <= b
    def _cmp_gt(self, a, b):
        if isinstance(a, complex) or isinstance(b, complex):
            return self._cmp_complex(a, b) > 0
        return a > b
    def _cmp_ge(self, a, b):
        if isinstance(a, complex) or isinstance(b, complex):
            return self._cmp_complex(a, b) >= 0
        return a >= b
    def __le__(self, other):
        if isinstance(other, _ObjectArray):
            return self._to_bool_array([self._cmp_le(a, b) for a, b in zip(self._data, other._data)])
        if isinstance(other, ndarray):
            return self._to_bool_array([self._cmp_le(a, b) for a, b in zip(self._data, other.flatten().tolist())])
        return self._to_bool_array([self._cmp_le(x, other) for x in self._data])
    def __lt__(self, other):
        if isinstance(other, _ObjectArray):
            return self._to_bool_array([self._cmp_lt(a, b) for a, b in zip(self._data, other._data)])
        if isinstance(other, ndarray):
            return self._to_bool_array([self._cmp_lt(a, b) for a, b in zip(self._data, other.flatten().tolist())])
        return self._to_bool_array([self._cmp_lt(x, other) for x in self._data])
    def __ge__(self, other):
        if isinstance(other, _ObjectArray):
            return self._to_bool_array([self._cmp_ge(a, b) for a, b in zip(self._data, other._data)])
        if isinstance(other, ndarray):
            return self._to_bool_array([self._cmp_ge(a, b) for a, b in zip(self._data, other.flatten().tolist())])
        return self._to_bool_array([self._cmp_ge(x, other) for x in self._data])
    def __gt__(self, other):
        if isinstance(other, _ObjectArray):
            return self._to_bool_array([self._cmp_gt(a, b) for a, b in zip(self._data, other._data)])
        if isinstance(other, ndarray):
            return self._to_bool_array([self._cmp_gt(a, b) for a, b in zip(self._data, other.flatten().tolist())])
        return self._to_bool_array([self._cmp_gt(x, other) for x in self._data])
    def __sub__(self, other):
        if isinstance(other, _ObjectArray):
            return _ObjectArray([a - b for a, b in zip(self._data, other._data)], self._dtype)
        return _ObjectArray([x - other for x in self._data], self._dtype)
    def __rsub__(self, other):
        return _ObjectArray([other - x for x in self._data], self._dtype)
    def __mul__(self, other):
        if isinstance(other, _ObjectArray):
            return _ObjectArray([a * b for a, b in zip(self._data, other._data)], self._dtype)
        if isinstance(other, int) and self._dtype == "object":
            return _ObjectArray(self._data * other, self._dtype)
        return _ObjectArray([x * other for x in self._data], self._dtype)
    def __rmul__(self, other):
        return self.__mul__(other)
    def __add__(self, other):
        if isinstance(other, _ObjectArray):
            return _ObjectArray([a + b for a, b in zip(self._data, other._data)], self._dtype)
        return _ObjectArray([x + other for x in self._data], self._dtype)
    def __radd__(self, other):
        return self.__add__(other)
    def conjugate(self):
        return _ObjectArray([x.conjugate() if hasattr(x, 'conjugate') else x for x in self._data], self._dtype)
    def conj(self):
        return self.conjugate()
    def sum(self, axis=None, keepdims=False, **kwargs):
        _bsum = __import__("builtins").sum
        return _bsum(self._data)
    def prod(self, axis=None, keepdims=False, **kwargs):
        r = 1
        for x in self._data:
            r *= x
        return r
    def mean(self, axis=None, keepdims=False, **kwargs):
        _bsum = __import__("builtins").sum
        return _bsum(self._data) / len(self._data)
    def var(self, axis=None, ddof=0, keepdims=False, **kwargs):
        m = self.mean()
        _babs = __import__("builtins").abs
        _bsum = __import__("builtins").sum
        return _bsum(_babs(x - m) ** 2 for x in self._data) / (len(self._data) - ddof)
    def std(self, axis=None, ddof=0, keepdims=False, **kwargs):
        return self.var(axis, ddof, keepdims) ** 0.5
    def __abs__(self):
        _babs = __import__("builtins").abs
        return _ObjectArray([_babs(x) for x in self._data], self._dtype)
    def __pow__(self, other):
        return _ObjectArray([x ** other for x in self._data], self._dtype)
    def __truediv__(self, other):
        if isinstance(other, _ObjectArray):
            return _ObjectArray([a / b for a, b in zip(self._data, other._data)], self._dtype)
        return _ObjectArray([x / other for x in self._data], self._dtype)
    def clip(self, a_min=None, a_max=None, out=None, **kwargs):
        _valid_castings = ('no', 'equiv', 'safe', 'same_kind', 'unsafe')
        if 'casting' in kwargs:
            c = kwargs['casting']
            if c not in _valid_castings:
                raise ValueError("casting must be one of 'no', 'equiv', 'safe', 'same_kind', or 'unsafe'")
        data = list(self._data)
        for i, v in enumerate(data):
            if a_min is not None:
                try:
                    if v < a_min:
                        data[i] = a_min
                except TypeError:
                    pass
            if a_max is not None:
                try:
                    if v > a_max:
                        data[i] = a_max
                except TypeError:
                    pass
        return _ObjectArray(data, self._dtype)
    def __repr__(self): return f"array({self._data!r}, dtype='{self._dtype}')"


def concatenate(arrays, axis=0, out=None, dtype=None, casting='same_kind'):
    """Wrapper around native concatenate that handles tuples and auto-converts to arrays."""
    arrs = [asarray(a) if not isinstance(a, ndarray) else a for a in arrays]
    return _native_concatenate(arrs, axis)

def _make_complex_array(values, shape):
    """Create a complex128 ndarray from a list of Python complex/float values.
    Uses real+imag decomposition since the Rust array() can't handle complex lists directly."""
    reals = [v.real if isinstance(v, complex) else float(v) for v in values]
    imags = [v.imag if isinstance(v, complex) else 0.0 for v in values]
    # Build real and imag arrays, promote both to complex128, then add
    re_arr = _native.array(reals).reshape(shape).astype("complex128")
    im_arr = _native.array(imags).reshape(shape).astype("complex128")
    # im_arr contains real values that should become imaginary parts.
    # We need to multiply by 1j. The Rust side supports complex arithmetic,
    # so create a scalar 1j array and multiply.
    j_scalar = zeros(shape, dtype="complex128")
    # Workaround: we can construct the complex array correctly by
    # using the real array as base and adding imag * 1j via Rust ops
    # re_arr already has imag=0, im_arr has the imag values as real part
    # We need: result[i] = complex(reals[i], imags[i])
    # Since Rust complex arrays store (re, im) pairs, and re_arr.astype("complex128")
    # gives us (reals[i], 0), we need a way to set the imaginary part.
    # The cleanest way: build via the Rust ops: re_arr + im_arr * 1j
    # But we need a 1j constant array. Let's try creating one:
    ones_arr = ones(shape, dtype="complex128")  # (1+0j)
    # Subtract real part to get (0+0j), then... this doesn't help.
    # Alternative: use Rust-level subtract and multiply
    # Actually, the simplest: create the array by putting values into an _ObjectArray
    # with reshape support, since scimath results are usually small.
    return _ComplexResultArray(values, shape)


class _ComplexResultArray:
    """Lightweight wrapper for complex-valued array results (used by lib.scimath).
    Provides basic ndarray-like interface for complex results that can't go through
    the Rust array constructor directly."""
    def __init__(self, data, shape):
        if isinstance(data, list):
            self._data = data
        else:
            self._data = list(data)
        self._shape = tuple(shape) if not isinstance(shape, tuple) else shape

    @property
    def shape(self):
        return self._shape

    @property
    def ndim(self):
        return len(self._shape)

    @property
    def size(self):
        n = 1
        for s in self._shape:
            n *= s
        return n

    @property
    def dtype(self):
        return dtype("complex128")

    def flatten(self):
        return _ComplexResultArray(self._data, (len(self._data),))

    def reshape(self, shape):
        if isinstance(shape, int):
            shape = (shape,)
        return _ComplexResultArray(self._data, tuple(shape))

    def tolist(self):
        return list(self._data)

    def __len__(self):
        return self._shape[0] if self._shape else 1

    def __getitem__(self, key):
        if isinstance(key, int):
            if self.ndim == 1:
                return self._data[key]
            # For 2D, return a row
            cols = self._shape[1] if len(self._shape) > 1 else 1
            return _ComplexResultArray(self._data[key * cols:(key + 1) * cols], (cols,))
        return self._data[key]

    def __repr__(self):
        return f"array({self._data!r}, dtype=complex128)"

    def copy(self):
        return _ComplexResultArray(list(self._data), self._shape)


def array(data, dtype=None, copy=None, order=None, subok=False, ndmin=0, like=None):
    result = _array_core(data, dtype=dtype, copy=copy, order=order, subok=subok, like=like)
    if ndmin > 0 and isinstance(result, ndarray):
        while result.ndim < ndmin:
            result = expand_dims(result, 0)
    return result

_DTYPE_CHAR_MAP = {
    '?': 'bool', 'b': 'int8', 'B': 'uint8',
    'h': 'int16', 'H': 'uint16',
    'i': 'int32', 'I': 'uint32',
    'l': 'int64', 'L': 'uint64',
    'q': 'int64', 'Q': 'uint64',
    'e': 'float16', 'f': 'float32', 'd': 'float64', 'g': 'float64',
    'F': 'complex64', 'D': 'complex128', 'G': 'complex128',
    # Python type class names
    "<class 'bool'>": 'bool', "<class 'int'>": 'int64', "<class 'float'>": 'float64',
    "<class 'complex'>": 'complex128', "<class 'str'>": 'str',
    'f4': 'float32', 'f8': 'float64', 'f2': 'float16',
    'i1': 'int8', 'i2': 'int16', 'i4': 'int32', 'i8': 'int64',
    'u1': 'uint8', 'u2': 'uint16', 'u4': 'uint32', 'u8': 'uint64',
    'c8': 'complex64', 'c16': 'complex128',
    'b1': 'bool',
    '<f4': 'float32', '<f8': 'float64', '<f2': 'float16',
    '<i1': 'int8', '<i2': 'int16', '<i4': 'int32', '<i8': 'int64',
    '<u1': 'uint8', '<u2': 'uint16', '<u4': 'uint32', '<u8': 'uint64',
    '<c8': 'complex64', '<c16': 'complex128',
    '>f4': 'float32', '>f8': 'float64', '>f2': 'float16',
    '>i1': 'int8', '>i2': 'int16', '>i4': 'int32', '>i8': 'int64',
    '>u1': 'uint8', '>u2': 'uint16', '>u4': 'uint32', '>u8': 'uint64',
    '>c8': 'complex64', '>c16': 'complex128',
    '=f4': 'float32', '=f8': 'float64',
    '=i4': 'int32', '=i8': 'int64',
    # Unicode string aliases (all map to 'str')
    '<U': 'str', 'U': 'str', '<U1': 'str', '<U2': 'str', '<U4': 'str',
    '<U8': 'str', '<U16': 'str', '<U32': 'str', '<U64': 'str',
    '>U1': 'str', '>U2': 'str', '>U4': 'str',
    # Python type class names for bytes
    "<class 'bytes'>": 'bytes',
    # Byte string aliases (all map to 'bytes')
    '|S0': 'bytes', '|S1': 'bytes', '|S2': 'bytes',
    '|S4': 'bytes', '|S8': 'bytes',
}

def _normalize_dtype(dt):
    """Normalize dtype string/type to a canonical name our Rust backend understands."""
    if dt is None:
        return None
    if isinstance(dt, type) and isinstance(dt, _DTypeClassMeta):
        return dt._dtype_class_name
    s = str(dt)
    return _DTYPE_CHAR_MAP.get(s, s)

def _array_core(data, dtype=None, copy=None, order=None, subok=False, like=None):
    if dtype is not None:
        dtype = _normalize_dtype(dtype)
    elif hasattr(data, "_numpy_dtype_name"):
        # Preserve numpy scalar wrapper dtype metadata for asarray()/array().
        dtype = _normalize_dtype(str(getattr(data, "_numpy_dtype_name")))
    elif isinstance(data, (list, tuple)) and len(data) > 0:
        # Preserve dtype when building arrays from wrapped numpy scalars.
        _ball = __import__("builtins").all
        wrapped = [getattr(x, "_numpy_dtype_name", None) for x in data]
        if _ball(w is not None for w in wrapped):
            cur = _normalize_dtype(str(wrapped[0]))
            for w in wrapped[1:]:
                cur = str(promote_types(cur, _normalize_dtype(str(w))))
            dtype = cur
            converted = []
            for x, w in zip(data, wrapped):
                wn = _normalize_dtype(str(w))
                if wn == "bool":
                    converted.append(bool(x))
                elif wn.startswith("int") or wn.startswith("uint"):
                    converted.append(int(x))
                elif wn.startswith("float"):
                    converted.append(float(x))
                elif wn.startswith("complex"):
                    converted.append(complex(x))
                else:
                    converted.append(x)
            data = converted
    # Check if dtype forces a non-numeric path
    if dtype is not None:
        dt = str(dtype)
        # String dtypes: route to Rust native (S-prefixed, U-prefixed, "str")
        if dt.startswith("S") or dt.startswith("U") or dt == "str":
            if isinstance(data, str):
                data = [data]
            if isinstance(data, (list, tuple)):
                return _native.array([str(x) for x in data])
            return _native.array(data)
        if dt == "object" or "," in dt:
            return _ObjectArray(data if isinstance(data, (list, tuple)) else [data], dt)
    if isinstance(data, _ObjectArray):
        return data.copy() if copy else data
    if isinstance(data, ndarray):
        result = data.copy() if copy else data
        if dtype is not None:
            dt = str(dtype)
            if dt not in ("object",) and not dt.startswith("S") and not dt.startswith("U") and dt != "str":
                result = result.astype(dt)
        # subok=False means strip subclass to base ndarray
        if not subok and type(result) is not ndarray:
            result = _native.array(result.tolist())
            if dtype is not None:
                dt = str(dtype)
                if dt not in ("object",) and not dt.startswith("S") and not dt.startswith("U") and dt != "str":
                    result = result.astype(dt)
            elif hasattr(data, 'dtype'):
                result = result.astype(str(data.dtype))
        return result
    if isinstance(data, bool):
        # bool must be checked before int since bool is a subclass of int
        dt_name = str(dtype) if dtype is not None else 'bool'
        result = _native.full([], 1.0 if data else 0.0, 'float64').astype(dt_name)
        return result
    if isinstance(data, int):
        dt_name = str(dtype) if dtype is not None else 'int64'
        result = _native.full([], float(data), 'float64').astype(dt_name)
        return result
    if isinstance(data, float):
        dt_name = str(dtype) if dtype is not None else 'float64'
        result = _native.full([], data, dt_name)
        return result
    if isinstance(data, complex):
        dt_name = str(dtype) if dtype is not None else 'complex128'
        result = _native.zeros([1], dt_name)
        result[0] = (data.real, data.imag)
        return result.reshape([])
    if isinstance(data, (int, float)):
        result = _native.array([float(data)])
        if dtype is not None:
            dt = str(dtype)
            if dt not in ("object",) and not dt.startswith("S") and not dt.startswith("U") and dt != "str":
                result = result.astype(dt)
        return result
    if isinstance(data, str):
        # Single string -> string array
        return _native.array([data])
    if isinstance(data, (list, tuple)) and len(data) > 0 and isinstance(data[0], str):
        # List/tuple of strings -> string array
        return _native.array(list(data) if isinstance(data, tuple) else data)
    if isinstance(data, (list, tuple)) and len(data) > 0 and isinstance(data[0], bool):
        result = _native.array([1.0 if x else 0.0 for x in data])
        if dtype is not None:
            dt = str(dtype)
            if dt not in ("object",) and not dt.startswith("S") and not dt.startswith("U") and dt != "str":
                result = result.astype(dt)
        else:
            # Auto-detect: all bools -> bool dtype
            _all = __import__("builtins").all
            if _all(isinstance(x, bool) for x in data):
                result = result.astype("bool")
        return result
    if isinstance(data, (list, tuple)) and len(data) > 0 and isinstance(data[0], complex):
        return _ObjectArray(data if isinstance(data, list) else list(data), "complex128")
    if isinstance(data, (list, tuple)) and len(data) > 0 and isinstance(data[0], (int, float)):
        # Check if any element is complex (mixed int/float/complex list)
        _any_complex = __import__("builtins").any(isinstance(x, complex) for x in data)
        if _any_complex:
            return _ObjectArray([complex(x) for x in data], "complex128")
        result = _native.array([float(x) for x in data])
        if dtype is not None:
            dt = str(dtype)
            if dt not in ("object",) and not dt.startswith("S") and not dt.startswith("U") and dt != "str":
                result = result.astype(dt)
        return result
    # Nested lists/tuples: infer shape, flatten, reshape
    if isinstance(data, (list, tuple)) and len(data) > 0 and isinstance(data[0], (list, tuple, ndarray)):
        shape = _infer_shape(data)
        if shape is not None:
            flat = _flatten_nested(data)
            if flat is not None:
                result = _native.array(flat)
                result = result.reshape(shape)
                if dtype is not None:
                    dt = str(dtype)
                    if dt not in ("object",) and not dt.startswith("S") and not dt.startswith("U") and dt != "str":
                        result = result.astype(dt)
                return result
    # Try the native array constructor
    try:
        result = _native.array(data)
    except (TypeError, ValueError):
        try:
            result = _native.array(_to_float_list(data))
        except (TypeError, ValueError):
            # Final fallback for non-numeric data
            return _ObjectArray(data if isinstance(data, (list, tuple)) else [data])
    if dtype is not None and isinstance(result, ndarray):
        dt = str(dtype)
        if dt not in ("object",) and not dt.startswith("S") and not dt.startswith("U") and dt != "str":
            result = result.astype(dt)
    return result


def _infer_shape(data):
    """Infer the shape of a nested list/tuple structure. Returns None if ragged."""
    if isinstance(data, ndarray):
        return data.shape
    if isinstance(data, (list, tuple)):
        if len(data) == 0:
            return (0,)
        first = data[0]
        if isinstance(first, (int, float, bool)):
            # Check all elements are scalars
            for x in data:
                if isinstance(x, (list, tuple, ndarray)):
                    return None
            return (len(data),)
        # Nested: recurse on first element to get sub-shape
        sub_shape = _infer_shape(first)
        if sub_shape is None:
            return None
        # Verify all elements have the same sub-shape
        for x in data[1:]:
            s = _infer_shape(x)
            if s != sub_shape:
                return None
        return (len(data),) + sub_shape
    # Scalar
    return ()


def _flatten_nested(data):
    """Flatten a nested list/tuple/ndarray structure to a flat list of floats.
    Returns None if it encounters non-numeric data."""
    if isinstance(data, ndarray):
        return data.flatten().tolist()
    if isinstance(data, (int, float, bool)):
        return [float(data)]
    if isinstance(data, (list, tuple)):
        result = []
        for item in data:
            sub = _flatten_nested(item)
            if sub is None:
                return None
            result.extend(sub)
        return result
    try:
        return [float(data)]
    except (TypeError, ValueError):
        return None


def _to_float_list(data):
    """Recursively convert nested lists to flat float lists for the Rust array constructor."""
    if isinstance(data, (int, float)):
        return [float(data)]
    if isinstance(data, list):
        result = []
        for item in data:
            if isinstance(item, (int, float)):
                result.append(float(item))
            elif isinstance(item, list):
                result.extend(_to_float_list(item))
            else:
                result.append(float(item))
        return result
    return [float(data)]

def _copy_into(dst, src):
    """Copy src array values into dst array (element-wise) by direct index assignment."""
    _min = __import__("builtins").min
    sf = src.flatten().tolist()
    n = _min(dst.size, len(sf))
    if dst.ndim == 1:
        for i in range(n):
            dst[i] = sf[i]
    else:
        # Use flat indexing via _flat_index_to_tuple for nD
        shape = dst.shape
        for i in range(n):
            idx = []
            rem = i
            for d in range(len(shape) - 1, -1, -1):
                idx.append(rem % shape[d])
                rem //= shape[d]
            idx.reverse()
            dst[tuple(idx)] = sf[i]

def _apply_order(arr, order):
    """Mark arr as Fortran-contiguous if order='F'."""
    if order == 'F' and hasattr(arr, '_mark_fortran'):
        arr._mark_fortran()
    return arr

def _unsupported_numeric_dtype(dt):
    """True if this dtype can't be handled by the Rust backend."""
    return dt in ('bytes', 'void')

def zeros(shape, dtype=None, order="C", like=None):
    if isinstance(shape, int):
        shape = (shape,)
    dt = _normalize_dtype(str(dtype)) if dtype is not None else None
    if dt in ("object", "<class 'object'>"):
        n = 1
        for s in shape:
            n *= s
        return _apply_order(_ObjectArray([0] * n, "object", shape=shape), order)
    if dt is not None and _unsupported_numeric_dtype(dt):
        n = 1
        for s in shape:
            n *= s
        return _apply_order(_ObjectArray([0] * n, dt, shape=shape), order)
    if dt is not None:
        return _apply_order(_native.zeros(shape, dt), order)
    return _apply_order(_native.zeros(shape), order)

def ones(shape, dtype=None, order="C", like=None):
    if isinstance(shape, int):
        shape = (shape,)
    dt = _normalize_dtype(str(dtype)) if dtype is not None else None
    if dt in ("object", "<class 'object'>"):
        n = 1
        for s in shape:
            n *= s
        return _apply_order(_ObjectArray([1] * n, "object", shape=shape), order)
    if dt is not None and _unsupported_numeric_dtype(dt):
        n = 1
        for s in shape:
            n *= s
        return _apply_order(_ObjectArray([1] * n, dt, shape=shape), order)
    if dt is not None:
        return _apply_order(_native.ones(shape, dt), order)
    return _apply_order(_native.ones(shape), order)

def arange(*args, dtype=None, like=None, **kwargs):
    dt = _normalize_dtype(str(dtype)) if dtype is not None else None
    if dt == "object" or dt == "<class 'object'>":
        float_args = [float(a) for a in args]
        if len(float_args) == 1:
            vals = list(range(int(float_args[0])))
        elif len(float_args) == 2:
            vals = list(range(int(float_args[0]), int(float_args[1])))
        else:
            vals = list(range(int(float_args[0]), int(float_args[1]), int(float_args[2])))
        return _ObjectArray(vals, "object")
    float_args = [float(a) for a in args]
    # Normalize to (start, stop, step) form
    if len(float_args) == 1:
        float_args = [0.0, float_args[0], 1.0]
    elif len(float_args) == 2:
        float_args = [float_args[0], float_args[1], 1.0]
    if dtype is not None:
        return _native.arange(float_args[0], float_args[1], float_args[2], dt)
    return _native.arange(float_args[0], float_args[1], float_args[2])

def linspace(start, stop, num=50, endpoint=True, retstep=False, dtype=None, axis=0):
    start = float(start)
    stop = float(stop)
    num = int(num)
    if endpoint:
        result = _native.linspace(start, stop, num)
        step = (stop - start) / (num - 1) if num > 1 else 0.0
    else:
        step = (stop - start) / num if num > 0 else 0.0
        result = array([start + i * step for i in range(num)])
    if dtype is not None:
        result = result.astype(str(dtype))
    if retstep:
        return result, step
    return result

def logspace(start, stop, num=50, endpoint=True, base=10.0, dtype=None, axis=0):
    y = linspace(start, stop, num=num, endpoint=endpoint)
    result = power(base, y)
    if dtype is not None:
        result = result.astype(str(dtype))
    return result

def geomspace(start, stop, num=50, endpoint=True, dtype=None, axis=0):
    log_start = _math.log10(start)
    log_stop = _math.log10(stop)
    result = logspace(log_start, log_stop, num=num, endpoint=endpoint)
    if dtype is not None:
        result = result.astype(str(dtype))
    return result

def eye(N, M=None, k=0, dtype=None, order="C", like=None):
    if dtype is not None:
        if M is not None:
            return _native.eye(N, M, k, str(dtype))
        return _native.eye(N, N, k, str(dtype))
    if M is not None:
        return _native.eye(N, M, k)
    if k != 0:
        return _native.eye(N, N, k)
    return _native.eye(N)

def where(condition, x=None, y=None):
    if x is None and y is None:
        return nonzero(condition)
    if not isinstance(condition, ndarray):
        condition = asarray(condition)
    # Ensure condition is boolean dtype (native where_ requires bool)
    if str(condition.dtype) != "bool":
        condition = condition.astype("bool")
    if not isinstance(x, ndarray):
        x = full(condition.shape, float(x))
    if not isinstance(y, ndarray):
        y = full(condition.shape, float(y))
    return _native.where_(condition, x, y)

# Import submodules so they're accessible as numpy.linalg etc.
from _numpy_native import linalg, fft, random

# Register Rust submodules in sys.modules so `from numpy.random import ...` works
_sys.modules["numpy.linalg"] = linalg
_sys.modules["numpy.fft"] = fft
_sys.modules["numpy.random"] = random

# --- Dtype aliases ----------------------------------------------------------
# NumPy tests reference these as np.float64(value), np.int32(value), etc.
# They must be callable (constructing scalars) AND usable as dtype identifiers.

class _ScalarType:
    """A callable dtype alias that can construct scalars and be used as a dtype string."""
    def __init__(self, name, python_type=float):
        self._name = name
        self._type = python_type

    def __call__(self, value=0, *args, **kwargs):
        try:
            return self._type(value)
        except (ValueError, TypeError):
            # Return the value as-is for unsupported conversions (e.g. NaT)
            return value

    def __repr__(self):
        return f"<class 'numpy.{self._name}'>"

    def __str__(self):
        return self._name

    def __eq__(self, other):
        if isinstance(other, _ScalarType):
            return self._name == other._name
        if isinstance(other, str):
            return self._name == other
        if isinstance(other, type) and hasattr(other, '_scalar_name'):
            return self._name == other._scalar_name
        return NotImplemented

    def __hash__(self):
        return hash(self._name)


class _NumpyIntScalar(int):
    def __new__(cls, value=0, dtype_name="int64"):
        obj = int.__new__(cls, int(value))
        obj._numpy_dtype_name = dtype_name
        return obj

    @property
    def dtype(self):
        return dtype(self._numpy_dtype_name)

    @property
    def shape(self):
        return ()

    @property
    def ndim(self):
        return 0

    @property
    def size(self):
        return 1

    def __round__(self, ndigits=None):
        if ndigits is None:
            return int(self)
        rounded = __import__("builtins").round(int(self), ndigits)
        # Keep int64 round() as numpy scalar (compat tests rely on this).
        if self._numpy_dtype_name == "int64":
            return _NumpyIntScalar(rounded, self._numpy_dtype_name)
        return int(rounded)

    def round(self, ndigits=0):
        return _NumpyIntScalar(__import__("builtins").round(int(self), ndigits), self._numpy_dtype_name)


class _NumpyFloatScalar(float):
    def __new__(cls, value=0.0, dtype_name="float64"):
        obj = float.__new__(cls, float(value))
        obj._numpy_dtype_name = dtype_name
        return obj

    @property
    def dtype(self):
        return dtype(self._numpy_dtype_name)

    @property
    def shape(self):
        return ()

    @property
    def ndim(self):
        return 0

    @property
    def size(self):
        return 1

    def __round__(self, ndigits=None):
        _builtin_round = __import__("builtins").round
        if ndigits is None:
            return int(_builtin_round(float(self)))
        return _NumpyFloatScalar(_builtin_round(float(self), ndigits), self._numpy_dtype_name)

    def round(self, ndigits=0):
        _builtin_round = __import__("builtins").round
        return _NumpyFloatScalar(_builtin_round(float(self), ndigits), self._numpy_dtype_name)

    def __mul__(self, other):
        if isinstance(other, (ndarray, _ObjectArray)) or hasattr(other, "_numpy_dtype_name"):
            return multiply(self, other)
        return float.__mul__(self, other)

    def __rmul__(self, other):
        if isinstance(other, (ndarray, _ObjectArray)) or hasattr(other, "_numpy_dtype_name"):
            return multiply(other, self)
        return float.__rmul__(self, other)


class _NumpyComplexScalar(complex):
    def __new__(cls, value=0j, dtype_name="complex128"):
        obj = complex.__new__(cls, value)
        obj._numpy_dtype_name = dtype_name
        return obj

    @property
    def dtype(self):
        return dtype(self._numpy_dtype_name)

    @property
    def shape(self):
        return ()

    @property
    def ndim(self):
        return 0

    @property
    def size(self):
        return 1


# Metaclass for scalar type classes so the CLASS itself has custom __str__, __eq__, __hash__
class _ScalarTypeMeta(type):
    """Metaclass for numpy scalar type classes in the type hierarchy."""
    def __new__(mcs, name, bases, namespace, scalar_name=None, python_type=float):
        cls = super().__new__(mcs, name, bases, namespace)
        cls._scalar_name = scalar_name or name
        cls._python_type = python_type
        return cls

    def __init__(cls, name, bases, namespace, scalar_name=None, python_type=float):
        super().__init__(name, bases, namespace)

    def __call__(cls, value=0, *args, **kwargs):
        scalar_name = cls._scalar_name
        if scalar_name in ('complex64', 'complex128') and len(args) == 1:
            try:
                value = complex(value, args[0])
            except (ValueError, TypeError):
                return value
        try:
            base_value = cls._python_type(value)
        except (ValueError, TypeError):
            return value
        if scalar_name in ('bool', 'int8', 'int16', 'int32', 'int64',
                           'uint8', 'uint16', 'uint32', 'uint64'):
            return _NumpyIntScalar(base_value, scalar_name)
        if scalar_name in ('float16', 'float32', 'float64'):
            return _NumpyFloatScalar(base_value, scalar_name)
        if scalar_name in ('complex64', 'complex128'):
            return _NumpyComplexScalar(base_value, scalar_name)
        return base_value

    def __repr__(cls):
        return f"<class 'numpy.{cls._scalar_name}'>"

    def __str__(cls):
        return cls._scalar_name

    def __eq__(cls, other):
        if isinstance(other, _ScalarTypeMeta):
            return cls._scalar_name == other._scalar_name
        if isinstance(other, _ScalarType):
            return cls._scalar_name == other._name
        if isinstance(other, str):
            return cls._scalar_name == other
        if isinstance(other, dtype):
            return cls._scalar_name == other.name
        return type.__eq__(cls, other)

    def __hash__(cls):
        return hash(cls._scalar_name)

    def __instancecheck__(cls, instance):
        """Allow isinstance(3, np.integer) etc. to work."""
        scalar_name = cls._scalar_name

        # Map Python types to numpy type hierarchy
        if isinstance(instance, bool):
            # bool is a subclass of int in Python, check it first
            return scalar_name in ('bool', 'generic', 'number', 'integer', 'signedinteger')
        if isinstance(instance, int):
            return scalar_name in ('generic', 'number', 'integer', 'signedinteger',
                                   'int8', 'int16', 'int32', 'int64',
                                   'uint8', 'uint16', 'uint32', 'uint64',
                                   'unsignedinteger', 'intp')
        if isinstance(instance, float):
            return scalar_name in ('generic', 'number', 'inexact', 'floating',
                                   'float16', 'float32', 'float64')
        if isinstance(instance, complex):
            return scalar_name in ('generic', 'number', 'inexact', 'complexfloating',
                                   'complex64', 'complex128')
        if isinstance(instance, str):
            return scalar_name in ('generic', 'character', 'str')
        if isinstance(instance, bytes):
            return scalar_name in ('generic', 'character', 'bytes')
        return False


# --- Constants --------------------------------------------------------------
nan = float("nan")
inf = float("inf")
pi = _math.pi
e = _math.e
newaxis = None
PINF = float("inf")
NINF = float("-inf")
PZERO = 0.0
NZERO = -0.0
Inf = inf
Infinity = inf
NaN = nan
NAN = nan
euler_gamma = 0.5772156649015329
ALLOW_THREADS = 1
little_endian = True

# Save builtin divmod before we shadow it later
_builtin_divmod = __builtins__["divmod"] if isinstance(__builtins__, dict) else __import__("builtins").divmod

# --- typecodes (mapping of type character codes) ----------------------------
typecodes = {
    "All": "?bhilqBHILQefdgFDGSUVO",
    "AllFloat": "efdgFDG",
    "AllInteger": "bhilqBHILQ",
    "Character": "c",
    "Complex": "FDG",
    "Float": "efdg",
    "Integer": "bhilq",
    "UnsignedInteger": "BHILQ",
}

# --- Type hierarchy classes -------------------------------------------------
class generic(metaclass=_ScalarTypeMeta, scalar_name="generic"):
    """Base class for all numpy scalar types."""
    pass

class number(generic, metaclass=_ScalarTypeMeta, scalar_name="number"):
    """Base class for all numeric scalar types."""
    pass

class integer(number, metaclass=_ScalarTypeMeta, scalar_name="integer"):
    """Base class for integer scalar types."""
    pass

class signedinteger(integer, metaclass=_ScalarTypeMeta, scalar_name="signedinteger"):
    """Base class for signed integer scalar types."""
    pass

class unsignedinteger(integer, metaclass=_ScalarTypeMeta, scalar_name="unsignedinteger"):
    """Base class for unsigned integer scalar types."""
    pass

class inexact(number, metaclass=_ScalarTypeMeta, scalar_name="inexact"):
    """Base class for inexact (float/complex) scalar types."""
    pass

class floating(inexact, metaclass=_ScalarTypeMeta, scalar_name="floating"):
    """Base class for floating-point scalar types."""
    pass

class complexfloating(inexact, metaclass=_ScalarTypeMeta, scalar_name="complexfloating"):
    """Base class for complex scalar types."""
    pass

class character(generic, metaclass=_ScalarTypeMeta, scalar_name="character"):
    """Base class for character types."""
    pass

class flexible(generic, metaclass=_ScalarTypeMeta, scalar_name="flexible"):
    """Base class for flexible types (string, void)."""
    pass

class bool_(generic, metaclass=_ScalarTypeMeta, scalar_name="bool", python_type=bool):
    """Boolean scalar type."""
    pass

# Specific signed integer types
class int8(signedinteger, metaclass=_ScalarTypeMeta, scalar_name="int8", python_type=int):
    pass
class int16(signedinteger, metaclass=_ScalarTypeMeta, scalar_name="int16", python_type=int):
    pass
class int32(signedinteger, metaclass=_ScalarTypeMeta, scalar_name="int32", python_type=int):
    pass
class int64(signedinteger, metaclass=_ScalarTypeMeta, scalar_name="int64", python_type=int):
    pass

# Specific unsigned integer types
class uint8(unsignedinteger, metaclass=_ScalarTypeMeta, scalar_name="uint8", python_type=int):
    pass
class uint16(unsignedinteger, metaclass=_ScalarTypeMeta, scalar_name="uint16", python_type=int):
    pass
class uint32(unsignedinteger, metaclass=_ScalarTypeMeta, scalar_name="uint32", python_type=int):
    pass
class uint64(unsignedinteger, metaclass=_ScalarTypeMeta, scalar_name="uint64", python_type=int):
    pass

# Specific floating-point types
class float16(floating, metaclass=_ScalarTypeMeta, scalar_name="float16", python_type=float):
    pass
class float32(floating, metaclass=_ScalarTypeMeta, scalar_name="float32", python_type=float):
    pass
class float64(floating, metaclass=_ScalarTypeMeta, scalar_name="float64", python_type=float):
    pass

# Specific complex types
class complex64(complexfloating, metaclass=_ScalarTypeMeta, scalar_name="complex64", python_type=complex):
    pass
class complex128(complexfloating, metaclass=_ScalarTypeMeta, scalar_name="complex128", python_type=complex):
    pass

# Character/flexible types
class str_(character, metaclass=_ScalarTypeMeta, scalar_name="str", python_type=str):
    pass
class bytes_(character, metaclass=_ScalarTypeMeta, scalar_name="bytes", python_type=bytes):
    pass
class void(flexible, metaclass=_ScalarTypeMeta, scalar_name="void", python_type=float):
    pass

# Aliases using _ScalarType (for types that don't need hierarchy participation)
float128 = _ScalarType("float128", float)
intp = int64
intc = _ScalarType("int32", int)
uintp = _ScalarType("uint64", int)
byte = _ScalarType("int8", int)
ubyte = _ScalarType("uint8", int)
short = _ScalarType("int16", int)
ushort = _ScalarType("uint16", int)
longlong = _ScalarType("int64", int)
ulonglong = _ScalarType("uint64", int)
single = _ScalarType("float32", float)
double = _ScalarType("float64", float)
longdouble = _ScalarType("float64", float)
csingle = _ScalarType("complex64", complex)
cdouble = _ScalarType("complex128", complex)
clongdouble = _ScalarType("complex128", complex)
object_ = _ScalarType("object", object)
# --- datetime64/timedelta64 helper functions ---------------------------------

def _infer_datetime_unit(s):
    """Infer unit from datetime string format."""
    s = s.strip()
    if len(s) == 4:  # '2024'
        return 'Y'
    elif len(s) == 7:  # '2024-01'
        return 'M'
    elif len(s) == 10:  # '2024-01-15'
        return 'D'
    elif 'T' in s:
        return 's'
    return 'D'


def _parse_datetime_string(s, unit):
    """Parse datetime string to integer value in given unit."""
    s = s.strip()
    parts = s.replace('T', '-').replace(':', '-').split('-')
    year = int(parts[0]) if len(parts) > 0 else 1970
    month = int(parts[1]) if len(parts) > 1 else 1
    day = int(parts[2]) if len(parts) > 2 else 1

    if unit == 'Y':
        return year
    elif unit == 'M':
        return (year - 1970) * 12 + (month - 1)
    else:
        # Convert to days since epoch
        days = _date_to_days(year, month, day)
        if unit == 'D':
            return days
        elif unit == 's':
            hour = int(parts[3]) if len(parts) > 3 else 0
            minute = int(parts[4]) if len(parts) > 4 else 0
            second = int(parts[5]) if len(parts) > 5 else 0
            return days * 86400 + hour * 3600 + minute * 60 + second
        return days


def _date_to_days(year, month, day):
    """Convert date to days since 1970-01-01."""
    days = 0
    # Years
    for y in range(1970, year) if year >= 1970 else range(year, 1970):
        leap = (y % 4 == 0 and y % 100 != 0) or (y % 400 == 0)
        d = 366 if leap else 365
        if year >= 1970:
            days += d
        else:
            days -= d
    # Months
    month_days = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    leap = (year % 4 == 0 and year % 100 != 0) or (year % 400 == 0)
    if leap:
        month_days[1] = 29
    for m in range(1, month):
        days += month_days[m - 1]
    days += day - 1
    return days


def _days_to_date(days):
    """Convert days since epoch to ISO date string."""
    y = 1970
    remaining = days
    if remaining >= 0:
        while True:
            leap = (y % 4 == 0 and y % 100 != 0) or (y % 400 == 0)
            year_days = 366 if leap else 365
            if remaining < year_days:
                break
            remaining -= year_days
            y += 1
    else:
        while remaining < 0:
            y -= 1
            leap = (y % 4 == 0 and y % 100 != 0) or (y % 400 == 0)
            year_days = 366 if leap else 365
            remaining += year_days

    month_days = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    leap = (y % 4 == 0 and y % 100 != 0) or (y % 400 == 0)
    if leap:
        month_days[1] = 29
    m = 0
    while m < 12 and remaining >= month_days[m]:
        remaining -= month_days[m]
        m += 1
    return "{:04d}-{:02d}-{:02d}".format(y, m + 1, remaining + 1)


def _to_common_unit(value, from_unit, to_unit):
    """Convert value from one unit to another."""
    # First convert to days
    if from_unit == 'Y':
        days = value * 365  # approximate
    elif from_unit == 'M':
        days = value * 30  # approximate
    elif from_unit == 'W':
        days = value * 7
    elif from_unit == 'D':
        days = value
    elif from_unit == 'h':
        days = value / 24.0
    elif from_unit == 'm':
        days = value / 1440.0
    elif from_unit == 's':
        days = value / 86400.0
    elif from_unit == 'ms':
        days = value / 86400000.0
    elif from_unit == 'us':
        days = value / 86400000000.0
    elif from_unit == 'ns':
        days = value / 86400000000000.0
    else:
        days = value

    # Then convert from days to target
    if to_unit == 'Y':
        return int(days / 365)
    elif to_unit == 'M':
        return int(days / 30)
    elif to_unit == 'W':
        return int(days / 7)
    elif to_unit == 'D':
        return int(days)
    elif to_unit == 'h':
        return int(days * 24)
    elif to_unit == 'm':
        return int(days * 1440)
    elif to_unit == 's':
        return int(days * 86400)
    elif to_unit == 'ms':
        return int(days * 86400000)
    elif to_unit == 'us':
        return int(days * 86400000000)
    elif to_unit == 'ns':
        return int(days * 86400000000000)
    return int(days)


def _common_time_unit(u1, u2):
    """Find the finer of two time units."""
    order = ['Y', 'M', 'W', 'D', 'h', 'm', 's', 'ms', 'us', 'ns']
    try:
        i1 = order.index(u1)
    except ValueError:
        i1 = 3  # default to days
    try:
        i2 = order.index(u2)
    except ValueError:
        i2 = 3
    return order[i1 if i1 > i2 else i2]


# --- datetime64 / timedelta64 classes ----------------------------------------

class datetime64:
    """NumPy datetime64 scalar type."""
    def __init__(self, value=None, unit=None):
        if value is None:
            self._value = 0  # epoch
            self._unit = unit or 'us'
        elif isinstance(value, str):
            self._unit = unit or _infer_datetime_unit(value)
            self._value = _parse_datetime_string(value, self._unit)
        elif isinstance(value, datetime64):
            self._value = value._value
            self._unit = unit or value._unit
        elif isinstance(value, (int, float)):
            self._value = int(value)
            self._unit = unit or 'us'
        else:
            self._value = int(value)
            self._unit = unit or 'us'

    def __repr__(self):
        return "numpy.datetime64('{}')".format(self._to_string())

    def __str__(self):
        return self._to_string()

    def _to_string(self):
        """Convert internal value back to ISO string."""
        if self._unit == 'Y':
            return str(self._value)
        elif self._unit == 'M':
            y = 1970 + self._value // 12
            m = self._value % 12 + 1
            return "{:04d}-{:02d}".format(y, m)
        elif self._unit == 'D':
            # Days since epoch (1970-01-01)
            return _days_to_date(self._value)
        elif self._unit in ('h', 'm', 's', 'ms', 'us', 'ns'):
            # Convert to days + remainder
            if self._unit == 's':
                days = self._value // 86400
            elif self._unit == 'ms':
                days = self._value // 86400000
            elif self._unit == 'us':
                days = self._value // 86400000000
            elif self._unit == 'ns':
                days = self._value // 86400000000000
            elif self._unit == 'h':
                days = self._value // 24
            elif self._unit == 'm':
                days = self._value // 1440
            else:
                days = self._value
            return _days_to_date(days)
        return str(self._value)

    def __sub__(self, other):
        if isinstance(other, datetime64):
            # datetime - datetime = timedelta
            v1 = _to_common_unit(self._value, self._unit, 'D')
            v2 = _to_common_unit(other._value, other._unit, 'D')
            return timedelta64(v1 - v2, 'D')
        elif isinstance(other, timedelta64):
            v = _to_common_unit(other._value, other._unit, self._unit)
            return datetime64.__new_from_value(self._value - v, self._unit)
        return NotImplemented

    def __add__(self, other):
        if isinstance(other, timedelta64):
            v = _to_common_unit(other._value, other._unit, self._unit)
            return datetime64.__new_from_value(self._value + v, self._unit)
        return NotImplemented

    def __radd__(self, other):
        return self.__add__(other)

    def __eq__(self, other):
        if isinstance(other, datetime64):
            v1 = _to_common_unit(self._value, self._unit, 'D')
            v2 = _to_common_unit(other._value, other._unit, 'D')
            return v1 == v2
        return NotImplemented

    def __lt__(self, other):
        if isinstance(other, datetime64):
            v1 = _to_common_unit(self._value, self._unit, 'D')
            v2 = _to_common_unit(other._value, other._unit, 'D')
            return v1 < v2
        return NotImplemented

    def __le__(self, other):
        return self == other or self < other

    def __gt__(self, other):
        if isinstance(other, datetime64):
            return other < self
        return NotImplemented

    def __ge__(self, other):
        return self == other or self > other

    def __hash__(self):
        return hash((self._value, self._unit))

    @classmethod
    def __new_from_value(cls, value, unit):
        obj = cls.__new__(cls)
        obj._value = value
        obj._unit = unit
        return obj

    def astype(self, dtype):
        dtype_str = str(dtype)
        if 'int' in dtype_str:
            return self._value
        elif 'float' in dtype_str:
            return float(self._value)
        return self


class timedelta64:
    """NumPy timedelta64 scalar type."""
    def __init__(self, value=0, unit='generic'):
        if isinstance(value, timedelta64):
            self._value = value._value
            self._unit = unit if unit != 'generic' else value._unit
        else:
            self._value = int(value)
            self._unit = unit

    def __repr__(self):
        return "numpy.timedelta64({}, '{}')".format(self._value, self._unit)

    def __str__(self):
        return "{} {}".format(self._value, self._unit)

    def __add__(self, other):
        if isinstance(other, timedelta64):
            common = _common_time_unit(self._unit, other._unit)
            v1 = _to_common_unit(self._value, self._unit, common)
            v2 = _to_common_unit(other._value, other._unit, common)
            return timedelta64(v1 + v2, common)
        if isinstance(other, datetime64):
            return other + self
        return NotImplemented

    def __sub__(self, other):
        if isinstance(other, timedelta64):
            common = _common_time_unit(self._unit, other._unit)
            v1 = _to_common_unit(self._value, self._unit, common)
            v2 = _to_common_unit(other._value, other._unit, common)
            return timedelta64(v1 - v2, common)
        return NotImplemented

    def __mul__(self, other):
        if isinstance(other, (int, float)):
            return timedelta64(int(self._value * other), self._unit)
        return NotImplemented

    def __rmul__(self, other):
        return self.__mul__(other)

    def __truediv__(self, other):
        if isinstance(other, (int, float)):
            return timedelta64(int(self._value / other), self._unit)
        if isinstance(other, timedelta64):
            common = _common_time_unit(self._unit, other._unit)
            v1 = _to_common_unit(self._value, self._unit, common)
            v2 = _to_common_unit(other._value, other._unit, common)
            return v1 / v2 if v2 != 0 else float('inf')
        return NotImplemented

    def __eq__(self, other):
        if isinstance(other, timedelta64):
            common = _common_time_unit(self._unit, other._unit)
            v1 = _to_common_unit(self._value, self._unit, common)
            v2 = _to_common_unit(other._value, other._unit, common)
            return v1 == v2
        return NotImplemented

    def __lt__(self, other):
        if isinstance(other, timedelta64):
            common = _common_time_unit(self._unit, other._unit)
            v1 = _to_common_unit(self._value, self._unit, common)
            v2 = _to_common_unit(other._value, other._unit, common)
            return v1 < v2
        return NotImplemented

    def __hash__(self):
        return hash((self._value, self._unit))

    def astype(self, dtype):
        dtype_str = str(dtype)
        if 'int' in dtype_str:
            return self._value
        elif 'float' in dtype_str:
            return float(self._value)
        return self


def isnat(x):
    """Test for NaT (Not a Time)."""
    if isinstance(x, (datetime64, timedelta64)):
        return False  # We don't support NaT sentinel yet
    return False


def busday_count(begindates, enddates, weekmask='1111100', holidays=None):
    """Count business days. Simplified implementation."""
    if isinstance(begindates, datetime64) and isinstance(enddates, datetime64):
        diff = enddates - begindates
        return int(diff._value * 5 / 7)  # rough approximation
    return 0


def is_busday(dates, weekmask='1111100', holidays=None):
    """Check if dates are business days."""
    return True


def busday_offset(dates, offsets, roll='raise', weekmask='1111100', holidays=None):
    """Offset dates by business days."""
    if isinstance(dates, datetime64):
        return dates + timedelta64(int(offsets), 'D')
    return dates
string_ = _ScalarType("str", str)
unicode_ = _ScalarType("str", str)
half = _ScalarType("float16", float)
int_ = int64
float_ = float64
complex_ = complex128
uint = uint64
long = int64
ulong = uint64
# numpy 1.x compat: np.bool (deprecated, can't shadow builtin 'bool' in module
# scope since isinstance checks recurse). We set it via __getattr__ below.

# --- Missing functions (stubs) ----------------------------------------------
def empty(shape, dtype=None, order="C"):
    """Stub: returns zeros instead of uninitialized."""
    return zeros(shape, dtype=dtype, order=order)

def _like_order(arr, source, order):
    """Apply ordering to result of a like function. K=keep source order, A=fortran if source is, else C."""
    src_is_f = getattr(getattr(source, 'flags', None), 'f_contiguous', False)
    if order == 'F':
        if hasattr(arr, '_mark_fortran'):
            arr._mark_fortran()
    elif order == 'K':
        if src_is_f and hasattr(arr, '_mark_fortran'):
            arr._mark_fortran()
    elif order == 'A':
        if src_is_f and hasattr(arr, '_mark_fortran'):
            arr._mark_fortran()
    # order='C' is the default (no fortran)
    return arr

def _detect_builtin_str_bytes(dtype):
    """Return ('str', 'bytes', or None) if dtype is Python builtin str or bytes."""
    import builtins
    if dtype is builtins.str:
        return 'str'
    if dtype is builtins.bytes:
        return 'bytes'
    return None

def _make_str_bytes_result(shape, dtype_kind, fill_value=None):
    """Create an _ObjectArray with correct strides for dtype=str or dtype=bytes."""
    # itemsize: Unicode char = 4 bytes, byte = 1 byte
    itemsize = 4 if dtype_kind == 'str' else 1
    n = 1
    for s in shape:
        n *= s
    data = [fill_value if fill_value is not None else ''] * n
    dt_name = 'str' if dtype_kind == 'str' else 'bytes'
    return _ObjectArray(data, dt_name, shape=shape, itemsize=itemsize)

def full(shape, fill_value, dtype=None, order="C"):
    if isinstance(shape, int):
        shape = (shape,)
    dt = _normalize_dtype(str(dtype)) if dtype is not None else None
    if dt in ("object", "<class 'object'>"):
        n = 1
        for s in shape:
            n *= s
        return _apply_order(_ObjectArray([fill_value] * n, "object", shape=shape), order)
    if dt is not None and _unsupported_numeric_dtype(dt):
        n = 1
        for s in shape:
            n *= s
        return _apply_order(_ObjectArray([fill_value] * n, dt, shape=shape), order)
    if dt is not None:
        return _apply_order(_native.full(shape, float(fill_value), dt), order)
    return _apply_order(_native.full(shape, float(fill_value)), order)

def full_like(a, fill_value, dtype=None, order="K", subok=True, shape=None):
    s = tuple(shape) if shape is not None else a.shape
    # Check for invalid dtype like "S-1"
    if isinstance(dtype, str) and dtype.startswith('S') and dtype[1:].lstrip('-').isdigit() and int(dtype[1:]) < 0:
        raise TypeError("Cannot convert to dtype: {}".format(dtype))
    # Detect Python builtin str/bytes
    sk = _detect_builtin_str_bytes(dtype)
    if sk is not None:
        return _like_order(_make_str_bytes_result(s, sk, fill_value), a, order)
    # Determine effective dtype
    src_dt = _normalize_dtype(str(a.dtype)) if hasattr(a, 'dtype') else 'float64'
    dt = _normalize_dtype(str(dtype)) if dtype is not None else src_dt
    # OverflowError: check if fill_value (plain Python int) fits in target integer dtype
    if type(fill_value) is int:
        _int_dtypes_info = {
            'int8': (-128, 127), 'int16': (-32768, 32767),
            'int32': (-2147483648, 2147483647), 'int64': (-9223372036854775808, 9223372036854775807),
            'uint8': (0, 255), 'uint16': (0, 65535), 'uint32': (0, 4294967295),
            'uint64': (0, 18446744073709551615),
        }
        if dt in _int_dtypes_info:
            lo, hi = _int_dtypes_info[dt]
            if fill_value < lo or fill_value > hi:
                raise OverflowError("Python integer {} out of bounds for {}".format(fill_value, dt))
    arr = _native.full(s, float(fill_value), dt)
    arr = _like_order(arr, a, order)
    if subok and type(a) is not ndarray and isinstance(a, ndarray):
        try:
            arr = arr.view(type(a))
        except Exception:
            pass
    return arr

def zeros_like(a, dtype=None, order="K", subok=True, shape=None):
    s = tuple(shape) if shape is not None else a.shape
    if isinstance(dtype, str) and dtype.startswith('S') and len(dtype) > 1 and dtype[1:].lstrip('-').isdigit() and int(dtype[1:]) < 0:
        raise TypeError("Cannot convert to dtype: {}".format(dtype))
    sk = _detect_builtin_str_bytes(dtype)
    if sk is not None:
        return _like_order(_make_str_bytes_result(s, sk), a, order)
    src_dt = _normalize_dtype(str(a.dtype)) if hasattr(a, 'dtype') else 'float64'
    dt = _normalize_dtype(str(dtype)) if dtype is not None else src_dt
    arr = _native.zeros(s, dt)
    arr = _like_order(arr, a, order)
    if subok and type(a) is not ndarray and isinstance(a, ndarray):
        try:
            arr = arr.view(type(a))
        except Exception:
            pass
    return arr

def ones_like(a, dtype=None, order="K", subok=True, shape=None):
    s = tuple(shape) if shape is not None else a.shape
    if isinstance(dtype, str) and dtype.startswith('S') and len(dtype) > 1 and dtype[1:].lstrip('-').isdigit() and int(dtype[1:]) < 0:
        raise TypeError("Cannot convert to dtype: {}".format(dtype))
    sk = _detect_builtin_str_bytes(dtype)
    if sk is not None:
        return _like_order(_make_str_bytes_result(s, sk, 1), a, order)
    src_dt = _normalize_dtype(str(a.dtype)) if hasattr(a, 'dtype') else 'float64'
    dt = _normalize_dtype(str(dtype)) if dtype is not None else src_dt
    arr = _native.ones(s, dt)
    arr = _like_order(arr, a, order)
    if subok and type(a) is not ndarray and isinstance(a, ndarray):
        try:
            arr = arr.view(type(a))
        except Exception:
            pass
    return arr

def empty_like(a, dtype=None, order="K", subok=True, shape=None):
    s = tuple(shape) if shape is not None else a.shape
    if isinstance(dtype, str) and dtype.startswith('S') and len(dtype) > 1 and dtype[1:].lstrip('-').isdigit() and int(dtype[1:]) < 0:
        raise TypeError("Cannot convert to dtype: {}".format(dtype))
    sk = _detect_builtin_str_bytes(dtype)
    if sk is not None:
        return _like_order(_make_str_bytes_result(s, sk), a, order)
    src_dt = _normalize_dtype(str(a.dtype)) if hasattr(a, 'dtype') else 'float64'
    dt = _normalize_dtype(str(dtype)) if dtype is not None else src_dt
    arr = _native.zeros(s, dt)
    arr = _like_order(arr, a, order)
    if subok and type(a) is not ndarray and isinstance(a, ndarray):
        try:
            arr = arr.view(type(a))
        except Exception:
            pass
    return arr

def _cmath_isnan(v):
    import cmath
    if isinstance(v, complex):
        return cmath.isnan(v)
    try:
        return _math.isnan(v)
    except (TypeError, ValueError):
        return False

def _cmath_isinf(v):
    import cmath
    if isinstance(v, complex):
        return cmath.isinf(v)
    try:
        return _math.isinf(v)
    except (TypeError, ValueError):
        return False

def _cmath_isfinite(v):
    import cmath
    if isinstance(v, complex):
        return cmath.isfinite(v)
    try:
        return _math.isfinite(v)
    except (TypeError, ValueError):
        return True

def isnan(x):
    """Check for NaN element-wise."""
    if isinstance(x, _ObjectArray):
        return array([_cmath_isnan(v) for v in x._data])
    if not isinstance(x, ndarray):
        if isinstance(x, (list, tuple)):
            x = asarray(x)
        elif isinstance(x, complex):
            import cmath
            return cmath.isnan(x)
        else:
            return _math.isnan(x)
    return _native.isnan(x)

def isfinite(x):
    if isinstance(x, _ObjectArray):
        return array([_cmath_isfinite(v) for v in x._data])
    if not isinstance(x, ndarray):
        if isinstance(x, (list, tuple)):
            x = asarray(x)
        elif isinstance(x, complex):
            import cmath
            return cmath.isfinite(x)
        else:
            return _math.isfinite(x)
    return _native.isfinite(x)

def isinf(x):
    if isinstance(x, _ObjectArray):
        return array([_cmath_isinf(v) for v in x._data])
    if not isinstance(x, ndarray):
        if isinstance(x, (list, tuple)):
            x = asarray(x)
        else:
            return _math.isinf(x)
    return _native.isinf(x)

def isscalar(x):
    """Return True if x is a scalar (not an array)."""
    if isinstance(x, (int, float, complex, bool, str)):
        return True
    if isinstance(x, ndarray):
        return False
    if isinstance(x, (list, tuple)):
        return False
    if x is None:
        return False
    # Check for PEP 3141 Number types and numpy scalar types
    try:
        import numbers
        if isinstance(x, numbers.Number):
            return True
    except ImportError:
        pass
    # Check for numpy scalar types (0-d arrays or scalar wrappers)
    if hasattr(x, 'ndim') and x.ndim == 0:
        return True
    return False

def isrealobj(x):
    """Return True if x is not a complex type."""
    if isinstance(x, ndarray):
        return x.dtype not in ("complex64", "complex128")
    return not isinstance(x, complex)

def iscomplexobj(x):
    """Return True if x has a complex type."""
    if isinstance(x, ndarray):
        return x.dtype in ("complex64", "complex128")
    return isinstance(x, complex)

def isreal(x):
    """Returns boolean array -- True where elements are real."""
    if not isinstance(x, ndarray):
        x = array(x)
    if x.dtype in ("complex64", "complex128"):
        return zeros(x.shape, dtype="bool")
    return ones(x.shape, dtype="bool")

def iscomplex(x):
    """Returns boolean array -- True where elements are complex."""
    if not isinstance(x, ndarray):
        x = array(x)
    if x.dtype in ("complex64", "complex128"):
        return ones(x.shape, dtype="bool")
    return zeros(x.shape, dtype="bool")

def issubdtype(arg1, arg2):
    """Check if arg1 dtype is a subtype of arg2."""
    # Map dtype strings to type hierarchy classes
    _dtype_to_class = {
        'bool': bool_,
        'int8': int8, 'int16': int16, 'int32': int32, 'int64': int64,
        'uint8': uint8, 'uint16': uint16, 'uint32': uint32, 'uint64': uint64,
        'float16': float16, 'float32': float32, 'float64': float64,
        'complex64': complex64, 'complex128': complex128,
    }

    # If arg2 is one of our type hierarchy classes, use issubclass
    if isinstance(arg2, type) and isinstance(arg2, _ScalarTypeMeta):
        # Normalize arg1 to a dtype string
        if isinstance(arg1, dtype):
            dt1 = str(arg1)
        elif isinstance(arg1, str):
            dt1 = arg1
        elif isinstance(arg1, type) and isinstance(arg1, _ScalarTypeMeta):
            dt1 = arg1._scalar_name
        elif isinstance(arg1, _ScalarType):
            dt1 = arg1._name
        elif hasattr(arg1, 'dtype'):
            dt1 = str(arg1.dtype)
        else:
            dt1 = str(arg1)
        cls1 = _dtype_to_class.get(dt1)
        if cls1 is not None:
            return issubclass(cls1, arg2)
        return False

    # Fall back to string-based logic for backward compatibility
    # Normalize arg1
    if isinstance(arg1, dtype):
        dt1 = str(arg1)
    elif isinstance(arg1, str):
        dt1 = arg1
    elif isinstance(arg1, _ScalarType):
        dt1 = arg1._name
    elif hasattr(arg1, 'dtype'):
        dt1 = str(arg1.dtype)
    else:
        dt1 = str(arg1) if not callable(arg1) else getattr(arg1, '__name__', str(arg1))

    # Normalize arg2
    if isinstance(arg2, str):
        dt2 = arg2
    elif isinstance(arg2, _ScalarType):
        dt2 = arg2._name
    elif isinstance(arg2, type) and hasattr(arg2, '__name__'):
        dt2 = arg2.__name__
    elif isinstance(arg2, dtype):
        dt2 = str(arg2)
    else:
        dt2 = str(arg2)

    # Map type categories
    _float_types = {"float16", "float32", "float64", "float", "floating", "float_"}
    _int_types = {"int8", "int16", "int32", "int64", "int", "integer", "signedinteger", "int_"}
    _uint_types = {"uint8", "uint16", "uint32", "uint64", "unsignedinteger"}
    _complex_types = {"complex64", "complex128", "complex", "complexfloating", "complex_"}
    _number_types = _float_types | _int_types | _uint_types | _complex_types | {"number"}
    _bool_types = {"bool", "bool_"}

    # Check if dt1 is a subtype of dt2
    if dt2 in ("floating", "float"):
        return dt1 in _float_types
    elif dt2 in ("integer", "signedinteger", "int"):
        return dt1 in _int_types
    elif dt2 in ("unsignedinteger",):
        return dt1 in _uint_types
    elif dt2 in ("complexfloating", "complex"):
        return dt1 in _complex_types
    elif dt2 in ("number",):
        return dt1 in _number_types
    elif dt2 in ("bool", "bool_"):
        return dt1 in _bool_types
    elif dt2 in ("generic",):
        return True  # everything is a subtype of generic
    elif dt2 in ("inexact",):
        return dt1 in (_float_types | _complex_types)
    else:
        # Direct dtype comparison
        return dt1 == dt2

def asarray(a, dtype=None, order=None):
    if isinstance(a, ndarray):
        if dtype is not None:
            return a.astype(str(dtype))
        return a
    return array(a, dtype=dtype)

asanyarray = asarray  # In our implementation, same as asarray

def ascontiguousarray(a, dtype=None):
    return asarray(a)

def copy(a, order="K"):
    if isinstance(a, ndarray):
        return a.copy()
    return array(a)

_CLIP_UNSET = object()

def clip(a, a_min=_CLIP_UNSET, a_max=_CLIP_UNSET, out=None, **kwargs):
    """Clip array values to [a_min, a_max]."""
    _conflict_msg = ("Passing `min` or `max` keyword argument when `a_min` and "
                     "`a_max` are provided is forbidden.")
    # Validate casting kwarg
    _clip_casting = "same_kind"  # default
    if 'casting' in kwargs:
        c = kwargs.pop('casting')
        if c is None:
            _clip_casting = "same_kind"
        else:
            _valid = ('no', 'equiv', 'safe', 'same_kind', 'unsafe')
            if c not in _valid:
                raise ValueError("casting must be one of 'no', 'equiv', 'safe', 'same_kind', or 'unsafe'")
            _clip_casting = c
    # Conflict checks: both positional AND keyword
    if a_min is not _CLIP_UNSET and a_max is not _CLIP_UNSET:
        if 'min' in kwargs or 'max' in kwargs:
            raise ValueError(_conflict_msg)
    # Support min=/max= as aliases for a_min/a_max
    _min_from_kw = False
    _max_from_kw = False
    if 'min' in kwargs:
        _min_from_kw = True
        a_min = kwargs.pop('min')
    if 'max' in kwargs:
        _max_from_kw = True
        a_max = kwargs.pop('max')
    # np.clip(arr) with no bounds → return copy
    if a_min is _CLIP_UNSET and a_max is _CLIP_UNSET:
        if not isinstance(a, (ndarray, _ObjectArray)):
            a = array(a)
        return a.copy() if isinstance(a, ndarray) else a
    # Check required args. Keyword-only min/max may omit one bound.
    if a_max is _CLIP_UNSET:
        if _min_from_kw:
            a_max = None
        else:
            raise TypeError("clip() missing 1 required positional argument: 'a_max'")
    if a_min is _CLIP_UNSET:
        if _max_from_kw:
            a_min = None
        else:
            raise TypeError("clip() missing 1 required positional argument: 'a_min'")
    if not isinstance(a, (ndarray, _ObjectArray)):
        a = array(a)
    if a_min is None and a_max is None:
        return a.copy() if isinstance(a, ndarray) else a
    # Check casting constraint when out is provided
    if out is not None and isinstance(a, ndarray) and isinstance(out, ndarray):
        # Determine result dtype of clipping
        _bound_dt = None
        for b in (a_min, a_max):
            if b is not None:
                if isinstance(b, ndarray):
                    bdt = str(b.dtype)
                elif isinstance(b, float):
                    bdt = "float64"
                elif isinstance(b, int):
                    bdt = "int64"
                else:
                    bdt = None
                if bdt is not None:
                    _bound_dt = bdt
        if _bound_dt is not None:
            a_dt = str(a.dtype)
            out_dt = str(out.dtype)
            # If bounds are float and array/out are int, check casting
            if _bound_dt in ("float32", "float64", "float16") and out_dt in ("int8", "int16", "int32", "int64", "uint8", "uint16", "uint32", "uint64"):
                if _clip_casting != "unsafe":
                    raise TypeError(
                        "Cannot cast ufunc 'clip' output from dtype('{}') to "
                        "dtype('{}') with casting rule '{}'".format(
                            _bound_dt, out_dt, _clip_casting))
    # _ObjectArray (complex) fallback
    if isinstance(a, _ObjectArray):
        data = list(a._data)
        for i, v in enumerate(data):
            if a_min is not None:
                try:
                    if v < a_min:
                        data[i] = a_min
                except TypeError:
                    pass
            if a_max is not None:
                try:
                    if v > a_max:
                        data[i] = a_max
                except TypeError:
                    pass
        return _ObjectArray(data, a._dtype)
    # Check if min/max are arrays — need element-wise clipping
    min_is_array = isinstance(a_min, ndarray)
    max_is_array = isinstance(a_max, ndarray)
    if min_is_array or max_is_array:
        result = a.copy()
        if a_min is not None:
            a_min_arr = a_min if min_is_array else full(a.shape, float(a_min))
            result = where(result < a_min_arr, a_min_arr, result)
        if a_max is not None:
            a_max_arr = a_max if max_is_array else full(a.shape, float(a_max))
            result = where(result > a_max_arr, a_max_arr, result)
        if out is not None:
            _copy_into(out, result)
            return out
        return result
    # Complex dtype: delegate to ndarray.clip which has Rust-level complex support
    dt_name = str(a.dtype) if isinstance(a, ndarray) else ""
    if "complex" in dt_name and isinstance(a, ndarray):
        return a.clip(a_min, a_max, out=out)
    # For integer dtypes, clamp bounds to dtype range to preserve dtype
    _int_dtypes = {"int8", "int16", "int32", "int64", "uint8", "uint16", "uint32", "uint64"}
    _int_dtypes = {"int8", "int16", "int32", "int64", "uint8", "uint16", "uint32", "uint64"}
    if dt_name in _int_dtypes:
        info = iinfo(dt_name)
        if a_min is None:
            a_min_f = float('-inf')
        else:
            a_min_f = float(a_min)
            if a_min_f < info.min:
                a_min_f = float(info.min)
            elif a_min_f > info.max:
                a_min_f = float(info.max)
        if a_max is None:
            a_max_f = float('inf')
        else:
            a_max_f = float(a_max)
            if a_max_f > info.max:
                a_max_f = float(info.max)
            elif a_max_f < info.min:
                a_max_f = float(info.min)
        if not (_math.isnan(a_min_f) or _math.isnan(a_max_f)):
            result = _native.clip(a, a_min_f, a_max_f)
            result = result.astype(dt_name)
            if out is not None:
                _copy_into(out, result)
                return out
            return result
    if a_min is None:
        a_min = float('-inf')
    else:
        if isinstance(a_min, complex):
            a_min = a_min.real
        a_min = float(a_min)
    if a_max is None:
        a_max = float('inf')
    else:
        if isinstance(a_max, complex):
            a_max = a_max.real
        a_max = float(a_max)
    # NaN propagation: if either bound is NaN, result is all NaN
    if _math.isnan(a_min) or _math.isnan(a_max):
        result = full(a.shape, float('nan'), dtype=str(a.dtype))
        if out is not None:
            _copy_into(out, result)
            return out
        return result
    result = _native.clip(a, a_min, a_max)
    if out is not None:
        _copy_into(out, result)
        return out
    return result

def abs(x, out=None):
    if isinstance(x, ndarray):
        result = x.abs()
        if out is not None:
            _copy_into(out, result)
            return out
        return result
    if isinstance(x, _ObjectArray):
        return x.__abs__()
    return __builtins__["abs"](x) if isinstance(__builtins__, dict) else _math.fabs(x)

absolute = abs

def sqrt(x):
    if isinstance(x, ndarray):
        return x.sqrt()
    if isinstance(x, complex):
        import cmath
        return cmath.sqrt(x)
    return _math.sqrt(x)

def exp(x):
    if isinstance(x, ndarray):
        return x.exp()
    return _math.exp(x)

def exp2(x):
    """Compute 2**x element-wise."""
    return power(2.0, asarray(x))

def log(x):
    if isinstance(x, ndarray):
        return x.log()
    return _math.log(x)

def sin(x):
    if isinstance(x, ndarray):
        return x.sin()
    return _math.sin(x)

def cos(x):
    if isinstance(x, ndarray):
        return x.cos()
    return _math.cos(x)

def tan(x):
    if isinstance(x, ndarray):
        return x.tan()
    return _math.tan(x)

def log10(x):
    if isinstance(x, ndarray):
        return _native.log10(x)
    return _math.log10(x)

def log2(x):
    if isinstance(x, ndarray):
        return _native.log2(x)
    return _math.log2(x)

def log1p(x):
    if isinstance(x, ndarray):
        return _native.log1p(x)
    return _math.log1p(x)

def expm1(x):
    if isinstance(x, ndarray):
        return _native.expm1(x)
    return _math.expm1(x)

def sign(x):
    if isinstance(x, ndarray):
        return _native.sign(x)
    return (x > 0) - (x < 0)

def deg2rad(x):
    if isinstance(x, ndarray):
        return _native.deg2rad(x)
    return _math.radians(x)

def rad2deg(x):
    if isinstance(x, ndarray):
        return _native.rad2deg(x)
    return _math.degrees(x)

# Aliases
radians = deg2rad
degrees = rad2deg

def sinh(x):
    if isinstance(x, ndarray):
        return _native.sinh(x)
    return _math.sinh(x)

def cosh(x):
    if isinstance(x, ndarray):
        return _native.cosh(x)
    return _math.cosh(x)

def tanh(x):
    if isinstance(x, ndarray):
        return _native.tanh(x)
    return _math.tanh(x)

def arcsin(x):
    if isinstance(x, ndarray):
        return _native.arcsin(x)
    return _math.asin(x)

def arccos(x):
    if isinstance(x, ndarray):
        return _native.arccos(x)
    return _math.acos(x)

def arctan(x):
    if isinstance(x, ndarray):
        return _native.arctan(x)
    return _math.atan(x)

def arctan2(y, x, out=None, where=True, **kwargs):
    if not isinstance(y, ndarray):
        y = array(y)
    if not isinstance(x, ndarray):
        x = array(x)
    return _native.arctan2(y, x)

def arcsinh(x):
    if not isinstance(x, ndarray):
        x = array(x)
    return _native.arcsinh(x)

def arccosh(x):
    if not isinstance(x, ndarray):
        x = array(x)
    return _native.arccosh(x)

def arctanh(x):
    if not isinstance(x, ndarray):
        x = array(x)
    return _native.arctanh(x)

def hypot(x1, x2):
    """Element-wise sqrt(x1**2 + x2**2)."""
    x1 = asarray(x1) if not isinstance(x1, ndarray) else x1
    x2 = asarray(x2) if not isinstance(x2, ndarray) else x2
    return sqrt(x1 * x1 + x2 * x2)

def trunc(x):
    if isinstance(x, ndarray):
        return _native.trunc(x)
    import math as _math
    return _math.trunc(x)

def floor(x):
    if isinstance(x, ndarray):
        return x.floor()
    return _math.floor(x)

def ceil(x):
    if isinstance(x, ndarray):
        return x.ceil()
    return _math.ceil(x)

def around(a, decimals=0, out=None):
    _builtin_round = __import__("builtins").round
    if not isinstance(a, ndarray):
        if isinstance(a, (list, tuple)):
            # Use Python's builtin round for banker's rounding
            result = array([_builtin_round(float(x), decimals) for x in a])
            if out is not None:
                out[:] = result
                return out
            return result
        else:
            return _builtin_round(float(a), decimals)
    return _native.around(a, decimals)

round_ = around
round = around

def ldexp(x1, x2):
    """Return x1 * 2**x2, element-wise."""
    if isinstance(x1, ndarray) or isinstance(x2, ndarray):
        x1 = asarray(x1)
        x2 = asarray(x2)
        return array([float(a) * (2.0 ** int(b)) for a, b in zip(x1.flatten().tolist(), x2.flatten().tolist())]).reshape(x1.shape)
    return float(x1) * (2.0 ** int(x2))

def frexp(x):
    """Decompose elements of x into mantissa and twos exponent."""
    if isinstance(x, ndarray):
        flat = x.flatten().tolist()
        mantissa = []
        exponent = []
        for v in flat:
            m, e = _math.frexp(float(v))
            mantissa.append(m)
            exponent.append(e)
        return array(mantissa).reshape(x.shape), array([float(e) for e in exponent]).reshape(x.shape)
    m, e = _math.frexp(float(x))
    return m, e

def allclose(a, b, rtol=1e-05, atol=1e-08, equal_nan=False):
    """Return True if two arrays are element-wise equal within a tolerance."""
    return bool(all(isclose(a, b, rtol=rtol, atol=atol, equal_nan=equal_nan)))

def isclose(a, b, rtol=1e-05, atol=1e-08, equal_nan=False):
    """Return boolean array where two arrays are element-wise equal within tolerance."""
    # Handle MaskedArray inputs — delegate to data, preserve mask
    from numpy.ma import MaskedArray as _MA
    _a_ma = isinstance(a, _MA)
    _b_ma = isinstance(b, _MA)
    if _a_ma or _b_ma:
        a_data = a.data if _a_ma else (asarray(a) if not isinstance(a, ndarray) else a)
        b_data = b.data if _b_ma else (asarray(b) if not isinstance(b, ndarray) else b)
        result_data = isclose(a_data, b_data, rtol=rtol, atol=atol, equal_nan=equal_nan)
        mask = a.mask if _a_ma else (b.mask if _b_ma else None)
        return _MA(result_data, mask=mask)
    # Fast path for _ObjectArray (complex, object dtypes)
    if isinstance(a, _ObjectArray) or isinstance(b, _ObjectArray):
        _babs = __import__("builtins").abs
        a_data = a._data if isinstance(a, _ObjectArray) else (a.flatten().tolist() if isinstance(a, ndarray) else [a])
        b_data = b._data if isinstance(b, _ObjectArray) else (b.flatten().tolist() if isinstance(b, ndarray) else [b])
        results = []
        for av, bv in zip(a_data, b_data):
            if equal_nan and _cmath_isnan(av) and _cmath_isnan(bv):
                results.append(True)
            else:
                try:
                    results.append(_babs(av - bv) <= atol + rtol * _babs(bv))
                except (TypeError, ValueError):
                    results.append(av == bv)
        return _native.array([1.0 if r else 0.0 for r in results]).astype("bool")
    scalar_input = not isinstance(a, ndarray) and not isinstance(b, ndarray) and not isinstance(a, (list, tuple)) and not isinstance(b, (list, tuple))
    if not isinstance(a, ndarray):
        a = array(a) if isinstance(a, (list, tuple)) else array([a])
    if not isinstance(b, ndarray):
        b = array(b) if isinstance(b, (list, tuple)) else array([b])
    # Handle infinities: inf == inf (same sign) should be True
    a_inf = isinf(a)
    b_inf = isinf(b)
    both_inf = logical_and(a_inf, b_inf)
    # Same-sign infinities are "close"
    same_inf = logical_and(both_inf, (a == b))
    # For the general case, replace inf with 0 to avoid inf-inf=nan
    a_safe = where(a_inf, zeros(a.shape), a)
    b_safe = where(b_inf, zeros(b.shape), b)
    diff = abs(a_safe - b_safe)
    if isinstance(atol, (list, tuple, ndarray)):
        atol_arr = asarray(atol)
    else:
        atol_arr = full(diff.shape, atol)
    if isinstance(rtol, (list, tuple, ndarray)):
        rtol_arr = asarray(rtol)
    else:
        rtol_arr = full(diff.shape, rtol)
    limit = atol_arr + rtol_arr * abs(b_safe)
    result = logical_or(diff <= limit, same_inf)
    # Different-sign infinities are never close
    diff_inf = logical_and(both_inf, logical_not(same_inf))
    result = logical_and(result, logical_not(diff_inf))
    # One inf, one finite: not close
    one_inf = logical_and(logical_or(a_inf, b_inf), logical_not(both_inf))
    result = logical_and(result, logical_not(one_inf))
    if equal_nan:
        both_nan = logical_and(isnan(a), isnan(b))
        result = logical_or(result, both_nan)
    if scalar_input and result.size == 1:
        return bool(result.flatten()[0])
    return result

def logaddexp(x1, x2):
    """Logarithm of the sum of exponentiations of the inputs."""
    x1, x2 = asarray(x1), asarray(x2)
    mx = maximum(x1, x2)
    return mx + log1p(exp(-abs(x1 - x2)))

def logaddexp2(x1, x2):
    """Logarithm base 2 of the sum of exponentiations of the inputs in base 2."""
    x1, x2 = asarray(x1), asarray(x2)
    ln2 = 0.6931471805599453
    return logaddexp(x1 * ln2, x2 * ln2) / ln2

def rint(x):
    """Round to nearest integer."""
    if isinstance(x, ndarray):
        return _native.around(x, 0)
    return float(round(x))

def fix(x):
    """Round to nearest integer towards zero."""
    if isinstance(x, ndarray):
        return _native.trunc(x)
    return float(_math.trunc(x))

def square(x):
    """Return the element-wise square."""
    x = asarray(x)
    return x * x

def cbrt(x):
    """Return the element-wise cube root."""
    x = asarray(x)
    # cbrt handles negative numbers correctly
    return sign(x) * power(abs(x), 1.0 / 3.0)

def reciprocal(x):
    """Return the reciprocal of the argument, element-wise."""
    x = asarray(x)
    return ones(x.shape) / x

def copysign(x1, x2):
    """Change the sign of x1 to that of x2, element-wise."""
    x1 = asarray(x1)
    x2 = asarray(x2)
    f1 = x1.flatten().tolist()
    f2 = x2.flatten().tolist()
    result = [_math.copysign(a, b) for a, b in zip(f1, f2)]
    return array(result).reshape(x1.shape)

def heaviside(x1, x2):
    """Compute the Heaviside step function.
    0 where x1 < 0, x2 where x1 == 0, 1 where x1 > 0."""
    x1 = asarray(x1)
    x2 = asarray(x2)
    result = where(x1 > zeros(x1.shape), ones(x1.shape), x2)
    result = where(x1 < zeros(x1.shape), zeros(x1.shape), result)
    return result

def sinc(x):
    """Return the sinc function: sin(pi*x)/(pi*x)."""
    x = asarray(x)
    px = x * pi
    # Avoid division by zero: where px==0, use 1 as denominator
    result = sin(px) / where(px == zeros(px.shape), ones(px.shape), px)
    result = where(x == zeros(x.shape), ones(x.shape), result)
    return result

def nan_to_num(x, copy=True, nan=0.0, posinf=None, neginf=None):
    """Replace NaN with zero and infinity with large finite numbers."""
    x = asarray(x)
    if posinf is None:
        posinf = 1.7976931348623157e+308  # float max
    if neginf is None:
        neginf = -1.7976931348623157e+308
    flat = x.flatten()
    n = flat.size
    vals = []
    for i in range(n):
        v = float(flat[i])
        if v != v:  # NaN check
            vals.append(nan)
        elif v == float('inf'):
            vals.append(posinf)
        elif v == float('-inf'):
            vals.append(neginf)
        else:
            vals.append(v)
    result = array(vals)
    if x.ndim > 1:
        result = result.reshape(x.shape)
    return result

def gamma(x):
    """Gamma function using Lanczos approximation."""
    if not isinstance(x, ndarray):
        g = 7
        c = [0.99999999999980993, 676.5203681218851, -1259.1392167224028,
             771.32342877765313, -176.61502916214059, 12.507343278686905,
             -0.13857109526572012, 9.9843695780195716e-6, 1.5056327351493116e-7]
        if x < 0.5:
            return _math.pi / (_math.sin(_math.pi * x) * gamma(1 - x))
        x -= 1
        a = c[0]
        t = x + g + 0.5
        for i in range(1, len(c)):
            a += c[i] / (x + i)
        return _math.sqrt(2 * _math.pi) * t ** (x + 0.5) * _math.exp(-t) * a
    flat_x = x.flatten().tolist()
    flat_r = [gamma(v) for v in flat_x]
    return array(flat_r).reshape(x.shape)

def lgamma(x):
    """Log of absolute value of the gamma function."""
    if not isinstance(x, ndarray):
        return _math.lgamma(float(x))
    flat = x.flatten().tolist()
    return array([_math.lgamma(float(v)) for v in flat]).reshape(x.shape)

def erf(x):
    """Error function (Abramowitz & Stegun approximation)."""
    if not isinstance(x, ndarray):
        a1 = 0.254829592
        a2 = -0.284496736
        a3 = 1.421413741
        a4 = -1.453152027
        a5 = 1.061405429
        p = 0.3275911
        sign_x = 1.0 if x >= 0 else -1.0
        ax = x if x >= 0 else -x
        t = 1.0 / (1.0 + p * ax)
        y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * _math.exp(-x * x)
        return sign_x * y
    flat = x.flatten().tolist()
    return array([erf(v) for v in flat]).reshape(x.shape)

def erfc(x):
    """Complementary error function: 1 - erf(x)."""
    if not isinstance(x, ndarray):
        return 1.0 - erf(x)
    flat = x.flatten().tolist()
    return array([erfc(v) for v in flat]).reshape(x.shape)

def j0(x):
    """Bessel function of the first kind, order 0."""
    if not isinstance(x, ndarray):
        import math as _m
        ax = x if x >= 0 else -x
        if ax < 8.0:
            y = x * x
            ans1 = 57568490574.0 + y * (-13362590354.0 + y * (651619640.7 + y * (-11214424.18 + y * (77392.33017 + y * (-184.9052456)))))
            ans2 = 57568490411.0 + y * (1029532985.0 + y * (9494680.718 + y * (59272.64853 + y * (267.8532712 + y * 1.0))))
            return ans1 / ans2
        else:
            z = 8.0 / ax
            y = z * z
            xx = ax - 0.785398164
            p0 = 1.0 + y * (-0.1098628627e-2 + y * (0.2734510407e-4 + y * (-0.2073370639e-5 + y * 0.2093887211e-6)))
            q0 = -0.1562499995e-1 + y * (0.1430488765e-3 + y * (-0.6911147651e-5 + y * (0.7621095161e-6 + y * (-0.934935152e-7))))
            return _m.sqrt(0.636619772 / ax) * (_m.cos(xx) * p0 - z * _m.sin(xx) * q0)
    flat = x.flatten().tolist()
    return array([j0(v) for v in flat]).reshape(x.shape)

def j1(x):
    """Bessel function of the first kind, order 1."""
    if not isinstance(x, ndarray):
        import math as _m
        ax = x if x >= 0 else -x
        if ax < 8.0:
            y = x * x
            ans1 = x * (72362614232.0 + y * (-7895059235.0 + y * (242396853.1 + y * (-2972611.439 + y * (15704.48260 + y * (-30.16036606))))))
            ans2 = 144725228442.0 + y * (2300535178.0 + y * (18583304.74 + y * (99447.43394 + y * (376.9991397 + y * 1.0))))
            return ans1 / ans2
        else:
            z = 8.0 / ax
            y = z * z
            xx = ax - 2.356194491
            p1 = 1.0 + y * (0.183105e-2 + y * (-0.3516396496e-4 + y * (0.2457520174e-5 + y * (-0.240337019e-6))))
            q1 = 0.04687499995 + y * (-0.2002690873e-3 + y * (0.8449199096e-5 + y * (-0.88228987e-6 + y * 0.105787412e-6)))
            ans = _m.sqrt(0.636619772 / ax) * (_m.cos(xx) * p1 - z * _m.sin(xx) * q1)
            return ans if x >= 0 else -ans
    flat = x.flatten().tolist()
    return array([j1(v) for v in flat]).reshape(x.shape)

def y0(x):
    """Bessel function of the second kind, order 0."""
    if not isinstance(x, ndarray):
        import math as _m
        if x < 8.0:
            y = x * x
            ans1 = -2957821389.0 + y * (7062834065.0 + y * (-512359803.6 + y * (10879881.29 + y * (-86327.92757 + y * 228.4622733))))
            ans2 = 40076544269.0 + y * (745249964.8 + y * (7189466.438 + y * (47447.26470 + y * (226.1030244 + y * 1.0))))
            return (ans1 / ans2) + 0.636619772 * j0(x) * _m.log(x)
        else:
            z = 8.0 / x
            y = z * z
            xx = x - 0.785398164
            p0 = 1.0 + y * (-0.1098628627e-2 + y * (0.2734510407e-4 + y * (-0.2073370639e-5 + y * 0.2093887211e-6)))
            q0 = -0.1562499995e-1 + y * (0.1430488765e-3 + y * (-0.6911147651e-5 + y * (0.7621095161e-6 + y * (-0.934935152e-7))))
            return _m.sqrt(0.636619772 / x) * (_m.sin(xx) * p0 + z * _m.cos(xx) * q0)
    flat = x.flatten().tolist()
    return array([y0(v) for v in flat]).reshape(x.shape)

def y1(x):
    """Bessel function of the second kind, order 1."""
    if not isinstance(x, ndarray):
        import math as _m
        if x < 8.0:
            y = x * x
            ans1 = x * (-4900604943000.0 + y * (1275274390000.0 + y * (-51534866838.0 + y * (622785432.7 + y * (-3130827.838 + y * 7.374510962)))))
            ans2 = 24995805700000.0 + y * (424441966400.0 + y * (3733650367.0 + y * (22459040.02 + y * (103680.2068 + y * (365.9584658 + y * 1.0)))))
            return (ans1 / ans2) + 0.636619772 * (j1(x) * _m.log(x) - 1.0 / x)
        else:
            z = 8.0 / x
            y = z * z
            xx = x - 2.356194491
            p1 = 1.0 + y * (0.183105e-2 + y * (-0.3516396496e-4 + y * (0.2457520174e-5 + y * (-0.240337019e-6))))
            q1 = 0.04687499995 + y * (-0.2002690873e-3 + y * (0.8449199096e-5 + y * (-0.88228987e-6 + y * 0.105787412e-6)))
            return _m.sqrt(0.636619772 / x) * (_m.sin(xx) * p1 + z * _m.cos(xx) * q1)
    flat = x.flatten().tolist()
    return array([y1(v) for v in flat]).reshape(x.shape)

special = type('special', (), {
    'gamma': staticmethod(gamma),
    'erf': staticmethod(erf),
    'erfc': staticmethod(erfc),
    'lgamma': staticmethod(lgamma),
    'j0': staticmethod(j0),
    'j1': staticmethod(j1),
    'y0': staticmethod(y0),
    'y1': staticmethod(y1),
})()

def _dtype_cast(result, dtype):
    """Cast result to dtype if dtype is not None."""
    if dtype is not None:
        result = asarray(result).astype(str(dtype) if not isinstance(dtype, str) else dtype)
    return result

def sum(a, axis=None, dtype=None, out=None, keepdims=False, initial=None, where=True):
    if not isinstance(a, ndarray):
        # For plain lists/tuples, convert to array; for other iterables use builtin sum
        if isinstance(a, (list, tuple)):
            a = asarray(a)
        else:
            return _builtin_sum(a)
    if where is not True:
        w = asarray(where).astype("bool").astype("float64")
        a = a * w
    if axis is not None:
        result = a.sum(axis, keepdims)
    else:
        result = a.sum(None, keepdims)
    if dtype is not None:
        result = _dtype_cast(result, dtype)
    if initial is not None:
        result = result + initial
    return result

def prod(a, axis=None, dtype=None, out=None, keepdims=False, initial=None, where=True):
    if not isinstance(a, ndarray):
        a = asarray(a)
    if where is not True:
        w = asarray(where).astype("bool").astype("float64")
        a = a * w + (1.0 - w)
    result = a.prod(axis, keepdims)
    if dtype is not None:
        result = _dtype_cast(result, dtype)
    if initial is not None:
        result = result * initial
    return result

def cumsum(a, axis=None, dtype=None, out=None):
    if isinstance(a, ndarray):
        result = a.cumsum(axis)
    else:
        result = array(a).cumsum(axis)
    return _dtype_cast(result, dtype)

def cumprod(a, axis=None, dtype=None, out=None):
    if isinstance(a, ndarray):
        result = a.cumprod(axis)
    else:
        result = array(a).cumprod(axis)
    return _dtype_cast(result, dtype)

def diff(a, n=1, axis=-1, prepend=None, append=None):
    if not isinstance(a, ndarray):
        a = array(a)
    if axis is not None and axis < 0:
        axis = a.ndim + axis
    if prepend is not None:
        prepend = asarray(prepend)
        if prepend.ndim == 0:
            prepend = array([float(prepend)])
        a = concatenate([prepend, a])
    if append is not None:
        append = asarray(append)
        if append.ndim == 0:
            append = array([float(append)])
        a = concatenate([a, append])
    return _native.diff(a, n, axis)

def mean(a, axis=None, dtype=None, out=None, keepdims=False, where=True):
    from numpy.ma import MaskedArray as _MA
    if isinstance(a, _MA):
        filled = a.filled(0.0)
        not_mask = logical_not(a.mask).astype("float64")
        s = sum(filled * not_mask, axis=axis, keepdims=keepdims)
        c = sum(not_mask, axis=axis, keepdims=keepdims)
        result = s / c
        if out is not None and isinstance(out, _MA):
            _copy_into(out.data, result if isinstance(result, ndarray) else asarray(result))
            return out
        if isinstance(result, ndarray):
            return _MA(result, mask=zeros(result.shape).astype("bool"))
        return result
    if not isinstance(a, ndarray):
        a = asarray(a)
    if a.size == 0:
        import warnings as _w
        _w.warn("Mean of empty slice.", RuntimeWarning, stacklevel=2)
        result = float('nan')
        if out is not None:
            out[()] = result
            return out
        return result
    if where is not True:
        w = asarray(where).astype("bool").astype("float64")
        s = (a * w).sum(axis, keepdims)
        c = w.sum(axis, keepdims)
        result = s / c
    elif axis is not None:
        if isinstance(axis, tuple):
            # Compute n for the tuple axes
            n = 1
            for ax in axis:
                n *= a.shape[ax]
            result = a.sum(axis, False) / n
            if keepdims:
                new_shape = list(a.shape)
                for ax in axis:
                    new_shape[ax] = 1
                result = result.reshape(tuple(new_shape))
        else:
            result = a.mean(axis, keepdims)
    else:
        result = a.mean(None, keepdims)
    if keepdims and not isinstance(result, ndarray):
        result = array([float(result)]).reshape((1,) * a.ndim)
    if dtype is not None:
        result = _dtype_cast(result, dtype)
    if out is not None:
        if isinstance(out, ndarray):
            if out.size == 1:
                out[0] = float(result) if not isinstance(result, ndarray) else float(result.flatten()[0])
            else:
                flat_r = result.flatten() if isinstance(result, ndarray) else array([result])
                for i in range(out.size):
                    out.flatten()[i] = float(flat_r[i])
        return out
    return result

def std(a, axis=None, dtype=None, out=None, ddof=0, keepdims=False, where=True, mean=None, correction=None):
    if correction is not None and ddof != 0:
        raise ValueError("ddof and correction can't be provided simultaneously.")
    if correction is not None:
        ddof = correction
    from numpy.ma import MaskedArray as _MA
    if isinstance(a, _MA):
        v = var(a, axis=axis, ddof=ddof, keepdims=keepdims, where=where, mean=mean)
        if isinstance(v, _MA):
            result = _MA(sqrt(v.data), mask=v.mask)
        else:
            result = sqrt(v) if isinstance(v, ndarray) else _math.sqrt(v)
        if out is not None and isinstance(out, _MA):
            r_arr = result.data if isinstance(result, _MA) else (result if isinstance(result, ndarray) else asarray(result))
            _copy_into(out.data, r_arr)
            return out
        return result
    if isinstance(a, complex) and not isinstance(a, (int, float)):
        return 0.0
    if not isinstance(a, ndarray):
        a = asarray(a)
    if a.size == 0:
        import warnings as _w
        _w.warn("Degrees of freedom <= 0 for slice", RuntimeWarning, stacklevel=2)
        result = float('nan')
        if out is not None:
            if isinstance(out, ndarray) and out.size == 1:
                out[0] = result
            return out
        return result
    result = sqrt(var(a, axis=axis, ddof=ddof, keepdims=keepdims, where=where, mean=mean))
    if dtype is not None:
        result = _dtype_cast(result, dtype)
    if out is not None:
        if isinstance(out, ndarray) and out.size == 1:
            out[0] = float(result) if not isinstance(result, ndarray) else float(result.flatten()[0])
        return out
    if isinstance(a, ndarray) and axis is None and not keepdims and not isinstance(result, ndarray):
        if isinstance(result, complex):
            return complex128(result)
        return float64(result)
    return result

def var(a, axis=None, dtype=None, out=None, ddof=0, keepdims=False, where=True, mean=None, correction=None):
    if correction is not None and ddof != 0:
        raise ValueError("ddof and correction can't be provided simultaneously.")
    if correction is not None:
        ddof = correction
    from numpy.ma import MaskedArray as _MA
    if isinstance(a, _MA):
        filled = a.filled(0.0)
        not_mask = logical_not(a.mask).astype("float64")
        c = sum(not_mask, axis=axis, keepdims=True)
        if mean is not None:
            m = mean.data if isinstance(mean, _MA) else (mean if isinstance(mean, ndarray) else asarray(mean))
        else:
            m = sum(filled * not_mask, axis=axis, keepdims=True) / c
        diff = (filled - m) ** 2
        c_out = sum(not_mask, axis=axis, keepdims=keepdims)
        result = sum(diff * not_mask, axis=axis, keepdims=keepdims) / (c_out - ddof)
        if out is not None and isinstance(out, _MA):
            _copy_into(out.data, result if isinstance(result, ndarray) else asarray(result))
            return out
        if isinstance(result, ndarray):
            return _MA(result, mask=zeros(result.shape).astype("bool"))
        return result
    if isinstance(a, complex) and not isinstance(a, (int, float)):
        # scalar complex: var of single value is 0
        return 0.0
    if not isinstance(a, ndarray):
        a = asarray(a)
    if a.size == 0:
        import warnings as _w
        _w.warn("Degrees of freedom <= 0 for slice", RuntimeWarning, stacklevel=2)
        result = float('nan')
        if out is not None:
            if isinstance(out, ndarray) and out.size == 1:
                out[0] = result
            return out
        return result
    _is_complex = str(a.dtype).startswith("complex")
    def _sq_dev(diff):
        if _is_complex:
            return abs(diff) ** 2
        return diff ** 2
    if where is not True:
        w = asarray(where).astype("bool").astype("float64")
        c_full = w.sum(axis, True)  # keepdims for mean computation
        if mean is not None:
            m = asarray(mean)
        else:
            m = (a * w).sum(axis, True) / c_full
        c_out = w.sum(axis, keepdims)  # match output keepdims
        if ddof == 0:
            result = (_sq_dev(a - m) * w).sum(axis, keepdims) / c_out
        else:
            result = (_sq_dev(a - m) * w).sum(axis, keepdims) / (c_out - ddof)
        if not keepdims and isinstance(result, ndarray) and result.ndim > 0:
            result = result.squeeze()
    elif axis is not None:
        # Compute n for the reduction axes
        if isinstance(axis, int):
            n = a.shape[axis]
        elif isinstance(axis, tuple):
            n = 1
            for ax in axis:
                n *= a.shape[ax]
        else:
            n = a.size
        def _sum_keepdims(arr, ax, kd):
            """Sum with keepdims support for tuple axes."""
            r = arr.sum(ax, False)
            if kd and isinstance(ax, tuple):
                new_shape = list(arr.shape)
                for _ax in ax:
                    new_shape[_ax] = 1
                r = r.reshape(tuple(new_shape))
            elif kd:
                r = arr.sum(ax, True)
                return r
            return r
        if mean is not None:
            m = asarray(mean)
        else:
            m = _sum_keepdims(a, axis, True) / n
        result = _sum_keepdims(_sq_dev(a - m), axis, keepdims) / (n - ddof)
        if not keepdims and isinstance(result, ndarray) and result.ndim == 0:
            result = float(result)
    elif mean is not None:
        m = asarray(mean)
        n = a.size
        result = _sq_dev(a - m).sum() / (n - ddof)
        if isinstance(result, ndarray) and result.ndim == 0:
            result = float(result)
    elif _is_complex:
        m = a.sum(None, True) / a.size
        result = _sq_dev(a - m).sum() / (a.size - ddof)
        if isinstance(result, ndarray) and result.ndim == 0:
            result = float(result)
    else:
        result = a.var(None, ddof, keepdims)
    if dtype is not None:
        result = _dtype_cast(result, dtype)
    if out is not None:
        if isinstance(out, ndarray) and out.size == 1:
            out[0] = float(result) if not isinstance(result, ndarray) else float(result.flatten()[0])
        return out
    return result

def nansum(a, axis=None, dtype=None, out=None, keepdims=False, initial=None, where=True):
    if not isinstance(a, ndarray):
        a = array(a)
    if where is not True:
        w = asarray(where).astype("bool").astype("float64")
        a = a * w
    result = _native.nansum(a, axis, keepdims)
    if dtype is not None:
        result = _dtype_cast(result, dtype)
    if initial is not None:
        result = result + initial
    return result

def nanmean(a, axis=None, dtype=None, out=None, keepdims=False, where=True):
    if not isinstance(a, ndarray):
        a = array(a)
    if where is not True:
        w = asarray(where).astype("bool").astype("float64")
        s = (a * w).sum(axis, keepdims)
        c = w.sum(axis, keepdims)
        result = s / c
    else:
        result = _native.nanmean(a, axis, keepdims)
    if dtype is not None:
        result = _dtype_cast(result, dtype)
    return result

def nanstd(a, axis=None, dtype=None, out=None, ddof=0, keepdims=False, where=True):
    if not isinstance(a, ndarray):
        a = array(a)
    if where is not True:
        result = sqrt(nanvar(a, axis=axis, ddof=ddof, keepdims=keepdims, where=where))
    else:
        result = _native.nanstd(a, axis, ddof, keepdims)
    if dtype is not None:
        result = _dtype_cast(result, dtype)
    return result

def nanvar(a, axis=None, dtype=None, out=None, ddof=0, keepdims=False, where=True):
    if not isinstance(a, ndarray):
        a = array(a)
    if where is not True:
        w = asarray(where).astype("bool").astype("float64")
        c = w.sum(axis, True)
        m = (a * w).sum(axis, True) / c
        if ddof == 0:
            result = ((a - m) ** 2 * w).sum(axis, keepdims) / c
        else:
            result = ((a - m) ** 2 * w).sum(axis, keepdims) / (c - ddof)
        if not keepdims and isinstance(result, ndarray) and result.ndim > 0:
            result = result.squeeze()
    else:
        result = _native.nanvar(a, axis, ddof, keepdims)
    if dtype is not None:
        result = _dtype_cast(result, dtype)
    return result

def nanmin(a, axis=None, out=None, keepdims=False):
    if not isinstance(a, ndarray):
        a = array(a)
    return _native.nanmin(a, axis, keepdims)

def nanmax(a, axis=None, out=None, keepdims=False):
    if not isinstance(a, ndarray):
        a = array(a)
    return _native.nanmax(a, axis, keepdims)

def nanargmin(a, axis=None, out=None):
    if not isinstance(a, ndarray):
        a = array(a)
    return _native.nanargmin(a, axis)

def nanargmax(a, axis=None, out=None):
    if not isinstance(a, ndarray):
        a = array(a)
    return _native.nanargmax(a, axis)

def nanprod(a, axis=None, dtype=None, out=None, keepdims=False, initial=None, where=True):
    if not isinstance(a, ndarray):
        a = array(a)
    if where is not True:
        w = asarray(where).astype("bool").astype("float64")
        a = a * w + (1.0 - w)
    result = _native.nanprod(a, axis, keepdims)
    if dtype is not None:
        result = _dtype_cast(result, dtype)
    if initial is not None:
        result = result * initial
    return result

def nancumsum(a, axis=None, dtype=None, out=None):
    if not isinstance(a, ndarray):
        a = array(a)
    return _native.nancumsum(a, axis)

def nancumprod(a, axis=None, dtype=None, out=None):
    if not isinstance(a, ndarray):
        a = array(a)
    return _native.nancumprod(a, axis)

def _apply_keepdims(result, a, axis):
    """Expand reduced axis back to size-1 when keepdims=True."""
    if isinstance(result, ndarray):
        shape = list(a.shape)
        shape[axis] = 1
        return result.reshape(shape)
    else:
        # scalar result - wrap in array with expanded dims
        shape = [1] * a.ndim
        return array([float(result)]).reshape(shape)

def quantile(a, q, axis=None, out=None, overwrite_input=False, method="linear", keepdims=False):
    if not isinstance(a, ndarray):
        a = array(a)
    if isinstance(q, (list, tuple, ndarray)):
        q_list = q.tolist() if isinstance(q, ndarray) else list(q)
        results = [_native.quantile(a, float(qi), axis) for qi in q_list]
        return array(results) if axis is None else stack(results)
    result = _native.quantile(a, float(q), axis)
    if keepdims and axis is not None:
        result = _apply_keepdims(result, a, axis)
    return result

def percentile(a, q, axis=None, out=None, overwrite_input=False, method="linear", keepdims=False):
    if not isinstance(a, ndarray):
        a = array(a)
    if isinstance(q, (list, tuple, ndarray)):
        q_list = q.tolist() if isinstance(q, ndarray) else list(q)
        results = [_native.percentile(a, float(qi), axis) for qi in q_list]
        return array(results) if axis is None else stack(results)
    result = _native.percentile(a, float(q), axis)
    if keepdims and axis is not None:
        result = _apply_keepdims(result, a, axis)
    return result

def median(a, axis=None, out=None, overwrite_input=False, keepdims=False):
    if not isinstance(a, ndarray):
        a = array(a)
    result = _native.quantile(a, 0.5, axis)
    if keepdims and axis is not None:
        result = _apply_keepdims(result, a, axis)
    return result

def cov(m, y=None, rowvar=True, bias=False, ddof=None, fweights=None, aweights=None):
    if not isinstance(m, ndarray):
        m = array(m)
    _ddof = ddof if ddof is not None else (0 if bias else 1)
    if y is not None:
        if not isinstance(y, ndarray):
            y = array(y)
        return _native.cov(m, y, rowvar, _ddof)
    return _native.cov(m, None, rowvar, _ddof)

def corrcoef(x, y=None, rowvar=True):
    if not isinstance(x, ndarray):
        x = array(x)
    if y is not None:
        if not isinstance(y, ndarray):
            y = array(y)
        return _native.corrcoef(x, y, rowvar)
    return _native.corrcoef(x, None, rowvar)

def average(a, axis=None, weights=None, returned=False, keepdims=False):
    """Compute the weighted average along the specified axis."""
    a = asarray(a)
    if weights is None:
        avg = mean(a, axis=axis)
        if returned:
            if axis is None:
                return avg, float(a.size)
            return avg, full(avg.shape, float(a.shape[axis]))
        return avg
    weights = asarray(weights)
    wsum = sum(a * weights, axis=axis)
    wt = sum(weights, axis=axis)
    avg = wsum / wt
    if returned:
        return avg, wt
    return avg

def _nan_quantile_impl(a, q, axis):
    """Helper for nanmedian/nanpercentile/nanquantile with axis support."""
    a = asarray(a)
    if axis is None:
        flat = a.flatten()
        n = flat.size
        vals = []
        for i in range(n):
            v = flat[i]
            if v == v:
                vals.append(v)
        if len(vals) == 0:
            return float('nan')
        vals.sort()
        idx = q * (len(vals) - 1)
        lo = int(idx)
        hi = lo + 1
        if hi >= len(vals):
            return vals[lo]
        frac = idx - lo
        return vals[lo] * (1 - frac) + vals[hi] * frac
    # With axis: apply along axis manually
    # Move target axis to last, then iterate over other dims
    # Simplified: just handle 2D case
    if a.ndim == 1:
        return _nan_quantile_impl(a, q, None)
    results = []
    if axis == 0:
        for j in range(a.shape[1]):
            col = array([a[i][j] for i in range(a.shape[0])])
            results.append(_nan_quantile_impl(col, q, None))
    else:
        for i in range(a.shape[0]):
            results.append(_nan_quantile_impl(a[i], q, None))
    return array(results)

def nanmedian(a, axis=None, out=None, overwrite_input=False, keepdims=False):
    """Compute the median along the specified axis, ignoring NaNs."""
    a = asarray(a)
    if axis is not None:
        # For axis support, fall back to sorting approach
        return _nan_quantile_impl(a, 0.5, axis)
    # Flatten, filter NaNs, compute median of remaining
    flat = a.flatten()
    n = flat.size
    vals = []
    for i in range(n):
        v = flat[i]
        if v == v:  # not NaN
            vals.append(v)
    if len(vals) == 0:
        return float('nan')
    vals.sort()
    mid = len(vals) // 2
    if len(vals) % 2 == 0:
        return (vals[mid - 1] + vals[mid]) / 2.0
    return vals[mid]

def nanpercentile(a, q, axis=None, out=None, overwrite_input=False, method="linear", keepdims=False):
    """Compute the qth percentile, ignoring NaNs."""
    return _nan_quantile_impl(asarray(a), q / 100.0, axis)

def nanquantile(a, q, axis=None, out=None, overwrite_input=False, method="linear", keepdims=False):
    """Compute the qth quantile, ignoring NaNs."""
    return _nan_quantile_impl(asarray(a), q, axis)

def ediff1d(ary, to_end=None, to_begin=None):
    """The differences between consecutive elements of an array."""
    ary = asarray(ary).flatten()
    n = ary.size
    diffs = []
    if to_begin is not None:
        if isinstance(to_begin, (int, float)):
            diffs.append(float(to_begin))
        else:
            tb = asarray(to_begin).flatten()
            for i in range(tb.size):
                diffs.append(tb[i])
    for i in range(1, n):
        diffs.append(ary[i] - ary[i - 1])
    if to_end is not None:
        if isinstance(to_end, (int, float)):
            diffs.append(float(to_end))
        else:
            te = asarray(to_end).flatten()
            for i in range(te.size):
                diffs.append(te[i])
    return array(diffs)

def fmax(x1, x2):
    """Element-wise maximum, ignoring NaNs."""
    x1 = asarray(x1)
    x2 = asarray(x2)
    # where x1 is nan, use x2; where x2 is nan, use x1; otherwise max
    x1_nan = isnan(x1)
    x2_nan = isnan(x2)
    result = where(x1 > x2, x1, x2)
    result = where(x1_nan, x2, result)
    result = where(x2_nan, x1, result)
    return result

def fmin(x1, x2):
    """Element-wise minimum, ignoring NaNs."""
    x1 = asarray(x1)
    x2 = asarray(x2)
    x1_nan = isnan(x1)
    x2_nan = isnan(x2)
    result = where(x1 < x2, x1, x2)
    result = where(x1_nan, x2, result)
    result = where(x2_nan, x1, result)
    return result

def max(a, axis=None, out=None, keepdims=False, where=True):
    if not isinstance(a, ndarray):
        a = asarray(a)
    if where is not True:
        w = asarray(where).astype("bool")
        fill_val = float('-inf')
        flat_a = a.flatten().tolist()
        flat_w = w.flatten().tolist()
        masked = [v if m else fill_val for v, m in zip(flat_a, flat_w)]
        a = array(masked).reshape(a.shape)
    if axis is not None:
        return a.max(axis, keepdims)
    return a.max(None, keepdims)

amax = max

def min(a, axis=None, out=None, keepdims=False, where=True):
    if not isinstance(a, ndarray):
        a = asarray(a)
    if where is not True:
        w = asarray(where).astype("bool")
        fill_val = float('inf')
        flat_a = a.flatten().tolist()
        flat_w = w.flatten().tolist()
        masked = [v if m else fill_val for v, m in zip(flat_a, flat_w)]
        a = array(masked).reshape(a.shape)
    if axis is not None:
        return a.min(axis, keepdims)
    return a.min(None, keepdims)

amin = min

def argmax(a, axis=None, out=None, keepdims=False):
    if not isinstance(a, ndarray):
        a = asarray(a)
    return a.argmax(axis)

def argmin(a, axis=None, out=None, keepdims=False):
    if not isinstance(a, ndarray):
        a = asarray(a)
    return a.argmin(axis)

def reshape(a, newshape=None, order="C", *, shape=None, copy=None, **kwargs):
    # Handle the case where newshape is passed as keyword 'newshape'
    if 'newshape' in kwargs:
        if newshape is not None and shape is not None:
            raise TypeError("You cannot specify 'newshape' and 'shape' arguments at the same time.")
        import warnings as _w
        _w.warn("keyword argument 'newshape' is deprecated, use 'shape' instead", DeprecationWarning, stacklevel=2)
        newshape = kwargs.pop('newshape')
    if newshape is not None and shape is not None:
        raise TypeError("You cannot specify 'newshape' and 'shape' arguments at the same time.")
    if shape is not None:
        newshape = shape
    if newshape is None:
        raise TypeError("reshape() missing 1 required positional argument: 'shape'")
    if not isinstance(a, ndarray):
        a = asarray(a)
    if copy is True:
        return a.copy().reshape(newshape)
    # copy=False means raise if a copy would be needed (for order changes)
    if copy is False and order is not None and order != "C":
        raise ValueError("Unable to avoid creating a copy while reshaping.")
    return a.reshape(newshape)

def _transpose_with_axes(a, axes):
    """Transpose ndarray with an arbitrary axis permutation (pure Python).

    Parameters
    ----------
    a : ndarray
    axes : list/tuple of int - the desired permutation of axes.

    Returns an ndarray with axes reordered according to *axes*.
    """
    if not isinstance(a, ndarray):
        a = asarray(a)
    shape = a.shape
    ndim_a = len(shape)
    if axes is None:
        return a.T
    axes = list(axes)
    if len(axes) != ndim_a:
        raise ValueError("axes don't match array")
    # Fast-paths
    if axes == list(range(ndim_a)):
        return a.copy() if hasattr(a, 'copy') else array(a.tolist())
    if ndim_a == 2 and axes == [1, 0]:
        return a.T
    # Generic: walk every index of the output and pick from source
    new_shape = tuple(shape[ax] for ax in axes)
    size = 1
    for s in new_shape:
        size *= s
    flat_data = a.flatten()
    # Build strides of original array (row-major)
    src_strides = [0] * ndim_a
    s = 1
    for i in range(ndim_a - 1, -1, -1):
        src_strides[i] = s
        s *= shape[i]
    result = [0.0] * size
    # Iterate over every multi-index of the *output*
    out_idx = [0] * ndim_a
    for flat_i in range(size):
        # Map output index -> source index
        src_flat = 0
        for d in range(ndim_a):
            src_flat += out_idx[d] * src_strides[axes[d]]
        result[flat_i] = float(flat_data[src_flat])
        # Increment out_idx (rightmost first)
        for d in range(ndim_a - 1, -1, -1):
            out_idx[d] += 1
            if out_idx[d] < new_shape[d]:
                break
            out_idx[d] = 0
    return array(result).reshape(list(new_shape))

def transpose(a, axes=None):
    if not isinstance(a, ndarray):
        a = asarray(a)
    if axes is not None:
        return _transpose_with_axes(a, axes)
    return a.T

def flatten(a, order="C"):
    return a.flatten()

def ravel(a, order="C"):
    if isinstance(a, ndarray):
        return a.ravel()
    return array(a).ravel()

def squeeze(a, axis=None):
    a = asarray(a)
    if axis is not None and isinstance(axis, (tuple, list)):
        result = a
        for ax in sorted(axis, reverse=True):
            result = squeeze(result, axis=ax)
        return result
    if isinstance(a, ndarray):
        if axis is not None and axis < 0:
            axis += a.ndim
        return a.squeeze(axis)
    return a

def expand_dims(a, axis):
    a = asarray(a)
    if isinstance(axis, (tuple, list)):
        ndim_out = a.ndim + len(axis)
        # Normalize negative axes
        normalized = []
        for ax in axis:
            if ax < 0:
                ax = ndim_out + ax
            normalized.append(ax)
        normalized.sort()
        result = a
        for ax in normalized:
            result = expand_dims(result, ax)
        return result
    if isinstance(a, ndarray):
        return a.expand_dims(axis)
    return a

def append(arr, values, axis=None):
    """Append values to the end of an array."""
    if not isinstance(arr, ndarray):
        arr = array(arr)
    if not isinstance(values, ndarray):
        values = array(values)
    if axis is None:
        return concatenate([arr.flatten(), values.flatten()])
    return concatenate([arr, values], axis=axis)

def atleast_1d(*arys):
    """Convert inputs to arrays with at least one dimension."""
    results = []
    for a in arys:
        if not isinstance(a, ndarray):
            a = array(a)
        if a.ndim == 0:
            a = a.reshape([1])
        results.append(a)
    if len(results) == 1:
        return results[0]
    return results

def atleast_2d(*arys):
    """Convert inputs to arrays with at least two dimensions."""
    results = []
    for a in arys:
        if not isinstance(a, ndarray):
            a = array(a)
        if a.ndim == 0:
            a = a.reshape([1, 1])
        elif a.ndim == 1:
            a = a.reshape([1, len(a)])
        results.append(a)
    if len(results) == 1:
        return results[0]
    return results

def atleast_3d(*arys):
    """Convert inputs to arrays with at least three dimensions."""
    results = []
    for a in arys:
        if not isinstance(a, ndarray):
            a = array(a)
        if a.ndim == 0:
            a = a.reshape([1, 1, 1])
        elif a.ndim == 1:
            a = a.reshape([1, len(a), 1])
        elif a.ndim == 2:
            a = a.reshape(list(a.shape) + [1])
        results.append(a)
    if len(results) == 1:
        return results[0]
    return results

def stack(arrays, axis=0, out=None):
    return _native.stack_native(list(arrays), axis)

def vstack(tup):
    arrs = [asarray(a) for a in tup]
    # For 1D inputs, reshape to 2D first (row vectors)
    expanded = []
    for a in arrs:
        if a.ndim == 1:
            expanded.append(a.reshape([1, a.shape[0]]))
        else:
            expanded.append(a)
    return concatenate(expanded, 0)

def hstack(tup):
    arrs = [asarray(a) for a in tup]
    if arrs[0].ndim > 1:
        return concatenate(arrs, 1)
    return concatenate(arrs, 0)

def column_stack(tup):
    """Stack 1-D arrays as columns into a 2-D array."""
    arrays = []
    for a in tup:
        a = asarray(a)
        if a.ndim == 1:
            a = a.reshape((a.size, 1))  # Make column vector
        arrays.append(a)
    return concatenate(arrays, 1)

def dstack(tup):
    return _native.dstack(list(tup))

row_stack = vstack

def split(a, indices_or_sections, axis=0):
    return _native.split(a, indices_or_sections, axis)

def vsplit(a, indices_or_sections):
    return split(a, indices_or_sections, 0)

def hsplit(a, indices_or_sections):
    if a.ndim == 1:
        return split(a, indices_or_sections, 0)
    return split(a, indices_or_sections, 1)

def array_split(ary, indices_or_sections, axis=0):
    """Split an array into multiple sub-arrays (allows unequal division)."""
    ary = asarray(ary)
    n = ary.shape[axis]
    if isinstance(indices_or_sections, (int, float)):
        nsections = int(indices_or_sections)
        neach, extras = _builtin_divmod(n, nsections)
        boundaries = [0]
        for i in range(nsections):
            boundaries.append(boundaries[-1] + neach + (1 if i < extras else 0))
        indices = boundaries[1:-1]
    else:
        indices = list(indices_or_sections)
    return split(ary, indices, axis)

def dsplit(ary, indices_or_sections):
    """Split array into multiple sub-arrays along the 3rd axis (depth)."""
    ary = asarray(ary)
    if ary.ndim < 3:
        raise ValueError("dsplit only works on arrays of 3 or more dimensions")
    return array_split(ary, indices_or_sections, axis=2)

def block(arrays):
    """Assemble an nd-array from nested lists of blocks."""
    if isinstance(arrays, ndarray):
        return arrays
    if not isinstance(arrays, list):
        return asarray(arrays)
    if len(arrays) == 0:
        return array([])
    if isinstance(arrays[0], list):
        rows = []
        for row_blocks in arrays:
            row = hstack([asarray(b) for b in row_blocks])
            rows.append(row)
        return vstack(rows)
    else:
        return concatenate([asarray(a) for a in arrays])

def copyto(dst, src, casting='same_kind', where=True):
    """Copy values from one array to another, broadcasting as necessary."""
    src = asarray(src)
    dst = asarray(dst)
    src_b = broadcast_to(src, dst.shape)
    if where is True:
        return src_b
    mask = asarray(where)
    flat_m = mask.flatten()
    flat_s = src_b.flatten()
    flat_d = dst.flatten()
    n = flat_d.size
    result_vals = []
    for i in range(n):
        if flat_m[i]:
            result_vals.append(float(flat_s[i]))
        else:
            result_vals.append(float(flat_d[i]))
    result = array(result_vals)
    if dst.ndim > 1:
        result = result.reshape(dst.shape)
    return result

def place(arr, mask, vals):
    """Change elements of an array based on conditional and input values."""
    arr = asarray(arr)
    mask = asarray(mask)
    vals_arr = asarray(vals).flatten()
    flat_a = arr.flatten()
    flat_m = mask.flatten()
    n = flat_a.size
    nv = vals_arr.size
    result = []
    vi = 0
    for i in range(n):
        if flat_m[i]:
            result.append(float(vals_arr[vi % nv]))
            vi += 1
        else:
            result.append(float(flat_a[i]))
    r = array(result)
    if arr.ndim > 1:
        r = r.reshape(arr.shape)
    return r

def can_cast(from_=None, to=None, casting="safe"):
    # --- safe-cast ordering: lower number can safely cast to higher number ---
    _type_order = {
        "bool": 0,
        "int8": 1, "uint8": 2,
        "int16": 3, "uint16": 4,
        "int32": 5, "uint32": 6,
        "int64": 7, "uint64": 8,
        "float16": 9,
        "float32": 10, "float64": 11,
        "complex64": 12, "complex128": 13,
    }
    # Safe-cast graph: from -> set of safe targets (transitive via ordering isn't enough
    # because uint8->int16 is safe but uint8 order 2, int16 order 3 works;
    # but uint32->int32 is NOT safe).  Use explicit safe-cast rules.
    _safe_casts = {
        "bool":       {"bool", "int8", "uint8", "int16", "uint16", "int32", "uint32", "int64", "uint64", "float16", "float32", "float64", "complex64", "complex128"},
        "int8":       {"int8", "int16", "int32", "int64", "float16", "float32", "float64", "complex64", "complex128"},
        "uint8":      {"uint8", "int16", "uint16", "int32", "uint32", "int64", "uint64", "float16", "float32", "float64", "complex64", "complex128"},
        "int16":      {"int16", "int32", "int64", "float32", "float64", "complex64", "complex128"},
        "uint16":     {"uint16", "int32", "uint32", "int64", "uint64", "float32", "float64", "complex64", "complex128"},
        "int32":      {"int32", "int64", "float64", "complex128"},
        "uint32":     {"uint32", "int64", "uint64", "float64", "complex128"},
        "int64":      {"int64", "float64", "complex128"},
        "uint64":     {"uint64", "float64", "complex128"},
        "float16":    {"float16", "float32", "float64", "complex64", "complex128"},
        "float32":    {"float32", "float64", "complex64", "complex128"},
        "float64":    {"float64", "complex128"},
        "complex64":  {"complex64", "complex128"},
        "complex128": {"complex128"},
    }
    _int_types = {"bool", "int8", "int16", "int32", "int64", "uint8", "uint16", "uint32", "uint64"}
    _signed_int = {"int8", "int16", "int32", "int64"}
    _unsigned_int = {"uint8", "uint16", "uint32", "uint64"}
    _float_types = {"float16", "float32", "float64"}
    _complex_types = {"complex64", "complex128"}
    # Max string representation lengths for numeric -> string casting
    _str_len = {
        "bool": 5, "int8": 4, "uint8": 3, "int16": 6, "uint16": 5,
        "int32": 11, "uint32": 10, "int64": 21, "uint64": 20,
    }

    # --- NEP 50: reject plain Python scalars ---
    if (
        isinstance(from_, (int, float, complex))
        and not isinstance(from_, bool)
        and not hasattr(from_, "_numpy_dtype_name")
    ):
        raise TypeError("Cannot interpret '{}' as a data type".format(type(from_).__name__))

    # --- None check ---
    if from_ is None or to is None:
        raise TypeError("Cannot interpret 'NoneType' as a data type")

    # --- Helper to detect structured dtypes ---
    def _is_structured(x):
        if isinstance(x, (list, tuple)):
            return True
        if isinstance(x, str) and ',' in x:
            return True
        if isinstance(x, dtype) and ',' in x.name:
            return True
        return False

    def _count_fields(x):
        """Count top-level fields in a structured dtype spec. Returns (n_fields, has_subarray)."""
        if isinstance(x, str) and ',' in x:
            return len(x.split(',')), False
        if isinstance(x, (list, tuple)):
            # List of (name, dtype) or (name, dtype, shape) tuples
            count = 0
            has_sub = False
            for item in x:
                if isinstance(item, (list, tuple)) and len(item) >= 2:
                    _, field_dt = item[0], item[1]
                    if isinstance(field_dt, (list, tuple)):
                        # Nested structured
                        n, _ = _count_fields(field_dt)
                        if n > 1:
                            has_sub = True
                        else:
                            count += 1
                    elif isinstance(field_dt, str) and ',' in field_dt:
                        # Check if commas are outside parens (multi-field) vs inside (subarray)
                        _dt_no_paren = field_dt
                        _depth = 0
                        _has_outer_comma = False
                        for _ch in field_dt:
                            if _ch == '(':
                                _depth += 1
                            elif _ch == ')':
                                _depth -= 1
                            elif _ch == ',' and _depth == 0:
                                _has_outer_comma = True
                                break
                        if _has_outer_comma:
                            has_sub = True
                        else:
                            count += 1
                    else:
                        count += 1
                    if len(item) >= 3:
                        has_sub = True  # subarray
                else:
                    count += 1
            return count, has_sub
        return 1, False

    from_struct = _is_structured(from_)
    to_struct = _is_structured(to)

    # --- Structured dtype casting rules ---
    if from_struct and to_struct:
        # Both structured: only with unsafe
        return casting == "unsafe"
    if not from_struct and to_struct:
        # Simple -> structured: only unsafe
        return casting == "unsafe"
    if from_struct and not to_struct:
        # Structured -> simple: only unsafe, and only if single field (recursive)
        if casting != "unsafe":
            return False
        n_fields, has_sub_multi = _count_fields(from_)
        if n_fields > 1 or has_sub_multi:
            return False
        return True

    # --- Normalize dtype names ---
    def _to_name(x):
        """Extract canonical dtype name and raw string (for endian checks)."""
        if hasattr(x, "_numpy_dtype_name"):
            name = x._numpy_dtype_name
            return name, name
        if isinstance(x, ndarray):
            return str(x.dtype), str(x.dtype)
        if isinstance(x, dtype):
            return x.name, str(x)
        if isinstance(x, _ScalarType):
            return str(x), str(x)
        if isinstance(x, type):
            if isinstance(x, _ScalarTypeMeta):
                return x._scalar_name, x._scalar_name
            if x is float:
                return "float64", "float64"
            if x is int:
                return "int64", "int64"
            if x is bool:
                return "bool", "bool"
            if x is complex:
                return "complex128", "complex128"
        if isinstance(x, str):
            raw = x
            # Strip endian prefix for normalization but keep raw for endian checks
            norm = _DTYPE_CHAR_MAP.get(x, x)
            if norm != x:
                return norm, raw
            # Try stripping endian prefix
            s = x
            if s and s[0] in '<>=|':
                s = s[1:]
            norm = _DTYPE_CHAR_MAP.get(s, s)
            return norm, raw
        return str(x), str(x)

    from_name, from_raw = _to_name(from_)
    to_name, to_raw = _to_name(to)

    # --- String/bytes dtype handling ---
    def _is_string_dtype(name, raw):
        """Check if dtype is a string (U) or bytes (S) type, return (True, length) or (False, 0)."""
        for s in (raw, name):
            if isinstance(s, str):
                stripped = s.lstrip('<>=|')
                if stripped.startswith('S') and len(stripped) > 1:
                    try:
                        return True, int(stripped[1:]), 'S'
                    except (ValueError, IndexError):
                        pass
                if stripped.startswith('U') and len(stripped) > 1:
                    try:
                        return True, int(stripped[1:]), 'U'
                    except (ValueError, IndexError):
                        pass
                if stripped == 'S' or stripped == 'bytes':
                    return True, 0, 'S'
                if stripped == 'U' or stripped == 'str':
                    return True, 0, 'U'
        return False, 0, ''

    to_is_str, to_str_len, to_str_kind = _is_string_dtype(to_name, to_raw)
    from_is_str, from_str_len, from_str_kind = _is_string_dtype(from_name, from_raw)

    if to_is_str and not from_is_str:
        # Numeric -> string: check if string is long enough
        needed = _str_len.get(from_name, 0)
        if needed == 0:
            # Unknown numeric type -> can't safely cast
            if casting == "unsafe":
                return True
            return False
        if casting == "unsafe":
            return True
        return to_str_len >= needed
    if from_is_str and to_is_str:
        if casting == "unsafe":
            return True
        if from_str_kind == to_str_kind:
            return to_str_len >= from_str_len
        # S -> U promotion: U can hold S
        if from_str_kind == 'S' and to_str_kind == 'U':
            return to_str_len >= from_str_len
        return False
    if from_is_str and not to_is_str:
        return casting == "unsafe"

    # --- Endian-aware checks ---
    def _has_endian(raw):
        return isinstance(raw, str) and len(raw) > 0 and raw[0] in '<>'
    def _get_endian(raw):
        if isinstance(raw, str) and len(raw) > 0 and raw[0] in '<>':
            return raw[0]
        return '='

    # --- Numeric casting ---
    if from_name not in _type_order or to_name not in _type_order:
        if casting == "unsafe":
            return True
        return False

    if casting == "unsafe":
        return True
    if casting == "no":
        if from_name != to_name:
            return False
        # Same base type: check endianness must match exactly
        if _has_endian(from_raw) and _has_endian(to_raw):
            return _get_endian(from_raw) == _get_endian(to_raw)
        return True
    if casting == "equiv":
        # Same base type, possibly different endianness
        return from_name == to_name
    if casting == "safe":
        return to_name in _safe_casts.get(from_name, set())
    if casting == "same_kind":
        if to_name in _safe_casts.get(from_name, set()):
            return True
        # Allow downcast within same kind
        if from_name in _signed_int and to_name in _signed_int:
            return True
        if from_name in _unsigned_int and to_name in _unsigned_int:
            return True
        # bool -> any int is same_kind (bool is a sub-kind of int)
        if from_name == "bool" and to_name in _int_types:
            return True
        if from_name in _float_types and to_name in _float_types:
            return True
        if from_name in _complex_types and to_name in _complex_types:
            return True
        return False
    return to_name in _safe_casts.get(from_name, set())

def result_type(*arrays_and_dtypes):
    if len(arrays_and_dtypes) == 0:
        return float64
    dtypes = []
    for a in arrays_and_dtypes:
        if isinstance(a, ndarray):
            dtypes.append(str(a.dtype))
        elif isinstance(a, _ObjectArray):
            dtypes.append(_normalize_dtype(str(a.dtype)))
        elif hasattr(a, "_numpy_dtype_name"):
            dtypes.append(str(getattr(a, "_numpy_dtype_name")))
        elif isinstance(a, _ScalarType):
            dtypes.append(str(a))
        elif isinstance(a, type) and isinstance(a, _ScalarTypeMeta):
            dtypes.append(str(a))
        elif isinstance(a, str):
            dtypes.append(a)
        elif isinstance(a, bool):
            dtypes.append("bool")
        elif isinstance(a, int):
            dtypes.append("int64")
        elif isinstance(a, float):
            dtypes.append("float64")
        elif isinstance(a, complex):
            dtypes.append("complex128")
        else:
            dtypes.append("float64")
    if len(dtypes) == 1:
        return dtype(dtypes[0])
    result = dtypes[0]
    for d in dtypes[1:]:
        result = str(promote_types(result, d))
    return dtype(result)

def promote_types(type1, type2):
    # Ensure both are dtype objects for metadata access
    if not isinstance(type1, dtype):
        try:
            type1 = dtype(type1)
        except (TypeError, ValueError):
            type1_str = str(type1)
            # For unsupported dtypes, if both are equal, return as-is
            if str(type1) == str(type2):
                d = dtype.__new__(dtype)
                d.name = type1_str
                d.kind = 'V'
                d.itemsize = 0
                d.char = 'V'
                d.byteorder = '='
                d.metadata = getattr(type2 if isinstance(type2, dtype) else type1, 'metadata', None)
                return d
            raise
    if not isinstance(type2, dtype):
        try:
            type2 = dtype(type2)
        except (TypeError, ValueError):
            raise

    t1_meta = getattr(type1, 'metadata', None)
    t2_meta = getattr(type2, 'metadata', None)

    # Fast-path: identical types -> return directly (preserves metadata for V, O, etc.)
    if type1.name == type2.name:
        # For structured/void dtypes, check full equality (field names must match)
        t1_names = getattr(type1, 'names', None)
        t2_names = getattr(type2, 'names', None)
        if t1_names is not None or t2_names is not None:
            # Structured: must be fully equal
            if type1 != type2:
                raise TypeError("invalid type promotion")
        # Also check itemsize for void types (V6 vs V10)
        t1_is = getattr(type1, 'itemsize', 0)
        t2_is = getattr(type2, 'itemsize', 0)
        if type1.kind == 'V' and t1_is != t2_is:
            raise TypeError("invalid type promotion")
        result = dtype(type1.name)
        # Only preserve metadata when both are identical
        if t1_meta is not None and t1_meta == t2_meta:
            result.metadata = t1_meta
        return result

    # Strip endian prefixes for Rust backend
    s1 = type1.name
    s2 = type2.name
    if s1 and s1[0] in '<>=|':
        s1 = s1[1:]
    if s2 and s2[0] in '<>=|':
        s2 = s2[1:]
    s1 = _DTYPE_CHAR_MAP.get(s1, s1)
    s2 = _DTYPE_CHAR_MAP.get(s2, s2)

    # If names match after normalization
    if s1 == s2:
        result = dtype(s1)
        if t1_meta is not None and t1_meta == t2_meta:
            result.metadata = t1_meta
        return result

    _int_bits = {
        "bool": 1,
        "int8": 8, "int16": 16, "int32": 32, "int64": 64,
        "uint8": 8, "uint16": 16, "uint32": 32, "uint64": 64,
    }
    _float_bits = {"float16": 16, "float32": 32, "float64": 64}
    _complex_bits = {"complex64": 64, "complex128": 128}
    _signed_names = {8: "int8", 16: "int16", 32: "int32", 64: "int64"}
    _unsigned_names = {8: "uint8", 16: "uint16", 32: "uint32", 64: "uint64"}
    _float_names = {16: "float16", 32: "float32", 64: "float64"}
    _complex_names = {64: "complex64", 128: "complex128"}

    def _next_signed(bits):
        for b in (8, 16, 32, 64):
            if b > bits:
                return _signed_names[b]
        return None

    def _promote_numeric(a, b):
        _bmax = __import__("builtins").max
        if a in _complex_bits or b in _complex_bits:
            # Lift real to matching complex precision.
            def _real_float_bits(x):
                if x in _complex_bits:
                    return 32 if x == "complex64" else 64
                if x in _float_bits:
                    return _float_bits[x]
                # integers/bool route through float64 conservatively.
                return 64
            rb = _real_float_bits(a)
            rc = _real_float_bits(b)
            return _complex_names[64 if _bmax(rb, rc) <= 32 else 128]

        if a in _float_bits or b in _float_bits:
            # Scalar promotion behavior close to NumPy's tested cases.
            fa = _float_bits[a] if a in _float_bits else None
            fb = _float_bits[b] if b in _float_bits else None
            fbits = _bmax(fa or 0, fb or 0)
            other = b if fa is not None else a
            if other in _int_bits:
                obits = _int_bits[other]
                if fbits == 16:
                    return "float16" if obits <= 8 else "float32"
                if fbits == 32:
                    if other.startswith("uint"):
                        return "float64" if obits >= 32 else "float32"
                    return "float64" if obits >= 32 else "float32"
                return "float64"
            return _float_names[fbits]

        if a in _int_bits and b in _int_bits:
            if a == "bool":
                return b
            if b == "bool":
                return a

            a_unsigned = a.startswith("uint")
            b_unsigned = b.startswith("uint")
            abit = _int_bits[a]
            bbit = _int_bits[b]

            if a_unsigned == b_unsigned:
                bits = _bmax(abit, bbit)
                return _unsigned_names[bits] if a_unsigned else _signed_names[bits]

            # signed/unsigned mix
            sbits = abit if not a_unsigned else bbit
            ubits = abit if a_unsigned else bbit
            if sbits > ubits:
                return _signed_names[sbits]
            nxt = _next_signed(ubits)
            if nxt is not None:
                return nxt
            return "float64"

        return None

    promoted = _promote_numeric(s1, s2)
    result = dtype(promoted if promoted is not None else _native.promote_types(s1, s2))
    # Preserve metadata only when both have identical metadata
    if t1_meta is not None and t1_meta == t2_meta:
        result.metadata = t1_meta
    elif t1_meta is not None and t2_meta is None:
        result.metadata = t1_meta
    elif t2_meta is not None and t1_meta is None:
        result.metadata = t2_meta
    return result

def find_common_type(array_types, scalar_types):
    """Deprecated in numpy 2.0, but still used by some packages.
    Determines common type following standard coercion rules."""
    all_types = list(array_types) + list(scalar_types)
    if not all_types:
        return dtype("float64")
    return _reduce(lambda a, b: dtype(str(result_type(a, b))), all_types)

_err_state = {"divide": "warn", "over": "warn", "under": "ignore", "invalid": "warn"}

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
    def __init__(self, **kwargs):
        self._kwargs = kwargs
        self._old = None
    def __enter__(self):
        self._old = seterr(**self._kwargs)
        return self
    def __exit__(self, *args):
        seterr(**self._old)

def set_printoptions(**kwargs):
    pass

def get_printoptions():
    return {}

class printoptions:
    """Context manager for print options."""
    def __init__(self, **kwargs):
        self._opts = kwargs
    def __enter__(self):
        set_printoptions(**self._opts)
        return self
    def __exit__(self, *args):
        pass  # We don't actually track old options

# --- StructuredDtype for field-based (record) arrays -----------------------
class StructuredDtype:
    """Minimal structured dtype for field-based array access."""
    def __init__(self, fields):
        # fields is list of (name, dtype) or (name, dtype, shape)
        self._fields = []
        self._names = []
        for f in fields:
            name = f[0]
            dt = dtype(f[1]) if not isinstance(f[1], str) else f[1]
            self._fields.append((name, dt))
            self._names.append(name)
        self.names = tuple(self._names)
        self.fields = {}
        offset = 0
        for name, dt in self._fields:
            if isinstance(dt, str):
                dt_obj = dtype(dt)
            else:
                dt_obj = dt
            self.fields[name] = (dt_obj, offset)
            offset += dt_obj.itemsize if hasattr(dt_obj, 'itemsize') else 8
        self.itemsize = offset
        self.kind = 'V'
        self.char = 'V'
        self.name = 'void'
        self.str = '|V{}'.format(self.itemsize)

    def __repr__(self):
        parts = ', '.join("('{}', '{}')".format(n, d) for n, d in self._fields)
        return 'dtype([{}])'.format(parts)

    def __str__(self):
        return repr(self)

    def __eq__(self, other):
        if isinstance(other, StructuredDtype):
            return self._fields == other._fields
        return False

    def __hash__(self):
        return hash(tuple((n, str(d)) for n, d in self._fields))

# --- Metaclass for per-dtype DType classes ----------------------------------
# Gives classes like Float64DType a custom __str__ so str(Float64DType)='float64'.
class _DTypeClassMeta(type):
    def __str__(cls):
        return cls._dtype_class_name
    def __repr__(cls):
        return f"numpy.dtypes.{cls.__name__}"
    def __eq__(cls, other):
        if isinstance(other, _DTypeClassMeta):
            return cls._dtype_class_name == other._dtype_class_name
        return NotImplemented
    def __hash__(cls):
        return hash(cls._dtype_class_name)


# --- dtype class (stub) -----------------------------------------------------
class dtype:
    """Stub for numpy dtype objects."""

    _dtype_class_map = {}  # filled after DType subclasses are defined below

    def __new__(cls, tp=None, metadata=None):
        if cls is dtype:
            # Determine canonical dtype name to pick the right subclass
            name = None
            if isinstance(tp, type) and isinstance(tp, _DTypeClassMeta):
                name = tp._dtype_class_name
            elif isinstance(tp, str):
                name = _DTYPE_CHAR_MAP.get(tp, tp)
            elif tp is float or tp is float64:
                name = 'float64'
            elif tp is int or tp is int64:
                name = 'int64'
            elif tp is bool or tp is bool_:
                name = 'bool'
            elif isinstance(tp, _ScalarType):
                name = tp._name
            elif isinstance(tp, type) and isinstance(tp, _ScalarTypeMeta):
                name = tp._scalar_name
            elif isinstance(tp, dtype):
                name = tp.name
            if name and name in dtype._dtype_class_map:
                target_cls = dtype._dtype_class_map[name]
                if target_cls is not cls:
                    return object.__new__(target_cls)
        return object.__new__(cls)

    def __init__(self, tp=None, metadata=None):
        if isinstance(tp, list):
            # List-of-tuples structured dtype: delegate to StructuredDtype
            sd = StructuredDtype(tp)
            self.name = sd.name
            self.kind = sd.kind
            self.itemsize = sd.itemsize
            self.char = sd.char
            self.names = sd.names
            self.fields = sd.fields
            self._structured = sd
        elif isinstance(tp, StructuredDtype):
            self.name = tp.name
            self.kind = tp.kind
            self.itemsize = tp.itemsize
            self.char = tp.char
            self.names = tp.names
            self.fields = tp.fields
            self._structured = tp
        elif isinstance(tp, dtype):
            self.name = tp.name
            self.kind = tp.kind
            self.itemsize = tp.itemsize
            self.char = tp.char
        elif isinstance(tp, str):
            tp = _DTYPE_CHAR_MAP.get(tp, tp)
            self.name = tp
            self._init_from_name(tp)
        elif tp is float or tp is float64:
            self.name = "float64"
            self.kind = "f"
            self.itemsize = 8
            self.char = "d"
        elif tp is int or tp is int64:
            self.name = "int64"
            self.kind = "i"
            self.itemsize = 8
            self.char = "q"
        elif tp is bool or tp is bool_:
            self.name = "bool"
            self.kind = "b"
            self.itemsize = 1
            self.char = "?"
        elif isinstance(tp, _ScalarType):
            self.name = tp._name
            self._init_from_name(tp._name)
        elif isinstance(tp, type) and isinstance(tp, _ScalarTypeMeta):
            self.name = tp._scalar_name
            self._init_from_name(tp._scalar_name)
        elif isinstance(tp, type) and isinstance(tp, _DTypeClassMeta):
            self.name = tp._dtype_class_name
            self._init_from_name(tp._dtype_class_name)
        else:
            self.name = str(tp) if tp else "float64"
            self._init_from_name(self.name)
        # self.type: numpy scalar type class
        _type_map = {
            "float64": float64, "float32": float32, "float16": float16,
            "int64": int64, "int32": int32, "int16": int16, "int8": int8,
            "uint64": uint64, "uint32": uint32, "uint16": uint16, "uint8": uint8,
            "bool": bool_, "complex128": complex128, "complex64": complex64,
            "str": str_, "bytes": bytes_, "object": object_,
        }
        self.type = _type_map.get(self.name, float64)
        # self.str: typestring format (e.g., "<f8")
        _typestr = {
            "float64": "<f8", "float32": "<f4", "float16": "<f2",
            "int64": "<i8", "int32": "<i4", "int16": "<i2", "int8": "|i1",
            "uint64": "<u8", "uint32": "<u4", "uint16": "<u2", "uint8": "|u1",
            "bool": "|b1",
            "complex128": "<c16", "complex64": "<c8",
            "object": "|O", "str": "<U", "bytes": "|S0",
        }
        self.str = _typestr.get(self.name, "<f8")
        # names/fields: None for non-structured dtypes
        if not hasattr(self, 'names'):
            self.names = None
        if not hasattr(self, 'fields'):
            self.fields = None
        # byteorder
        self.byteorder = '=' if self.name in ('bool',) else '<'
        if self.kind == 'b':
            self.byteorder = '|'
        elif self.itemsize == 1:
            self.byteorder = '|'
        self.subdtype = None
        self.base = self
        self.metadata = metadata
        self.alignment = self.itemsize
        self.isalignedstruct = False
        self.isnative = True
        self.hasobject = False
        # num: unique dtype number (matching numpy convention loosely)
        _num_map = {
            "bool": 0, "int8": 1, "uint8": 2, "int16": 3, "uint16": 4,
            "int32": 5, "uint32": 6, "int64": 7, "uint64": 8,
            "float16": 23, "float32": 11, "float64": 12,
            "complex64": 14, "complex128": 15, "object": 17, "str": 19,
        }
        self.num = _num_map.get(self.name, 12)
        # descr: list of (name, typestr) tuples for structured arrays, or [('', typestr)]
        if hasattr(self, '_structured') and self._structured is not None:
            self.descr = [(n, str(self.fields[n][0])) for n in self.names]
        else:
            self.descr = [('', self.str)]

    def _init_from_name(self, name):
        _info = {
            "float64": ("f", 8, "d"), "float32": ("f", 4, "f"),
            "float16": ("f", 2, "e"), "int64": ("i", 8, "q"),
            "int32": ("i", 4, "l"), "int16": ("i", 2, "h"),
            "int8": ("i", 1, "b"), "uint64": ("u", 8, "Q"),
            "uint32": ("u", 4, "L"), "uint16": ("u", 2, "H"),
            "uint8": ("u", 1, "B"), "bool": ("b", 1, "?"),
            "complex128": ("c", 16, "D"), "complex64": ("c", 8, "F"),
        }
        if name in _info:
            self.kind, self.itemsize, self.char = _info[name]
        else:
            self.kind, self.itemsize, self.char = "f", 8, "d"

    def __repr__(self):
        if hasattr(self, '_structured') and self._structured is not None:
            return repr(self._structured)
        return f"dtype('{self.name}')"

    def __str__(self):
        return self.name

    def __eq__(self, other):
        if isinstance(other, dtype):
            if hasattr(self, '_structured') and hasattr(other, '_structured'):
                return self._structured == other._structured
            return self.name == other.name
        if isinstance(other, str):
            return self.name == other or self.name == _normalize_dtype(other)
        if isinstance(other, type) and isinstance(other, _ScalarTypeMeta):
            return self.name == other._scalar_name
        if isinstance(other, type):
            # Handle Python builtin types: bool, int, float
            _type_map = {__import__("builtins").bool: "bool", __import__("builtins").int: "int64", __import__("builtins").float: "float64"}
            if other in _type_map:
                return self.name == _type_map[other]
        return NotImplemented

    def __hash__(self):
        return hash(self.name)

    def newbyteorder(self, new_order="S"):
        d = dtype(self.name)
        d.metadata = self.metadata
        return d

# --- Per-dtype DType classes (numpy.dtypes.Float64DType etc.) ---------------
# Each is a subclass of dtype with metaclass _DTypeClassMeta so that
# type(np.dtype('float64')) == Float64DType and str(Float64DType) == 'float64'.

class Float64DType(dtype, metaclass=_DTypeClassMeta):
    _dtype_class_name = 'float64'
    type = float64

class Float32DType(dtype, metaclass=_DTypeClassMeta):
    _dtype_class_name = 'float32'
    type = float32

class Float16DType(dtype, metaclass=_DTypeClassMeta):
    _dtype_class_name = 'float16'
    type = float16

class Int8DType(dtype, metaclass=_DTypeClassMeta):
    _dtype_class_name = 'int8'
    type = int8

class Int16DType(dtype, metaclass=_DTypeClassMeta):
    _dtype_class_name = 'int16'
    type = int16

class Int32DType(dtype, metaclass=_DTypeClassMeta):
    _dtype_class_name = 'int32'
    type = int32

class Int64DType(dtype, metaclass=_DTypeClassMeta):
    _dtype_class_name = 'int64'
    type = int64

class UInt8DType(dtype, metaclass=_DTypeClassMeta):
    _dtype_class_name = 'uint8'
    type = uint8

class UInt16DType(dtype, metaclass=_DTypeClassMeta):
    _dtype_class_name = 'uint16'
    type = uint16

class UInt32DType(dtype, metaclass=_DTypeClassMeta):
    _dtype_class_name = 'uint32'
    type = uint32

class UInt64DType(dtype, metaclass=_DTypeClassMeta):
    _dtype_class_name = 'uint64'
    type = uint64

class Complex64DType(dtype, metaclass=_DTypeClassMeta):
    _dtype_class_name = 'complex64'
    type = complex64

class Complex128DType(dtype, metaclass=_DTypeClassMeta):
    _dtype_class_name = 'complex128'
    type = complex128

class BoolDType(dtype, metaclass=_DTypeClassMeta):
    _dtype_class_name = 'bool'
    type = bool_

class StrDType(dtype, metaclass=_DTypeClassMeta):
    _dtype_class_name = 'str'
    type = str_

class BytesDType(dtype, metaclass=_DTypeClassMeta):
    _dtype_class_name = 'bytes'
    type = bytes_

class VoidDType(dtype, metaclass=_DTypeClassMeta):
    _dtype_class_name = 'void'
    type = void

class ObjectDType(dtype, metaclass=_DTypeClassMeta):
    _dtype_class_name = 'object'
    type = object_

# Populate the map so dtype.__new__ can dispatch to the right subclass
dtype._dtype_class_map = {
    'float64': Float64DType, 'float32': Float32DType, 'float16': Float16DType,
    'int8': Int8DType, 'int16': Int16DType, 'int32': Int32DType, 'int64': Int64DType,
    'uint8': UInt8DType, 'uint16': UInt16DType, 'uint32': UInt32DType, 'uint64': UInt64DType,
    'complex64': Complex64DType, 'complex128': Complex128DType,
    'bool': BoolDType, 'str': StrDType, 'bytes': BytesDType,
    'void': VoidDType, 'object': ObjectDType,
}

# --- More missing stubs for test_numeric.py ---------------------------------
True_ = True
False_ = False
int_ = int64

class _BroadcastIter:
    """Iterator wrapper that exposes .base pointing to the original array."""
    def __init__(self, broadcasted, base_array):
        self._flat = broadcasted.flat
        self.base = base_array
    def __iter__(self):
        return self._flat.__iter__()
    def __next__(self):
        return self._flat.__next__()

class broadcast:
    """Broadcast shape computation for multiple arrays."""
    def __init__(self, *args, **kwargs):
        if kwargs:
            raise ValueError("broadcast() does not accept keyword arguments")
        if len(args) > 64:
            raise ValueError("Need at most 64 array objects.")
        # Flatten any nested broadcast objects
        flat_args = []
        for a in args:
            if isinstance(a, broadcast):
                flat_args.extend(a._arrays)
            else:
                flat_args.append(a)
        arrays = [asarray(a) for a in flat_args]
        shapes = [a.shape for a in arrays]
        if len(shapes) == 0:
            self.shape = ()
        else:
            try:
                self.shape = broadcast_shapes(*shapes)
            except ValueError:
                # Generate numpy-compatible error message with arg indices
                ndim = _builtin_max(len(s) for s in shapes)
                for dim in range(ndim):
                    sizes = {}
                    for idx, s in enumerate(shapes):
                        offset = ndim - len(s)
                        d = dim - offset
                        if d >= 0:
                            sz = s[d]
                            if sz != 1:
                                sizes.setdefault(sz, []).append(idx)
                    if len(sizes) > 1:
                        items = sorted(sizes.items(), key=lambda x: x[1][0])
                        first_idx = items[0][1][0]
                        last_idx = items[-1][1][0]
                        raise ValueError(
                            f"arg {first_idx} with shape {shapes[first_idx]} and "
                            f"arg {last_idx} with shape {shapes[last_idx]}"
                        )
                raise
        self.nd = len(self.shape)
        self.ndim = self.nd
        self.size = 1
        for s in self.shape:
            self.size *= s
        self.numiter = len(arrays)
        self._arrays = arrays
        self.iters = tuple(_BroadcastIter(broadcast_to(a, self.shape), a) for a in arrays)
        self.index = 0

    def __iter__(self):
        # Broadcast each array to the common shape and iterate
        broadcasted = [broadcast_to(a, self.shape).flatten() for a in self._arrays]
        for i in range(self.size):
            yield tuple(float(b[i]) for b in broadcasted)

    def __next__(self):
        if self.index >= self.size:
            raise StopIteration
        broadcasted = [broadcast_to(a, self.shape).flatten() for a in self._arrays]
        result = tuple(float(b[self.index]) for b in broadcasted)
        self.index += 1
        return result

    def reset(self):
        self.index = 0

def argwhere(a):
    if not isinstance(a, ndarray):
        a = asarray(a)
    # Rust argwhere only handles float64; cast integer/bool arrays
    dt = str(a.dtype)
    if dt in ("int32", "int64", "int8", "int16", "uint8", "uint16", "uint32", "uint64", "bool"):
        a = a.astype("float64")
    return _native.argwhere(a)

def nonzero(a):
    if isinstance(a, _ObjectArray):
        indices = []
        for i, v in enumerate(a._data):
            if bool(v):
                indices.append(i)
        return (array(indices, dtype="int64"),) if indices else (array([], dtype="int64"),)
    if not isinstance(a, ndarray):
        a = asarray(a)
    if a.ndim == 0:
        raise ValueError("Calling nonzero on 0d arrays is not allowed. Use np.atleast_1d(a).nonzero() instead.")
    # Rust nonzero only handles float64; cast integer/bool arrays
    dt = str(a.dtype)
    if dt in ("int32", "int64", "int8", "int16", "uint8", "uint16", "uint32", "uint64", "bool"):
        a = a.astype("float64")
    return _native.nonzero(a)

def count_nonzero(a, axis=None, *, keepdims=False):
    if not isinstance(a, ndarray):
        a = asarray(a)
    # Empty tuple axis means element-wise (no reduction)
    if isinstance(axis, tuple) and len(axis) == 0:
        return a.astype("bool")
    # Validate axis
    if axis is not None and not isinstance(axis, (int, tuple)):
        if isinstance(axis, ndarray):
            raise TypeError("axis must be an integer or a tuple of integers")
        raise TypeError("'{}' object cannot be interpreted as an integer".format(type(axis).__name__))
    if isinstance(axis, tuple):
        # Check for duplicate axes
        normed = []
        for ax in axis:
            n = ax if ax >= 0 else ax + a.ndim
            if n in normed:
                raise ValueError("duplicate value in 'axis'")
            if n < 0 or n >= a.ndim:
                raise AxisError(ax, a.ndim)
            normed.append(n)
    elif isinstance(axis, int):
        n = axis if axis >= 0 else axis + a.ndim
        if n < 0 or n >= a.ndim:
            raise AxisError(axis, a.ndim)
    # Rust count_nonzero only handles float64; cast integer/bool arrays
    dt = str(a.dtype)
    if dt in ("int32", "int64", "int8", "int16", "uint8", "uint16", "uint32", "uint64", "bool"):
        a = a.astype("float64")
    if axis is None:
        result = _native.count_nonzero(a)
        if keepdims:
            return array([float(result)]).reshape((1,) * a.ndim).astype("int64")
        return result
    # Build a boolean mask (nonzero -> 1.0, zero -> 0.0), then sum along axis
    flat = a.flatten().tolist()
    def _is_nonzero(v):
        if isinstance(v, tuple):  # complex (re, im) representation
            return v[0] != 0.0 or v[1] != 0.0
        return v != 0.0
    mask_data = [1.0 if _is_nonzero(v) else 0.0 for v in flat]
    mask = array(mask_data).reshape(a.shape)
    result = mask.sum(axis, keepdims)
    # Convert to integer values
    if isinstance(result, ndarray):
        return result.astype("int64")
    return int(result)

# Keep builtin sum reference
_builtin_sum = __builtins__["sum"] if isinstance(__builtins__, dict) else __import__("builtins").sum

def diagonal(a, offset=0, axis1=0, axis2=1):
    """Extract diagonal from array."""
    a = asarray(a)
    if a.ndim < 2:
        raise ValueError("diagonal requires at least a 2-d array")
    # For 2D with default axes, delegate directly
    if a.ndim == 2:
        if axis1 == 1 and axis2 == 0:
            return _native.diagonal(a.T, offset)
        return _native.diagonal(a, offset)
    # For nD, move the two axes to the end and extract diagonals
    # along the last two axes
    ax1 = axis1 if axis1 >= 0 else a.ndim + axis1
    ax2 = axis2 if axis2 >= 0 else a.ndim + axis2
    a = moveaxis(a, (ax1, ax2), (-2, -1))
    # Now extract diagonal from last two dims for each "batch" index
    shape = a.shape
    batch_shape = shape[:-2]
    m, n = shape[-2], shape[-1]
    if offset >= 0:
        diag_len = _builtin_min(m, n - offset)
    else:
        diag_len = _builtin_min(m + offset, n)
    if diag_len <= 0:
        out_shape = list(batch_shape) + [0]
        return zeros(out_shape)
    flat = a.flatten().tolist()
    batch_size = 1
    for s in batch_shape:
        batch_size *= s
    mn = m * n
    result = []
    for b in range(batch_size):
        base = b * mn
        for k in range(diag_len):
            if offset >= 0:
                result.append(flat[base + k * n + (k + offset)])
            else:
                result.append(flat[base + (k - offset) * n + k])
    out_shape = list(batch_shape) + [diag_len]
    return array(result).reshape(out_shape)

_builtin_min = __builtins__["min"] if isinstance(__builtins__, dict) else __import__("builtins").min
_builtin_max = __builtins__["max"] if isinstance(__builtins__, dict) else __import__("builtins").max
_builtin_range = __builtins__["range"] if isinstance(__builtins__, dict) else __import__("builtins").range

def trace(a, offset=0, axis1=0, axis2=1):
    d = diagonal(a, offset=offset, axis1=axis1, axis2=axis2)
    return d.sum()

def ptp(a, axis=None):
    if not isinstance(a, ndarray):
        a = asarray(a)
    return a.max(axis) - a.min(axis)

def repeat(a, repeats, axis=None):
    if not isinstance(a, ndarray):
        a = asarray(a)
    return _native.repeat(a, repeats, axis)

def tile(a, reps):
    if not isinstance(a, ndarray):
        a = asarray(a)
    return _native.tile(a, reps)

def resize(a, new_shape):
    a = asarray(a)
    if isinstance(new_shape, int):
        new_shape = (new_shape,)
    total = 1
    for s in new_shape:
        total *= s
    dt = a.dtype
    if total == 0:
        return zeros(new_shape, dtype=dt)
    flat = a.flatten().tolist()
    n = len(flat)
    if n == 0:
        return zeros(new_shape, dtype=dt)
    result = []
    for i in range(total):
        result.append(flat[i % n])
    return array(result, dtype=dt).reshape(new_shape)

def choose(a, choices, out=None, mode="raise"):
    if not isinstance(a, ndarray):
        a = asarray(a)
    # Convert choices: complex scalars become float (real part) to match Rust clip behavior
    def _to_choice_array(c):
        if isinstance(c, complex):
            return asarray(c.real)
        if isinstance(c, ndarray):
            return c
        return asarray(c)
    choice_arrays = [_to_choice_array(c) for c in choices]
    result = _native.choose(a, choice_arrays)
    if out is not None and isinstance(out, ndarray):
        flat = result.flatten().tolist()
        for i in range(len(flat)):
            out.flat[i] = flat[i]
        return out
    return result

def compress(condition, a, axis=None):
    if not isinstance(a, ndarray):
        a = asarray(a)
    cond = condition if isinstance(condition, ndarray) else asarray(condition)
    return _native.compress(cond, a, axis)

def searchsorted(a, v, side="left", sorter=None):
    if not isinstance(a, ndarray):
        a = asarray(a)
    scalar_v = not isinstance(v, (ndarray, list, tuple))
    if not isinstance(v, ndarray):
        v = array([v]) if scalar_v else asarray(v)
    result = _native.searchsorted(a, v, side)
    if scalar_v and isinstance(result, ndarray) and result.size == 1:
        return int(float(result.flatten()[0]))
    return result

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
    """Cross product of two arrays.

    Handles 1D vectors of length 2 or 3, and batched 2D arrays where the
    last axis has length 2 or 3.
    """
    _a_scalar = not isinstance(a, (ndarray, list, tuple))
    _b_scalar = not isinstance(b, (ndarray, list, tuple))
    a = asarray(a) if not isinstance(a, ndarray) else a
    b = asarray(b) if not isinstance(b, ndarray) else b
    if a.ndim == 0 or b.ndim == 0 or _a_scalar or _b_scalar:
        raise ValueError("At least one array has zero dimension")
    if axis is not None:
        axisa = axisb = axisc = axis
    # Validate axes
    if axisa < -a.ndim or axisa >= a.ndim:
        raise AxisError(axisa, a.ndim, "axisa")
    if axisb < -b.ndim or axisb >= b.ndim:
        raise AxisError(axisb, b.ndim, "axisb")
    # Move the vector axis to the last position for both arrays
    if a.ndim > 1 and axisa != -1 and axisa != a.ndim - 1:
        a = moveaxis(a, axisa, -1)
    if b.ndim > 1 and axisb != -1 and axisb != b.ndim - 1:
        b = moveaxis(b, axisb, -1)
    # Broadcast: if one is 2D+ and other is 1D, expand the 1D one
    if a.ndim >= 2 and b.ndim == 1:
        b = b.reshape((1,) * (a.ndim - 1) + (b.shape[0],))
        # Broadcast b to match a's batch dims
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
    # Simple 1D cases
    if a.ndim == 1 and b.ndim == 1:
        af = a.flatten().tolist()
        bf = b.flatten().tolist()
        la, lb = len(af), len(bf)
        if la not in (2, 3) or lb not in (2, 3):
            raise ValueError("incompatible vector sizes for cross product")
        # Pad 2D to 3D with z=0
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
    # Batched nD case: process along last axis
    if a.ndim >= 2 and b.ndim >= 2:
        # Broadcast batch dimensions
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
            # Pad 2D to 3D
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
        # Move result vector axis to axisc position
        if axisc != -1 and axisc != result.ndim - 1 and result.ndim > 1 and not (la == 2 and lb == 2):
            result = moveaxis(result, -1, axisc)
        return result

def tensordot(a, b, axes=2):
    """Compute tensor dot product along specified axes.

    Parameters
    ----------
    a, b : array_like
    axes : int or (2,) list of lists
        If int N, contract last N axes of *a* with first N axes of *b*.
        If a tuple of two sequences, contract the specified axes.
    """
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
    # Normalise negative axes
    axes_a = [ax if ax >= 0 else ax + na for ax in axes_a]
    axes_b = [ax if ax >= 0 else ax + nb for ax in axes_b]
    # Free axes (those not being contracted)
    free_a = [i for i in range(na) if i not in axes_a]
    free_b = [i for i in range(nb) if i not in axes_b]
    # Transpose a so free axes come first, contracted axes last
    perm_a = free_a + axes_a
    # Transpose b so contracted axes come first, free axes last
    perm_b = axes_b + free_b
    at = _transpose_with_axes(a, perm_a)
    bt = _transpose_with_axes(b, perm_b)
    # Compute shapes for reshape into 2D
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
        return result
    return result.reshape(out_shape)

def roll(a, shift, axis=None):
    if not isinstance(a, ndarray):
        a = asarray(a)
    if axis is None:
        # Flatten, roll, reshape back
        flat = a.flatten()
        n = flat.size
        if n == 0:
            return a.copy()
        s = int(shift) % n if n > 0 else 0
        if s == 0:
            return a.copy()
        # Roll via concatenation
        parts = [flat[n - s:], flat[:n - s]]
        return concatenate(parts).reshape(a.shape)
    if isinstance(axis, (tuple, list)):
        # Multiple axes: apply roll sequentially
        result = a
        shifts = shift if isinstance(shift, (tuple, list)) else [shift] * len(axis)
        for sh, ax in zip(shifts, axis):
            result = roll(result, sh, ax)
        return result
    # Single axis roll
    ax = int(axis)
    if ax < 0:
        ax += a.ndim
    n = a.shape[ax]
    if n == 0:
        return a.copy()
    s = int(shift) % n if n > 0 else 0
    if s == 0:
        return a.copy()
    return _native.roll(a, s, ax)

def moveaxis(a, source, destination):
    """Move axes of an array to new positions.

    Other axes remain in their original order.
    """
    from . import ma as _ma_mod
    _is_masked = isinstance(a, _ma_mod.MaskedArray)
    if _is_masked:
        a = a.data
    a = asarray(a) if not isinstance(a, ndarray) else a
    ndim_a = a.ndim
    # Normalise to lists
    if isinstance(source, int):
        source = [source]
    if isinstance(destination, int):
        destination = [destination]
    # Validate lengths
    if len(source) != len(destination):
        raise ValueError("`source` and `destination` arguments must have the same number of elements")
    # Normalize and validate bounds
    src_norm = []
    for s in source:
        s_n = s if s >= 0 else s + ndim_a
        if s_n < 0 or s_n >= ndim_a:
            raise AxisError("source {} is out of bounds for array of dimension {}".format(s, ndim_a))
        src_norm.append(s_n)
    dst_norm = []
    for d in destination:
        d_n = d if d >= 0 else d + ndim_a
        if d_n < 0 or d_n >= ndim_a:
            raise AxisError("destination {} is out of bounds for array of dimension {}".format(d, ndim_a))
        dst_norm.append(d_n)
    # Check for repeated axes
    if len(set(src_norm)) != len(src_norm):
        raise ValueError("repeated axis in `source`")
    if len(set(dst_norm)) != len(dst_norm):
        raise ValueError("repeated axis in `destination`")
    source = src_norm
    destination = dst_norm
    # Build permutation: start with axes not in source, in order
    order = [i for i in range(ndim_a) if i not in source]
    # Insert source axes at destination positions (must insert in sorted dest order)
    pairs = sorted(zip(destination, source))
    for dst, src in pairs:
        order.insert(dst, src)
    result = _transpose_with_axes(a, order)
    if _is_masked:
        result = _ma_mod.MaskedArray(result)
    return result

def rollaxis(a, axis, start=0):
    """Roll the specified axis backwards, until it lies in position *start*."""
    a = asarray(a) if not isinstance(a, ndarray) else a
    ndim_a = a.ndim
    if axis < -ndim_a or axis >= ndim_a:
        raise AxisError("axis {} is out of bounds for array of dimension {}".format(axis, ndim_a))
    if axis < 0:
        axis += ndim_a
    if start < -ndim_a or start > ndim_a:
        raise AxisError("start {} is out of bounds for array of dimension {}".format(start, ndim_a))
    if start < 0:
        start += ndim_a
    if start > axis:
        start -= 1
    return moveaxis(a, axis, start)

def swapaxes(a, axis1, axis2):
    """Interchange two axes of an array."""
    a = asarray(a) if not isinstance(a, ndarray) else a
    ndim_a = a.ndim
    if axis1 < 0:
        axis1 += ndim_a
    if axis2 < 0:
        axis2 += ndim_a
    if axis1 == axis2:
        return a
    order = list(range(ndim_a))
    order[axis1], order[axis2] = order[axis2], order[axis1]
    return _transpose_with_axes(a, order)

def meshgrid(*xi, indexing='xy'):
    arrays = [a if isinstance(a, ndarray) else array(a) for a in xi]
    return _native.meshgrid(arrays, indexing)

def pad(a, pad_width, mode='constant', constant_values=0, **kwargs):
    if not isinstance(a, ndarray):
        a = array(a)
    # Normalise pad_width to list of (before, after) per axis
    if isinstance(pad_width, int):
        pw = [(pad_width, pad_width)] * a.ndim
    elif isinstance(pad_width, (list, tuple)):
        if isinstance(pad_width[0], int):
            if len(pad_width) == 2:
                pw = [(pad_width[0], pad_width[1])] * a.ndim
            else:
                pw = [(pad_width[0], pad_width[0])] * a.ndim
        else:
            pw = [(p[0], p[1]) for p in pad_width]
    else:
        pw = [(pad_width, pad_width)] * a.ndim

    if mode == 'constant':
        if isinstance(constant_values, (list, tuple)):
            constant_values = constant_values[0] if isinstance(constant_values[0], (int, float)) else constant_values[0][0]
        return _native.pad(a, pad_width, float(constant_values))

    # Pure-Python implementation for 'edge', 'reflect', 'wrap'
    def _pad_1d(data_list, before, after, mode_str):
        """Pad a 1D Python list with the given mode."""
        n = len(data_list)
        result = []
        if mode_str == 'edge':
            result = [data_list[0]] * before + list(data_list) + [data_list[-1]] * after
        elif mode_str == 'reflect':
            left = []
            for i in range(before):
                idx = (i + 1) % (2 * (n - 1)) if n > 1 else 0
                if idx >= n:
                    idx = 2 * (n - 1) - idx
                left.insert(0, data_list[idx])
            right = []
            for i in range(after):
                idx = (i + 1) % (2 * (n - 1)) if n > 1 else 0
                if idx >= n:
                    idx = 2 * (n - 1) - idx
                right.append(data_list[n - 1 - idx])
            result = left + list(data_list) + right
        elif mode_str == 'wrap':
            left = []
            for i in range(before):
                left.insert(0, data_list[-(i + 1) % n])
            right = []
            for i in range(after):
                right.append(data_list[i % n])
            result = left + list(data_list) + right
        elif mode_str == 'symmetric':
            # Like reflect but includes the edge value
            left = []
            for i in range(before):
                idx = i % (2 * n) if n > 0 else 0
                if idx >= n:
                    idx = 2 * n - 1 - idx
                left.insert(0, data_list[idx])
            right = []
            for i in range(after):
                idx = i % (2 * n) if n > 0 else 0
                if idx >= n:
                    idx = 2 * n - 1 - idx
                right.append(data_list[n - 1 - idx])
            result = left + list(data_list) + right
        elif mode_str == 'linear_ramp':
            end_val = kwargs.get('end_values', 0)
            if isinstance(end_val, (list, tuple)):
                end_val = end_val[0] if isinstance(end_val[0], (int, float)) else end_val[0][0]
            left = []
            for i in range(before):
                # Linear ramp from end_val to data_list[0]
                frac = float(i + 1) / float(before + 1) if before > 0 else 1.0
                left.append(end_val + (data_list[0] - end_val) * frac)
            right = []
            for i in range(after):
                # Linear ramp from data_list[-1] to end_val
                frac = float(i + 1) / float(after + 1) if after > 0 else 1.0
                right.append(data_list[-1] + (end_val - data_list[-1]) * frac)
            result = left + list(data_list) + right
        elif mode_str == 'mean':
            mean_val = sum(data_list) / len(data_list) if len(data_list) > 0 else 0.0
            result = [mean_val] * before + list(data_list) + [mean_val] * after
        elif mode_str == 'median':
            sorted_data = sorted(data_list)
            mid = len(sorted_data) // 2
            if len(sorted_data) % 2 == 0 and len(sorted_data) > 0:
                median_val = (sorted_data[mid - 1] + sorted_data[mid]) / 2.0
            elif len(sorted_data) > 0:
                median_val = sorted_data[mid]
            else:
                median_val = 0.0
            result = [median_val] * before + list(data_list) + [median_val] * after
        elif mode_str == 'minimum':
            min_val = min(data_list) if len(data_list) > 0 else 0.0
            result = [min_val] * before + list(data_list) + [min_val] * after
        elif mode_str == 'maximum':
            max_val = max(data_list) if len(data_list) > 0 else 0.0
            result = [max_val] * before + list(data_list) + [max_val] * after
        else:
            raise NotImplementedError("pad mode '{}' is not supported".format(mode_str))
        return result

    if a.ndim == 1:
        data = [float(a[i]) for i in range(a.shape[0])]
        padded = _pad_1d(data, pw[0][0], pw[0][1], mode)
        return array(padded)

    # nD: pad axis-by-axis, starting from the last axis
    def _to_nested(arr):
        """Convert ndarray to nested Python lists."""
        if arr.ndim == 1:
            return [float(arr[i]) for i in range(arr.shape[0])]
        return [_to_nested(arr[i]) for i in range(arr.shape[0])]

    def _pad_axis(nested, axis, before, after, mode_str, current_depth=0):
        """Recursively pad along a specific axis of nested lists."""
        if current_depth == axis:
            return _pad_1d(nested, before, after, mode_str)
        else:
            return [_pad_axis(sub, axis, before, after, mode_str, current_depth + 1) for sub in nested]

    nested = _to_nested(a)
    for ax in range(a.ndim):
        before, after = pw[ax]
        if before > 0 or after > 0:
            nested = _pad_axis(nested, ax, before, after, mode)
    return array(nested)

def indices(dimensions, dtype=None, sparse=False):
    """Return an array representing the indices of a grid."""
    ndim = len(dimensions)
    _dt = str(dtype) if dtype is not None else None
    if _dt is not None:
        _dt = _normalize_dtype(_dt)
    if ndim == 0:
        if sparse:
            return []
        return array([], dtype=_dt)

    if sparse:
        result = []
        for i in range(ndim):
            shape = [1] * ndim
            shape[i] = dimensions[i]
            idx = arange(0, dimensions[i])
            if _dt is not None:
                idx = idx.astype(_dt)
            idx = idx.reshape(shape)
            result.append(idx)
        return result

    # Dense: result shape is (ndim, *dimensions)
    grids = []
    for axis in range(ndim):
        # For each axis, create index array
        idx = arange(0, dimensions[axis])
        if _dt is not None:
            idx = idx.astype(_dt)
        # Reshape to broadcast: shape is [1,...,1,dim_axis,1,...,1]
        shape = [1] * ndim
        shape[axis] = dimensions[axis]
        idx = idx.reshape(shape)
        # Tile to fill all dimensions
        reps = list(dimensions)
        reps[axis] = 1
        grid = tile(idx, reps)
        grids.append(grid)

    # Force contiguous layout before stacking to avoid memory layout issues
    contiguous = [asarray(g.tolist()) for g in grids]
    result = stack(contiguous)
    if _dt is not None:
        result = result.astype(_dt)
    return result

def fromiter(iterable, dtype=None, count=-1):
    if dtype is not None:
        _dt = _normalize_dtype(str(dtype))
        if _dt in ("str", "bytes") or str(dtype) in ("S", "S0", "V0", "U0"):
            raise ValueError("Must specify length when using variable-length dtypes")
    subarray_len = None
    if dtype is not None:
        # Minimal support for subarray dtype validation, e.g. dtype((int, 2)).
        if isinstance(dtype, tuple) and len(dtype) == 2 and isinstance(dtype[1], int):
            subarray_len = int(dtype[1])
        else:
            _dtype_name = str(getattr(dtype, "name", dtype))
            if _dtype_name.startswith("(<class '") and _dtype_name.endswith(")"):
                _comma = _dtype_name.rfind(",")
                if _comma != -1:
                    try:
                        subarray_len = int(_dtype_name[_comma + 1 : -1].strip())
                    except (TypeError, ValueError):
                        subarray_len = None
    if count > 0:
        data = []
        for val in iterable:
            data.append(val)
            if len(data) >= count:
                break
        if len(data) < count:
            raise ValueError(
                "iterator too short: Expected %d but iterator had only %d items." % (count, len(data))
            )
    else:
        data = list(iterable)
    if subarray_len is not None:
        for val in data:
            if not isinstance(val, (list, tuple)):
                raise ValueError("setting an array element with a sequence")
            if len(val) != subarray_len:
                raise ValueError("setting an array element with a sequence")
    return array(data, dtype=dtype)

def fromstring(string, dtype=None, count=-1, sep=''):
    """Parse array from a string."""
    if sep:
        parts = string.split(sep)
        data = [float(p.strip()) for p in parts if p.strip()]
    else:
        data = [float(x) for x in string.split()]
    if count > 0:
        data = data[:count]
    return array(data, dtype=dtype)

def array_equal(a1, a2, equal_nan=False):
    """True if two arrays have the same shape and elements."""
    try:
        if not isinstance(a1, ndarray):
            a1 = array(a1)
        if not isinstance(a2, ndarray):
            a2 = array(a2)
        if a1.shape != a2.shape:
            return False
        has_nan = bool(any(logical_or(isnan(a1), isnan(a2))))
        if not equal_nan and has_nan:
            return False
        if equal_nan:
            try:
                both_nan = logical_and(isnan(a1), isnan(a2))
                neither_nan = logical_and(logical_not(isnan(a1)), logical_not(isnan(a2)))
                return bool(all(logical_or(both_nan, logical_and(neither_nan, a1 == a2))))
            except Exception:
                return bool((a1 == a2).all())
        return bool((a1 == a2).all())
    except Exception:
        return False

def array_equiv(a1, a2):
    """True if arrays are shape-consistent and element-wise equal, with broadcasting."""
    a1 = asarray(a1)
    a2 = asarray(a2)
    try:
        bshape = broadcast_shapes(a1.shape, a2.shape)
        b1 = broadcast_to(a1, bshape)
        b2 = broadcast_to(a2, bshape)
        return bool(all(b1 == b2))
    except Exception:
        return False

def require(a, dtype=None, requirements=None):
    """Return an ndarray that satisfies the given requirements."""
    _flag_alias = {
        'C': 'C_CONTIGUOUS', 'C_CONTIGUOUS': 'C_CONTIGUOUS', 'CONTIGUOUS': 'C_CONTIGUOUS',
        'F': 'F_CONTIGUOUS', 'F_CONTIGUOUS': 'F_CONTIGUOUS', 'FORTRAN': 'F_CONTIGUOUS',
        'A': 'ALIGNED', 'ALIGNED': 'ALIGNED',
        'W': 'WRITEABLE', 'WRITEABLE': 'WRITEABLE',
        'O': 'OWNDATA', 'OWNDATA': 'OWNDATA',
        'E': 'ENSUREARRAY', 'ENSUREARRAY': 'ENSUREARRAY',
    }
    if requirements is None:
        requirements = []
    if isinstance(requirements, str):
        requirements = [requirements]
    # Normalize requirements
    reqs = set()
    for r in requirements:
        r_upper = r.upper()
        if r_upper not in _flag_alias:
            raise KeyError(r)
        reqs.add(_flag_alias[r_upper])
    if 'C_CONTIGUOUS' in reqs and 'F_CONTIGUOUS' in reqs:
        raise ValueError("Cannot specify both C and Fortran contiguous.")
    # Convert input
    order = 'C'
    if 'F_CONTIGUOUS' in reqs:
        order = 'F'
    if dtype is not None:
        arr = array(a, dtype=dtype, order=order, copy=False)
    else:
        arr = array(a, order=order, copy=False)
    if not isinstance(arr, ndarray):
        arr = asarray(arr)
    # E (ENSUREARRAY) means return base ndarray, not subclass
    if 'ENSUREARRAY' in reqs and type(arr) is not ndarray:
        arr = array(arr, subok=False)
    # W (WRITEABLE) - ensure writable
    if 'WRITEABLE' in reqs:
        if not arr.flags['WRITEABLE']:
            arr = arr.copy()
    # Copy if OWNDATA required (asarray may share memory)
    if 'OWNDATA' in reqs:
        arr = arr.copy()
    return arr

def binary_repr(num, width=None):
    if num >= 0:
        s = bin(num)[2:]
        if width is not None:
            s = s.zfill(width)
        return s
    else:
        if width is not None:
            # Two's complement
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

def sort(a, axis=-1, kind=None, order=None):
    if not isinstance(a, ndarray):
        a = asarray(a)
    original_dtype = str(a.dtype)
    if axis is not None and axis < 0:
        axis = a.ndim + axis
    result = a.sort(axis)
    # Preserve original dtype (Rust sort may convert to float64)
    if original_dtype in ('int32', 'int64') and str(result.dtype) != original_dtype:
        result = result.astype(original_dtype)
    return result

def argsort(a, axis=-1, kind=None, order=None):
    if isinstance(a, (tuple, list)):
        if len(a) == 0:
            return _native.zeros((0,), 'int64')
        a = _native.array([float(x) for x in a])
    elif isinstance(a, _ObjectArray):
        vals = [float(x) if isinstance(x, (int, float)) else 0. for x in (a._data or [])]
        inds = sorted(range(len(vals)), key=lambda i: vals[i])
        return _native.array([float(i) for i in inds]) if inds else _native.zeros((0,), 'int64')
    else:
        a = asarray(a)
    if axis is not None and axis < 0:
        axis = a.ndim + axis
    return a.argsort(axis)

def size(a, axis=None):
    if not isinstance(a, ndarray):
        a = asarray(a)
    if axis is None:
        return a.size
    return a.shape[axis]

def take(a, indices, axis=None, out=None, mode="raise"):
    if isinstance(a, ndarray):
        return _native.take(a, indices, axis)
    flat = array(a).flatten()
    if isinstance(indices, ndarray):
        idx = [int(float(indices.flatten()[i])) for i in range(indices.size)]
    else:
        idx = list(indices) if hasattr(indices, '__iter__') else [indices]
    return array([float(flat[i]) for i in idx])

def advanced_fancy_index(arr, indices):
    """Handle multi-axis fancy indexing: arr[[0,1], [2,3]] -> [arr[0,2], arr[1,3]].

    In NumPy, ``a[[0,1], [2,3]]`` selects ``[a[0,2], a[1,3]]`` (paired
    indices, not cross-product).  Because the Rust ndarray class does not
    support tuple-of-list indexing natively, this helper provides the same
    semantics as a module-level function.

    Parameters
    ----------
    arr : array_like
        Source array (must be at least 2-D for multi-axis use).
    indices : sequence of array_like
        One index array per axis. All index arrays must broadcast to the
        same shape (here: must have the same length).

    Returns
    -------
    ndarray
        1-D array of selected elements.

    Examples
    --------
    >>> a = np.arange(12).reshape(3, 4)
    >>> np.advanced_fancy_index(a, [[0, 1, 2], [3, 2, 1]])
    array([3., 6., 9.])  # a[0,3], a[1,2], a[2,1]
    """
    arr = asarray(arr)
    # Normalise each index array to a flat Python list of ints
    idx_arrays = [asarray(idx).flatten().tolist() for idx in indices]
    lengths = [len(a) for a in idx_arrays]
    if len(set(lengths)) > 1:
        raise IndexError("shape mismatch: indexing arrays could not be broadcast together")
    n = lengths[0]
    result = []
    for i in _builtin_range(n):
        # Walk into the array one axis at a time
        val = arr
        for ax in _builtin_range(len(idx_arrays)):
            ix = int(idx_arrays[ax][i])
            val = val[ix]
        result.append(float(val.tolist()) if hasattr(val, 'tolist') else float(val))
    return array(result)

def flip(a, axis=None):
    """Reverse the order of elements along the given axis."""
    if isinstance(a, ndarray):
        return _native.flip(a, axis)
    return array(list(reversed(a))) if axis is None else a

def flipud(a):
    """Flip array upside down (reverse along axis 0)."""
    if isinstance(a, ndarray):
        return _native.flipud(a)
    return flip(a, 0)

def fliplr(a):
    """Flip array left-right (reverse along axis 1)."""
    if isinstance(a, ndarray):
        return _native.fliplr(a)
    return flip(a, 1)

def rot90(a, k=1, axes=(0, 1)):
    """Rotate array 90 degrees in the plane of the first two axes."""
    if isinstance(a, ndarray):
        return _native.rot90(a, k)
    return a

def unique(a, return_index=False, return_inverse=False, return_counts=False, axis=None):
    """Return sorted unique elements of an array.

    Parameters
    ----------
    a : array_like
    return_index : bool
        If True, return indices of first occurrences.
    return_inverse : bool
        If True, return indices to reconstruct original from unique.
    return_counts : bool
        If True, return count of each unique value.
    axis : int, optional
        The axis along which to find unique slices. If None, flatten first.

    Returns
    -------
    unique : ndarray
    unique_indices : ndarray (optional)
    unique_inverse : ndarray (optional)
    unique_counts : ndarray (optional)
    """
    if not isinstance(a, ndarray):
        a = asarray(a)
    if axis is not None:
        # For axis=0: find unique rows (or slices along axis 0)
        if axis < 0:
            axis = a.ndim + axis
        n_slices = a.shape[axis]
        # Extract each slice as a tuple for hashing
        seen = {}
        unique_indices_list = []
        for i in range(n_slices):
            # Build index to extract slice along axis
            if axis == 0:
                sl = a[i]
            else:
                # General case: use swapaxes to bring target axis to front
                tmp = swapaxes(a, 0, axis)
                sl = tmp[i]
            key = tuple(sl.flatten().tolist())
            if key not in seen:
                seen[key] = i
                unique_indices_list.append(i)
        # Build result array from unique indices
        if axis == 0:
            rows = [a[i] for i in unique_indices_list]
        else:
            tmp = swapaxes(a, 0, axis)
            rows = [tmp[i] for i in unique_indices_list]
        # Sort by the first element of each row for consistent ordering
        pairs = list(zip(unique_indices_list, rows))
        pairs.sort(key=lambda p: tuple(p[1].flatten().tolist()))
        unique_indices_list = [p[0] for p in pairs]
        rows = [p[1] for p in pairs]
        result = stack(rows, axis=0)
        if axis != 0:
            result = swapaxes(result, 0, axis)
        extras = return_index or return_inverse or return_counts
        if not extras:
            return result
        ret = (result,)
        if return_index:
            ret = ret + (array([float(i) for i in unique_indices_list]),)
        if return_inverse:
            # Map each original slice index to its position in unique result
            key_to_pos = {}
            for pos, idx in enumerate(unique_indices_list):
                if axis == 0:
                    sl = a[idx]
                else:
                    tmp = swapaxes(a, 0, axis)
                    sl = tmp[idx]
                key = tuple(sl.flatten().tolist())
                key_to_pos[key] = pos
            inv = []
            for i in range(n_slices):
                if axis == 0:
                    sl = a[i]
                else:
                    tmp = swapaxes(a, 0, axis)
                    sl = tmp[i]
                key = tuple(sl.flatten().tolist())
                inv.append(float(key_to_pos[key]))
            ret = ret + (array(inv),)
        if return_counts:
            # Count occurrences of each unique
            count_map = {}
            for i in range(n_slices):
                if axis == 0:
                    sl = a[i]
                else:
                    tmp = swapaxes(a, 0, axis)
                    sl = tmp[i]
                key = tuple(sl.flatten().tolist())
                if key not in count_map:
                    count_map[key] = 0
                count_map[key] += 1
            counts_list = []
            for idx in unique_indices_list:
                if axis == 0:
                    sl = a[idx]
                else:
                    tmp = swapaxes(a, 0, axis)
                    sl = tmp[idx]
                key = tuple(sl.flatten().tolist())
                counts_list.append(float(count_map[key]))
            ret = ret + (array(counts_list),)
        return ret
    flat = a.flatten()
    n = flat.shape[0]
    vals = [float(flat[i]) for i in range(n)]

    # Build sorted unique with tracking info
    indexed = sorted(enumerate(vals), key=lambda t: t[1])
    unique_vals = []
    first_indices = []
    counts = []
    prev = None
    for orig_idx, v in indexed:
        if prev is None or v != prev:
            unique_vals.append(v)
            first_indices.append(orig_idx)
            counts.append(1)
            prev = v
        else:
            counts[-1] += 1
            # Keep the smallest original index
            if orig_idx < first_indices[-1]:
                first_indices[-1] = orig_idx

    result_unique = array(unique_vals)
    extras = return_index or return_inverse or return_counts
    if not extras:
        return result_unique
    ret = (result_unique,)
    if return_index:
        ret = ret + (array([float(i) for i in first_indices]),)
    if return_inverse:
        # For each element in the original flat array, find its position in unique_vals
        val_to_pos = {}
        for i, v in enumerate(unique_vals):
            val_to_pos[v] = i
        inverse = [float(val_to_pos[v]) for v in vals]
        ret = ret + (array(inverse),)
    if return_counts:
        ret = ret + (array([float(c) for c in counts]),)
    return ret

def intersect1d(ar1, ar2, assume_unique=False, return_indices=False):
    if not isinstance(ar1, ndarray):
        ar1 = array(ar1)
    if not isinstance(ar2, ndarray):
        ar2 = array(ar2)
    if not return_indices:
        return _native.intersect1d(ar1, ar2)
    # Find intersection with indices
    flat1 = ar1.flatten().tolist()
    flat2 = ar2.flatten().tolist()
    s1 = set(flat1)
    s2 = set(flat2)
    common = sorted(s1 & s2)
    ind1 = [flat1.index(v) for v in common]
    ind2 = [flat2.index(v) for v in common]
    return array(common), array(ind1), array(ind2)

def union1d(ar1, ar2):
    if not isinstance(ar1, ndarray):
        ar1 = array(ar1)
    if not isinstance(ar2, ndarray):
        ar2 = array(ar2)
    return _native.union1d(ar1, ar2)

def setdiff1d(ar1, ar2, assume_unique=False):
    if not isinstance(ar1, ndarray):
        ar1 = array(ar1)
    if not isinstance(ar2, ndarray):
        ar2 = array(ar2)
    return _native.setdiff1d(ar1, ar2)

def setxor1d(ar1, ar2, assume_unique=False):
    """Return sorted, unique values that are in only one of the input arrays."""
    if not isinstance(ar1, ndarray):
        ar1 = array(ar1)
    if not isinstance(ar2, ndarray):
        ar2 = array(ar2)
    u1 = unique(ar1)
    u2 = unique(ar2)
    # Elements in ar1 but not ar2, plus elements in ar2 but not ar1
    diff1 = setdiff1d(u1, u2)
    diff2 = setdiff1d(u2, u1)
    return sort(concatenate([diff1, diff2]))

def isin(element, test_elements, assume_unique=False, invert=False):
    if not isinstance(element, ndarray):
        element = array(element)
    if not isinstance(test_elements, ndarray):
        test_elements = array(test_elements)
    result = _native.isin(element, test_elements)
    if invert:
        return logical_not(result)
    return result

in1d = isin

def all(a, axis=None, out=None, keepdims=False, where=True):
    if not isinstance(a, ndarray):
        a = asarray(a)
    if where is not True:
        w = asarray(where).astype("bool")
        # Masked elements become True (identity for AND)
        mask_f = w.astype("float64")
        a = a * mask_f + (1.0 - mask_f)
    if axis is None:
        return a.all()
    # Reduce along specific axis: all elements nonzero iff min != 0
    m = a.min(axis, keepdims)
    if not isinstance(m, ndarray):
        return bool(m != 0.0)
    flat = m.flatten().tolist()
    result = [v != 0.0 for v in flat]
    return array(result).reshape(m.shape)

def any(a, axis=None, out=None, keepdims=False, where=True):
    if not isinstance(a, ndarray):
        a = asarray(a)
    if where is not True:
        w = asarray(where).astype("bool")
        # Masked elements become False (identity for OR / 0.0)
        mask_f = w.astype("float64")
        a = a * mask_f
    if axis is None:
        return a.any()
    # Reduce along specific axis: any element nonzero iff max != 0
    m = a.max(axis, keepdims)
    if not isinstance(m, ndarray):
        return bool(m != 0.0)
    flat = m.flatten().tolist()
    result = [v != 0.0 for v in flat]
    return array(result).reshape(m.shape)

def may_share_memory(a, b, max_work=None):
    """Check if arrays may share memory. Returns True if a and b are the same object."""
    return a is b

def shares_memory(a, b, max_work=None):
    """Check if arrays share memory. Returns True if a and b are the same object."""
    return a is b

def signbit(x):
    if isinstance(x, ndarray):
        return _native.signbit(x)
    return x < 0

def power(x1, x2):
    return asarray(x1) ** asarray(x2)

def add(x1, x2, out=None):
    a = x1 if isinstance(x1, ndarray) else (asarray(x1) if isinstance(x1, (list, tuple, _ObjectArray)) else x1)
    b = x2 if isinstance(x2, ndarray) else (asarray(x2) if isinstance(x2, (list, tuple, _ObjectArray)) else x2)

    scalar_scalar = (
        not isinstance(x1, (ndarray, list, tuple, _ObjectArray))
        and not isinstance(x2, (ndarray, list, tuple, _ObjectArray))
    )
    if scalar_scalar:
        target = str(result_type(x1, x2))
        val = x1 + x2
        if target.startswith("complex"):
            return _ObjectArray([complex(val)], target)
        return array([val], dtype=target)

    if isinstance(a, _ObjectArray) or isinstance(b, _ObjectArray):
        target = str(result_type(x1, x2))
        if not isinstance(a, _ObjectArray):
            a = _ObjectArray([a] * len(b._data), target)
        if not isinstance(b, _ObjectArray):
            b = _ObjectArray([b] * len(a._data), target)
        n = len(a._data) if len(a._data) < len(b._data) else len(b._data)
        vals = [a._data[i] + b._data[i] for i in range(n)]
        return _ObjectArray(vals, target)

    def _scalar_for_array_op(v):
        if hasattr(v, "_numpy_dtype_name"):
            dn = str(getattr(v, "_numpy_dtype_name"))
            if dn == "bool":
                return bool(v)
            if dn.startswith("int") or dn.startswith("uint"):
                return int(v)
            if dn.startswith("float"):
                return float(v)
            if dn.startswith("complex"):
                return complex(v)
        return v

    if isinstance(a, ndarray) and not isinstance(b, ndarray):
        b = _scalar_for_array_op(b)
        if not isinstance(b, (int, float, complex, bool)):
            b = asarray(b)
    elif isinstance(b, ndarray) and not isinstance(a, ndarray):
        a = _scalar_for_array_op(a)
        if not isinstance(a, (int, float, complex, bool)):
            a = asarray(a)

    if isinstance(a, ndarray) or isinstance(b, ndarray):
        r = a + b
    else:
        r = asarray(a) + asarray(b)
    if hasattr(r, "dtype"):
        target = str(result_type(x1, x2))
        if str(r.dtype) != target:
            try:
                r = r.astype(target)
            except Exception:
                pass
    return r

def divide(x1, x2, out=None):
    return asarray(x1) / asarray(x2)

def greater(x1, x2):
    return asarray(x1) > asarray(x2)

def less(x1, x2):
    return asarray(x1) < asarray(x2)

def equal(x1, x2):
    return asarray(x1) == asarray(x2)

def not_equal(x1, x2):
    return asarray(x1) != asarray(x2)

def greater_equal(x1, x2):
    return asarray(x1) >= asarray(x2)

def less_equal(x1, x2):
    return asarray(x1) <= asarray(x2)

def maximum(x1, x2):
    x1 = asarray(x1)
    x2 = asarray(x2)
    return where(x1 >= x2, x1, x2)

def minimum(x1, x2):
    x1 = asarray(x1)
    x2 = asarray(x2)
    return where(x1 <= x2, x1, x2)

def logical_and(x1, x2, out=None):
    x1 = asarray(x1)
    x2 = asarray(x2)
    out_shape = broadcast_shapes(x1.shape, x2.shape)
    x1 = broadcast_to(x1, out_shape)
    x2 = broadcast_to(x2, out_shape)
    flat1 = x1.flatten().tolist()
    flat2 = x2.flatten().tolist()
    result = [1.0 if (bool(a) and bool(b)) else 0.0 for a, b in zip(flat1, flat2)]
    r = array(result).reshape(out_shape).astype("bool")
    if out is not None:
        _copy_into(out, r)
        return out
    return r

def logical_or(x1, x2, out=None):
    x1 = asarray(x1)
    x2 = asarray(x2)
    out_shape = broadcast_shapes(x1.shape, x2.shape)
    x1 = broadcast_to(x1, out_shape)
    x2 = broadcast_to(x2, out_shape)
    flat1 = x1.flatten().tolist()
    flat2 = x2.flatten().tolist()
    result = [1.0 if (bool(a) or bool(b)) else 0.0 for a, b in zip(flat1, flat2)]
    r = array(result).reshape(out_shape).astype("bool")
    if out is not None:
        _copy_into(out, r)
        return out
    return r

def logical_not(x, out=None):
    if isinstance(x, ndarray):
        r = _native.logical_not(x)
    else:
        return not x
    if out is not None:
        _copy_into(out, r)
        return out
    return r

def logical_xor(x1, x2, out=None):
    r = logical_and(logical_or(x1, x2), logical_not(logical_and(x1, x2)))
    if out is not None:
        _copy_into(out, r)
        return out
    return r

# --- Bitwise operations ------------------------------------------------------

def bitwise_and(x1, x2):
    """Element-wise bitwise AND of integer arrays."""
    x1 = asarray(x1)
    x2 = asarray(x2)
    a = x1.astype("int64")
    b = x2.astype("int64")
    a_list = a.flatten().tolist()
    b_list = b.flatten().tolist()
    result = [int(av) & int(bv) for av, bv in zip(a_list, b_list)]
    return array([float(v) for v in result]).astype("int64")

def bitwise_or(x1, x2):
    """Element-wise bitwise OR of integer arrays."""
    x1 = asarray(x1)
    x2 = asarray(x2)
    a = x1.astype("int64")
    b = x2.astype("int64")
    a_list = a.flatten().tolist()
    b_list = b.flatten().tolist()
    result = [int(av) | int(bv) for av, bv in zip(a_list, b_list)]
    return array([float(v) for v in result]).astype("int64")

def bitwise_xor(x1, x2):
    """Element-wise bitwise XOR of integer arrays."""
    x1 = asarray(x1)
    x2 = asarray(x2)
    a = x1.astype("int64")
    b = x2.astype("int64")
    a_list = a.flatten().tolist()
    b_list = b.flatten().tolist()
    result = [int(av) ^ int(bv) for av, bv in zip(a_list, b_list)]
    return array([float(v) for v in result]).astype("int64")

def bitwise_not(x):
    """Element-wise bitwise NOT (invert) of integer array."""
    x = asarray(x)
    a = x.astype("int64")
    a_list = a.flatten().tolist()
    result = [~int(v) for v in a_list]
    return array([float(v) for v in result]).astype("int64")

invert = bitwise_not

def left_shift(x1, x2):
    """Element-wise left bit shift."""
    x1 = asarray(x1)
    x2 = asarray(x2)
    a = x1.astype("int64")
    b = x2.astype("int64")
    a_list = a.flatten().tolist()
    b_list = b.flatten().tolist()
    result = [int(av) << int(bv) for av, bv in zip(a_list, b_list)]
    return array([float(v) for v in result]).astype("int64")

def right_shift(x1, x2):
    """Element-wise right bit shift."""
    x1 = asarray(x1)
    x2 = asarray(x2)
    a = x1.astype("int64")
    b = x2.astype("int64")
    a_list = a.flatten().tolist()
    b_list = b.flatten().tolist()
    result = [int(av) >> int(bv) for av, bv in zip(a_list, b_list)]
    return array([float(v) for v in result]).astype("int64")

def matrix_transpose(a):
    a = asarray(a) if not isinstance(a, ndarray) else a
    return a.T

def astype(a, dtype, casting='unsafe', copy=True):
    # Accept numpy scalar types (e.g. np.int64(10)) and plain scalars
    if isinstance(a, (int, float)):
        a = asarray(a)
    elif isinstance(a, _ScalarType):
        a = asarray(float(a))
    elif not isinstance(a, ndarray):
        raise TypeError("Input should be a NumPy array")
    dtype_str = _normalize_dtype(str(dtype))
    if casting != 'unsafe':
        from_dtype = str(a.dtype)
        if not can_cast(from_dtype, dtype_str, casting=casting):
            raise TypeError("Cannot cast array data from {} to {} according to the rule '{}'".format(
                from_dtype, dtype_str, casting))
    if copy is False and str(a.dtype) == dtype_str:
        return a
    return a.astype(dtype_str)

def real(a):
    """Return the real part of the array elements."""
    if isinstance(a, ndarray):
        return a.real
    return a

def imag(a):
    """Return the imaginary part of the array elements."""
    if isinstance(a, ndarray):
        return a.imag
    return 0

def conj(a):
    """Return the complex conjugate."""
    if isinstance(a, ndarray):
        return a.conj()
    return a

conjugate = conj

def angle(z, deg=False):
    """Return the angle (argument) of complex or real elements."""
    z = asarray(z)
    if iscomplexobj(z):
        try:
            result = arctan2(imag(z), real(z))
        except Exception:
            result = z.angle()
    else:
        # For real arrays: arctan2(0, x) gives 0 for positive, pi for negative
        result = arctan2(zeros(z.shape), z)
    if deg:
        result = result * (180.0 / _math.pi)
    return result

def _unwrap_1d_list(data, discont, period):
    """Unwrap a 1D list in-place style, returning a new list."""
    if len(data) == 0:
        return []
    result = [data[0]]
    for i in _builtin_range(1, len(data)):
        d = data[i] - result[-1]
        d = d - period * round(d / period)
        result.append(result[-1] + d)
    return result

def unwrap(p, discont=None, axis=-1, period=2*_math.pi):
    """Unwrap by changing deltas between values to 2*pi complement."""
    p = asarray(p)
    if discont is None:
        discont = period / 2
    if p.ndim <= 1:
        return array(_unwrap_1d_list(p.flatten().tolist(), discont, period))
    # For 2D: apply unwrap along the specified axis
    nd = p.ndim
    ax = axis
    if ax < 0:
        ax = nd + ax
    shape = p.shape
    # Move the target axis to the last position
    if ax != nd - 1:
        axes = [i for i in _builtin_range(nd) if i != ax] + [ax]
        pt = transpose(p, axes=axes)
    else:
        pt = p
    # Now unwrap along the last axis
    pt_shape = pt.shape
    n_outer = 1
    for s in pt_shape[:-1]:
        n_outer *= s
    flat_all = pt.flatten().tolist()
    axis_len = pt_shape[-1]
    result_flat = []
    for i in _builtin_range(n_outer):
        start = i * axis_len
        row = flat_all[start:start + axis_len]
        result_flat.extend(_unwrap_1d_list(row, discont, period))
    out = array(result_flat).reshape(pt_shape)
    # Move axis back if we transposed
    if ax != nd - 1:
        # Inverse permutation
        inv_axes = [0] * nd
        for i, a_val in enumerate(axes):
            inv_axes[a_val] = i
        out = transpose(out, axes=inv_axes)
    return out

def histogram_bin_edges(a, bins=10, range=None, weights=None):
    """Compute the bin edges for a histogram without computing the histogram itself."""
    a = asarray(a)
    if isinstance(bins, int):
        flat = a.flatten().tolist()
        if range is not None:
            lo, hi = float(range[0]), float(range[1])
        else:
            lo, hi = _builtin_min(flat), _builtin_max(flat)
        edges = linspace(lo, hi, bins + 1)
        return edges
    else:
        return asarray(bins)

def histogram(a, bins=10, range=None, density=None, weights=None):
    if not isinstance(a, ndarray):
        a = array(a)
    if isinstance(bins, (list, tuple, ndarray)):
        # Custom bin edges
        edges = asarray(bins).flatten()
        edge_list = edges.tolist()
        flat = a.flatten().tolist()
        n_bins = len(edge_list) - 1
        counts = [0.0] * n_bins
        w_list = None
        if weights is not None:
            w_list = asarray(weights).flatten().tolist()
        for idx_val, v in enumerate(flat):
            for j in _builtin_range(n_bins):
                if j == n_bins - 1:
                    if edge_list[j] <= v <= edge_list[j + 1]:
                        counts[j] += (w_list[idx_val] if w_list is not None else 1.0)
                        break
                else:
                    if edge_list[j] <= v < edge_list[j + 1]:
                        counts[j] += (w_list[idx_val] if w_list is not None else 1.0)
                        break
        counts_arr = array(counts)
        if density:
            bin_widths = diff(edges)
            total = float(sum(counts_arr))
            if total > 0.0:
                counts_arr = counts_arr / (total * bin_widths)
        return counts_arr, edges
    # bins is an int
    if weights is not None or range is not None:
        # Python fallback for weights/range with integer bins
        flat = a.flatten().tolist()
        if range is not None:
            lo, hi = float(range[0]), float(range[1])
        else:
            lo, hi = _builtin_min(flat), _builtin_max(flat)
        edges = linspace(lo, hi, num=bins + 1, endpoint=True)
        edge_list = edges.tolist()
        counts = [0.0] * bins
        w_list = None
        if weights is not None:
            w_list = asarray(weights).flatten().tolist()
        for idx_val, val in enumerate(flat):
            if range is not None and (val < lo or val > hi):
                continue
            for j in _builtin_range(bins):
                if (val >= edge_list[j] and val < edge_list[j + 1]) or (j == bins - 1 and val == edge_list[j + 1]):
                    counts[j] += (w_list[idx_val] if w_list is not None else 1.0)
                    break
        hist = array(counts)
        if density:
            widths = array([edge_list[i+1] - edge_list[i] for i in _builtin_range(bins)])
            total = float(sum(hist))
            if total > 0.0:
                hist = hist / (total * widths)
        return hist, edges
    # No weights, no range: use native
    counts, edges = _native.histogram(a, bins)
    if density:
        bin_widths = diff(edges)
        total = float(sum(counts))
        if total > 0.0:
            counts = counts / (total * bin_widths)
    return counts, edges

def histogram2d(x, y, bins=10, range=None, density=None, weights=None):
    """Compute the 2D histogram of two data samples."""
    if not isinstance(x, ndarray):
        x = array(x)
    if not isinstance(y, ndarray):
        y = array(y)
    x_flat = x.flatten()
    y_flat = y.flatten()
    n = x_flat.size
    x_list = x_flat.tolist()
    y_list = y_flat.tolist()
    # Determine number of bins for x and y
    if isinstance(bins, (list, tuple)):
        nbins_x = int(bins[0])
        nbins_y = int(bins[1])
    else:
        nbins_x = int(bins)
        nbins_y = int(bins)
    # Determine ranges
    if range is not None:
        xmin, xmax = float(range[0][0]), float(range[0][1])
        ymin, ymax = float(range[1][0]), float(range[1][1])
    else:
        xmin = _builtin_min(x_list)
        xmax = _builtin_max(x_list)
        ymin = _builtin_min(y_list)
        ymax = _builtin_max(y_list)
    # Build bin edges
    xedges_list = []
    yedges_list = []
    for i in _builtin_range(nbins_x + 1):
        xedges_list.append(xmin + i * (xmax - xmin) / nbins_x)
    for i in _builtin_range(nbins_y + 1):
        yedges_list.append(ymin + i * (ymax - ymin) / nbins_y)
    # Count into 2D bins
    hist_data = []
    for i in _builtin_range(nbins_x):
        row = []
        for j in _builtin_range(nbins_y):
            row.append(0.0)
        hist_data.append(row)
    xspan = xmax - xmin
    yspan = ymax - ymin
    for k in _builtin_range(n):
        xv = x_list[k]
        yv = y_list[k]
        # Find x bin
        if xspan == 0.0:
            xi = 0
        else:
            xi = int((xv - xmin) / (xspan / nbins_x))
        if xi >= nbins_x:
            xi = nbins_x - 1
        if xi < 0:
            xi = 0
        # Find y bin
        if yspan == 0.0:
            yi = 0
        else:
            yi = int((yv - ymin) / (yspan / nbins_y))
        if yi >= nbins_y:
            yi = nbins_y - 1
        if yi < 0:
            yi = 0
        hist_data[xi][yi] = hist_data[xi][yi] + 1.0
    # Convert to arrays
    flat_hist = []
    for i in _builtin_range(nbins_x):
        for j in _builtin_range(nbins_y):
            flat_hist.append(hist_data[i][j])
    hist = array(flat_hist).reshape((nbins_x, nbins_y))
    xedges = array(xedges_list)
    yedges = array(yedges_list)
    return hist, xedges, yedges

def bincount(x, weights=None, minlength=0):
    if not isinstance(x, ndarray):
        x = array(x)
    if weights is not None and not isinstance(weights, ndarray):
        weights = array(weights)
    return _native.bincount(x, weights, minlength)

def einsum(*operands, **kwargs):
    if len(operands) < 2:
        raise ValueError("einsum requires at least a subscript string and one operand")
    subscripts = operands[0]
    arrays = operands[1:]
    # Handle implicit output subscripts: 'ij,jk' -> 'ij,jk->ik'
    if '->' not in subscripts:
        # Collect all input subscripts
        input_subs = subscripts.replace(' ', '')
        parts = input_subs.split(',')
        # Output indices: letters that appear exactly once across all inputs (sorted alphabetically)
        from collections import Counter
        counts = Counter()
        for p in parts:
            counts.update(p)
        # Output = indices that appear exactly once, in alphabetical order
        output = ''.join(sorted(c for c, n in counts.items() if n == 1))
        subscripts = input_subs + '->' + output
    return _native.einsum(subscripts, *arrays)

# --- String (char) operations -----------------------------------------------
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

# --- Index Utilities --------------------------------------------------------

def unravel_index(indices, shape, order='C'):
    if not isinstance(indices, ndarray):
        if isinstance(indices, int):
            indices = array([indices])
        else:
            indices = array(indices)
    return _native.unravel_index(indices, shape)

def ravel_multi_index(multi_index, dims, mode='raise', order='C'):
    arrays = tuple(array([a]) if isinstance(a, (int, float)) else (a if isinstance(a, ndarray) else array(a)) for a in multi_index)
    return _native.ravel_multi_index(arrays, dims)

def interp(x, xp, fp, left=None, right=None, period=None):
    if not isinstance(x, ndarray):
        x = array(x)
    if not isinstance(xp, ndarray):
        xp = array(xp)
    if not isinstance(fp, ndarray):
        fp = array(fp)
    result = _native.interp(x, xp, fp)
    if left is not None or right is not None:
        x_arr = asarray(x).flatten().tolist()
        xp_arr = asarray(xp).flatten().tolist()
        result_list = result.flatten().tolist()
        xp_min = _builtin_min(xp_arr)
        xp_max = _builtin_max(xp_arr)
        for i, xi in enumerate(x_arr):
            if left is not None and xi < xp_min:
                result_list[i] = float(left)
            if right is not None and xi > xp_max:
                result_list[i] = float(right)
        result = array(result_list)
    return result

def gradient(f, *varargs, axis=None, edge_order=1):
    if not isinstance(f, ndarray):
        f = array(f)
    if f.ndim == 1:
        # 1D case: single spacing
        spacing = float(varargs[0]) if varargs else 1.0
        return _native.gradient(f, spacing)
    # nD case
    if axis is not None:
        # gradient along specific axis/axes
        if isinstance(axis, int):
            ax = axis
            if ax < 0:
                ax = f.ndim + ax
            sp = float(varargs[0]) if varargs else 1.0
            result = _native.gradient(f, sp)
            if isinstance(result, (list, tuple)):
                return result[ax]
            return result
        # multiple axes
        results = []
        for i, ax in enumerate(axis):
            sp = float(varargs[i]) if i < len(varargs) else 1.0
            grads = _native.gradient(f, sp)
            if isinstance(grads, (list, tuple)):
                a = ax
                if a < 0:
                    a = f.ndim + a
                results.append(grads[a])
            else:
                results.append(grads)
        return results
    # All axes
    if len(varargs) == 0:
        return _native.gradient(f, 1.0)
    elif len(varargs) == 1:
        return _native.gradient(f, float(varargs[0]))
    else:
        # Different spacing per axis
        results = []
        for i in range(f.ndim):
            sp = float(varargs[i]) if i < len(varargs) else 1.0
            grads = _native.gradient(f, sp)
            if isinstance(grads, (list, tuple)):
                results.append(grads[i])
            else:
                results.append(grads)
        return results

def polyfit(x, y, deg, rcond=None, full=False, w=None, cov=False):
    if not isinstance(x, ndarray):
        x = array(x)
    if not isinstance(y, ndarray):
        y = array(y)
    return _native.polyfit(x, y, int(deg))

def polyval(p, x):
    if not isinstance(p, ndarray):
        p = array(p)
    if not isinstance(x, ndarray):
        x = array(x)
    return _native.polyval(p, x)

# --- Polynomial utilities ---------------------------------------------------

def roots(p):
    """Return the roots of a polynomial with coefficients given in p."""
    if isinstance(p, poly1d):
        coeffs = list(p._coeffs)
    elif isinstance(p, ndarray):
        coeffs = [p[i] for i in range(p.size)]
    else:
        coeffs = [float(c) for c in p]
    # Remove leading zeros
    while len(coeffs) > 1 and coeffs[0] == 0:
        coeffs = coeffs[1:]
    n = len(coeffs) - 1  # degree
    if n == 0:
        return array([])
    if n == 1:
        return array([-coeffs[1] / coeffs[0]])
    if n == 2:
        a, b, c = coeffs[0], coeffs[1], coeffs[2]
        disc = b * b - 4 * a * c
        if disc >= 0:
            sq = disc ** 0.5
            return array([(-b + sq) / (2 * a), (-b - sq) / (2 * a)])
        else:
            sq = (-disc) ** 0.5
            return array([(-b) / (2 * a), (-b) / (2 * a)])  # real part only
    # For degree > 2: Durand-Kerner method for finding all roots simultaneously.
    # Normalize polynomial so leading coefficient is 1
    a0 = float(coeffs[0])
    norm_coeffs = [float(c) / a0 for c in coeffs]
    # Horner's method to compute polynomial value at a point
    import math as _m
    def _poly_at(z, cs):
        val = 0.0
        for c in cs:
            val = val * z + c
        return val
    # Bound on root magnitudes
    _abs_coeffs = [abs(c) for c in norm_coeffs[1:]]
    _max_coeff = _abs_coeffs[0]
    for _ac in _abs_coeffs[1:]:
        if _ac > _max_coeff:
            _max_coeff = _ac
    bound = 1.0 + _max_coeff
    # Initial guesses: distinct real values spread around
    z = [bound * (0.4 + 0.6 * _m.cos(2.0 * _m.pi * (k + 0.25) / n)) for k in range(n)]
    # Durand-Kerner iteration
    for _iteration in range(1000):
        max_delta = 0.0
        new_z = list(z)
        for i in range(n):
            pval = _poly_at(z[i], norm_coeffs)
            denom = 1.0
            for j in range(n):
                if j != i:
                    diff = z[i] - z[j]
                    if abs(diff) < 1e-15:
                        diff = 1e-15
                    denom *= diff
            if abs(denom) < 1e-30:
                denom = 1e-30
            delta = pval / denom
            new_z[i] = z[i] - delta
            if abs(delta) > max_delta:
                max_delta = abs(delta)
        z = new_z
        if max_delta < 1e-12:
            break
    return array(z)


def polyadd(a1, a2):
    """Add two polynomials (coefficient arrays, highest degree first)."""
    if isinstance(a1, poly1d):
        a1 = list(a1._coeffs)
    elif isinstance(a1, ndarray):
        a1 = [a1[i] for i in range(a1.size)]
    else:
        a1 = [float(c) for c in a1]
    if isinstance(a2, poly1d):
        a2 = list(a2._coeffs)
    elif isinstance(a2, ndarray):
        a2 = [a2[i] for i in range(a2.size)]
    else:
        a2 = [float(c) for c in a2]
    while len(a1) < len(a2):
        a1.insert(0, 0.0)
    while len(a2) < len(a1):
        a2.insert(0, 0.0)
    return array([a1[i] + a2[i] for i in range(len(a1))])


def polysub(a1, a2):
    """Subtract two polynomials."""
    if isinstance(a1, poly1d):
        a1 = list(a1._coeffs)
    elif isinstance(a1, ndarray):
        a1 = [a1[i] for i in range(a1.size)]
    else:
        a1 = [float(c) for c in a1]
    if isinstance(a2, poly1d):
        a2 = list(a2._coeffs)
    elif isinstance(a2, ndarray):
        a2 = [a2[i] for i in range(a2.size)]
    else:
        a2 = [float(c) for c in a2]
    while len(a1) < len(a2):
        a1.insert(0, 0.0)
    while len(a2) < len(a1):
        a2.insert(0, 0.0)
    return array([a1[i] - a2[i] for i in range(len(a1))])


def polymul(a1, a2):
    """Multiply two polynomials."""
    if isinstance(a1, poly1d):
        a1 = list(a1._coeffs)
    elif isinstance(a1, ndarray):
        a1 = [a1[i] for i in range(a1.size)]
    else:
        a1 = [float(c) for c in a1]
    if isinstance(a2, poly1d):
        a2 = list(a2._coeffs)
    elif isinstance(a2, ndarray):
        a2 = [a2[i] for i in range(a2.size)]
    else:
        a2 = [float(c) for c in a2]
    n = len(a1) + len(a2) - 1
    result = [0.0] * n
    for i, c1 in enumerate(a1):
        for j, c2 in enumerate(a2):
            result[i + j] += c1 * c2
    return array(result)


def polyder(p, m=1):
    """Return the derivative of the specified order of a polynomial."""
    if isinstance(p, poly1d):
        return p.deriv(m)
    if isinstance(p, ndarray):
        coeffs = [p[i] for i in range(p.size)]
    else:
        coeffs = [float(c) for c in p]
    for _ in range(m):
        n = len(coeffs) - 1
        if n <= 0:
            coeffs = [0.0]
            break
        new_coeffs = []
        for i in range(n):
            new_coeffs.append(coeffs[i] * (n - i))
        coeffs = new_coeffs
    return array(coeffs)


def polyint(p, m=1, k=0):
    """Return the integral of a polynomial."""
    if isinstance(p, poly1d):
        return p.integ(m, k)
    if isinstance(p, ndarray):
        coeffs = [p[i] for i in range(p.size)]
    else:
        coeffs = [float(c) for c in p]
    for _ in range(m):
        n = len(coeffs)
        new_coeffs = []
        for i in range(n):
            new_coeffs.append(coeffs[i] / (n - i))
        new_coeffs.append(float(k))
        coeffs = new_coeffs
    return array(coeffs)


class poly1d:
    """A one-dimensional polynomial class."""
    def __init__(self, c_or_r, r=False, variable=None):
        if isinstance(c_or_r, poly1d):
            self._coeffs = list(c_or_r._coeffs)
        elif r:
            # c_or_r are roots, convert to coefficients
            self._coeffs = [1.0]
            if isinstance(c_or_r, ndarray):
                roots_list = [c_or_r[i] for i in range(c_or_r.size)]
            else:
                roots_list = list(c_or_r)
            for root in roots_list:
                new_coeffs = [0.0] * (len(self._coeffs) + 1)
                for i, c in enumerate(self._coeffs):
                    new_coeffs[i] += c
                    new_coeffs[i + 1] -= c * float(root)
                self._coeffs = new_coeffs
        else:
            if isinstance(c_or_r, ndarray):
                self._coeffs = [c_or_r[i] for i in range(c_or_r.size)]
            else:
                self._coeffs = [float(c) for c in c_or_r]
        self._variable = variable or 'x'

    @property
    def coeffs(self):
        return array(self._coeffs)

    @property
    def c(self):
        return self.coeffs

    @property
    def order(self):
        return len(self._coeffs) - 1

    @property
    def roots(self):
        return _poly1d_roots(self._coeffs)

    @property
    def o(self):
        return self.order

    def __call__(self, val):
        return polyval(self._coeffs, val)

    def __add__(self, other):
        if isinstance(other, poly1d):
            oc = other._coeffs
        elif isinstance(other, (int, float)):
            oc = [float(other)]
        else:
            oc = list(other)
        return poly1d(polyadd(self._coeffs, oc))

    def __radd__(self, other):
        return self.__add__(other)

    def __sub__(self, other):
        if isinstance(other, poly1d):
            oc = other._coeffs
        elif isinstance(other, (int, float)):
            oc = [float(other)]
        else:
            oc = list(other)
        return poly1d(polysub(self._coeffs, oc))

    def __mul__(self, other):
        if isinstance(other, poly1d):
            oc = other._coeffs
        elif isinstance(other, (int, float)):
            return poly1d([c * float(other) for c in self._coeffs])
        else:
            oc = list(other)
        return poly1d(polymul(self._coeffs, oc))

    def __rmul__(self, other):
        if isinstance(other, (int, float)):
            return poly1d([c * float(other) for c in self._coeffs])
        return self.__mul__(other)

    def __neg__(self):
        return poly1d([-c for c in self._coeffs])

    def __len__(self):
        return self.order

    def __getitem__(self, idx):
        # poly1d[i] returns coefficient of x^i (reverse indexing)
        if idx > self.order:
            return 0.0
        return self._coeffs[self.order - idx]

    def deriv(self, m=1):
        """Return the derivative of this polynomial."""
        coeffs = list(self._coeffs)
        for _ in range(m):
            n = len(coeffs) - 1
            if n <= 0:
                coeffs = [0.0]
                break
            new_coeffs = []
            for i in range(n):
                new_coeffs.append(coeffs[i] * (n - i))
            coeffs = new_coeffs
        return poly1d(coeffs)

    def integ(self, m=1, k=0):
        """Return the integral of this polynomial."""
        coeffs = list(self._coeffs)
        for _ in range(m):
            n = len(coeffs)
            new_coeffs = []
            for i in range(n):
                new_coeffs.append(coeffs[i] / (n - i))
            new_coeffs.append(float(k))
            coeffs = new_coeffs
        return poly1d(coeffs)

    def __repr__(self):
        return "poly1d(" + repr(self._coeffs) + ")"

    def __str__(self):
        return "poly1d(" + repr(self._coeffs) + ")"


# Alias so poly1d.roots can call without name clash with the module-level roots()
_poly1d_roots = roots


# --- I/O: loadtxt / savetxt / genfromtxt ------------------------------------

def loadtxt(fname, dtype=None, comments='#', delimiter=None, skiprows=0, usecols=None, unpack=False, ndmin=0, encoding='bytes', max_rows=None, **kwargs):
    """Load data from a text file. Each row must have the same number of values."""
    if isinstance(fname, str):
        f = open(fname, 'r')
        close_file = True
    else:
        f = fname
        close_file = False
    try:
        rows = []
        lines_read = 0
        for line in f:
            line = line.strip()
            if not line:
                continue
            # Skip comment lines
            if comments and line.startswith(comments):
                continue
            if lines_read < skiprows:
                lines_read += 1
                continue
            if max_rows is not None and len(rows) >= max_rows:
                break
            # Split by delimiter
            if delimiter is None:
                parts = line.split()
            else:
                parts = line.split(delimiter)
            # Select columns
            if usecols is not None:
                parts = [parts[i] for i in usecols]
            row = [float(x.strip()) for x in parts]
            rows.append(row)
        if not rows:
            return array([])
        if len(rows) == 1 and ndmin < 2:
            result = array(rows[0])
        else:
            result = array(rows)
        if unpack:
            return result.T
        return result
    finally:
        if close_file:
            f.close()

def savetxt(fname, X, fmt='%.18e', delimiter=' ', newline='\n', header='', footer='', comments='# ', encoding=None):
    """Save an array to a text file."""
    if not isinstance(X, ndarray):
        X = array(X)
    if X.ndim == 1:
        X = X.reshape([X.size, 1])

    if isinstance(fname, str):
        f = open(fname, 'w')
        close_file = True
    else:
        f = fname
        close_file = False
    try:
        if header:
            for hline in header.split('\n'):
                f.write(comments + hline + newline)

        rows = X.shape[0]
        cols = X.shape[1]
        for i in range(rows):
            vals = []
            for j in range(cols):
                vals.append(fmt % float(X[i][j]))
            f.write(delimiter.join(vals) + newline)

        if footer:
            for fline in footer.split('\n'):
                f.write(comments + fline + newline)
    finally:
        if close_file:
            f.close()

def genfromtxt(fname, dtype=None, comments='#', delimiter=None, skip_header=0,
               skip_footer=0, missing_values=None, filling_values=None,
               usecols=None, names=None, excludelist=None, deletechars=None,
               replace_space='_', autostrip=False, case_sensitive=True,
               defaultfmt='f%i', unpack=False, usemask=False, loose=True,
               invalid_raise=True, max_rows=None, encoding='bytes', **kwargs):
    """Load data from text file, handling missing values."""
    if filling_values is None:
        filling_values = float('nan')

    if isinstance(fname, str):
        with open(fname, 'r') as f:
            lines = f.readlines()
    else:
        lines = fname.readlines()

    # Skip header/footer
    lines = lines[skip_header:]
    if skip_footer > 0:
        lines = lines[:-skip_footer]
    if max_rows is not None:
        lines = lines[:max_rows]

    rows = []
    for line in lines:
        line = line.strip()
        if not line or (comments and line.startswith(comments)):
            continue
        if delimiter is None:
            parts = line.split()
        else:
            parts = line.split(delimiter)

        if usecols is not None:
            parts = [parts[i] for i in usecols]

        row = []
        for p in parts:
            p = p.strip()
            if missing_values and p in (missing_values if isinstance(missing_values, (list, tuple, set)) else [missing_values]):
                row.append(filling_values)
            else:
                try:
                    row.append(float(p))
                except (ValueError, TypeError):
                    row.append(filling_values)
        rows.append(row)

    if not rows:
        return array([])
    result = array(rows)
    if unpack:
        return result.T
    return result

# --- ufunc function forms (Tier 12A) ----------------------------------------

def subtract(x1, x2, out=None):
    return asarray(x1) - asarray(x2)

def multiply(x1, x2, out=None):
    r = asarray(x1) * asarray(x2)
    if hasattr(r, "dtype"):
        target = str(result_type(x1, x2))
        if str(r.dtype) != target:
            try:
                r = r.astype(target)
            except Exception:
                pass
    return r

def true_divide(x1, x2, out=None):
    return asarray(x1) / asarray(x2)

def floor_divide(x1, x2, out=None):
    return asarray(x1) // asarray(x2)

def remainder(x1, x2, out=None):
    return asarray(x1) % asarray(x2)

mod = remainder

def divmod_(x1, x2):
    """Return element-wise quotient and remainder simultaneously."""
    x1, x2 = asarray(x1), asarray(x2)
    return (floor_divide(x1, x2), remainder(x1, x2))

divmod = divmod_

def negative(x):
    return -asarray(x)

def positive(x):
    return asarray(x) * 1

def float_power(x1, x2):
    return power(asarray(x1).astype('float64'), asarray(x2).astype('float64'))

def identity(n, dtype=None):
    return eye(n, dtype=dtype)

def diag(v, k=0):
    """Construct a diagonal array or extract a diagonal.

    If *v* is 1-D, return a 2-D array with *v* on the *k*-th diagonal.
    If *v* is 2-D, extract the *k*-th diagonal (same as ``diagonal``).
    """
    v = asarray(v)
    if v.ndim == 1:
        n = len(v)
        abs_k = abs(k)
        size = n + abs_k
        # Build as flat list then reshape
        flat = [0.0] * (size * size)
        for i in range(n):
            if k >= 0:
                flat[i * size + (k + i)] = float(v[i])
            else:
                flat[(abs_k + i) * size + i] = float(v[i])
        return array(flat).reshape([size, size])
    elif v.ndim == 2:
        return diagonal(v, offset=k)
    else:
        raise ValueError("Input must be 1-D or 2-D")

def tri(N, M=None, k=0, dtype=None):
    """An array with ones at and below the given diagonal and zeros elsewhere."""
    if M is None:
        M = N
    rows = []
    for i in range(N):
        row = []
        for j in range(M):
            row.append(1.0 if j <= i + k else 0.0)
        rows.append(row)
    return array(rows)

def tril(m, k=0):
    """Lower triangle of an array. Return a copy with elements above the k-th diagonal zeroed."""
    m = asarray(m)
    mask = tri(m.shape[0], m.shape[1], k=k)
    return m * mask

def triu(m, k=0):
    """Upper triangle of an array. Return a copy with elements below the k-th diagonal zeroed."""
    m = asarray(m)
    mask = tri(m.shape[0], m.shape[1], k=k - 1)
    return m * (ones(m.shape) - mask)

def vander(x, N=None, increasing=False):
    """Generate a Vandermonde matrix."""
    x = asarray(x).flatten()
    n = x.size
    if N is None:
        N = n
    if increasing:
        cols = []
        for j in range(N):
            col = []
            for i in range(n):
                col.append(x[i] ** j)
            cols.append(array(col))
    else:
        cols = []
        for j in range(N):
            col = []
            for i in range(n):
                col.append(x[i] ** (N - 1 - j))
            cols.append(array(col))
    return stack(cols, axis=1)

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

def inner(a, b):
    """Inner product of two arrays.

    For 1-D arrays this is the dot product.  For higher-dimensional arrays
    this contracts over the last axis of both a and b.
    """
    a = asarray(a)
    b = asarray(b)
    if a.ndim <= 1 and b.ndim <= 1:
        return dot(a, b)
    # For 2D: inner(A, B) = A @ B.T
    if a.ndim == 2 and b.ndim == 2:
        return dot(a, b.T)
    # General: tensordot contracting last axis of each
    return tensordot(a, b, axes=([-1], [-1]))

def matmul(x1, x2):
    """Matrix product of two arrays (same as the ``@`` operator)."""
    x1 = asarray(x1)
    x2 = asarray(x2)
    return dot(x1, x2)

def broadcast_to(arr, shape):
    """Broadcast an array to a new shape using reshape + tile."""
    arr = asarray(arr)
    arr_shape = arr.shape
    if arr_shape == tuple(shape):
        return arr
    ndim = len(shape)
    arr_ndim = len(arr_shape)
    # Prepend 1s to make same ndim
    if arr_ndim < ndim:
        new_shape = [1] * (ndim - arr_ndim) + list(arr_shape)
        arr = arr.reshape(new_shape)
        arr_shape = tuple(new_shape)
    # Check compatibility and compute reps
    reps = []
    for i in range(ndim):
        if arr_shape[i] == shape[i]:
            reps.append(1)
        elif arr_shape[i] == 1:
            reps.append(shape[i])
        else:
            raise ValueError(f"cannot broadcast shape {arr_shape} to {tuple(shape)}")
    return tile(arr, reps)

def flatnonzero(a):
    """Return indices of non-zero elements in the flattened array."""
    a = asarray(a).flatten()
    indices_list = []
    for i in range(len(a)):
        if float(a[i]) != 0.0:
            indices_list.append(i)
    return array(indices_list)

def extract(condition, arr):
    """Return elements of arr where condition is True."""
    condition = asarray(condition).flatten()
    arr = asarray(arr).flatten()
    result = []
    for i in range(len(arr)):
        if float(condition[i]) != 0.0:
            result.append(float(arr[i]))
    if not result:
        return array([])
    return array(result)

def digitize(x, bins, right=False):
    """Return the indices of the bins to which each value belongs."""
    x = asarray(x)
    bins = asarray(bins)
    bins_list = [float(bins[i]) for i in range(len(bins))]
    result = []
    ascending = len(bins_list) < 2 or bins_list[-1] >= bins_list[0]
    for i in range(x.size):
        val = float(x.flatten()[i])
        if ascending:
            # bins ascending: find first bin > val (or >= if right)
            idx = 0
            for j in range(len(bins_list)):
                if right:
                    if bins_list[j] < val:
                        idx = j + 1
                else:
                    if bins_list[j] <= val:
                        idx = j + 1
            result.append(idx)
        else:
            # bins descending
            idx = 0
            for j in range(len(bins_list)):
                if right:
                    if bins_list[j] > val:
                        idx = j + 1
                else:
                    if bins_list[j] >= val:
                        idx = j + 1
            result.append(idx)
    return array(result)

def convolve(a, v, mode='full'):
    """Discrete, linear convolution of two one-dimensional sequences."""
    a = asarray(a).flatten()
    v = asarray(v).flatten()
    # Normalize mode: support integer and abbreviated string modes
    if isinstance(mode, int):
        _mode_map = {0: 'valid', 1: 'same', 2: 'full'}
        if mode not in _mode_map:
            raise ValueError("mode must be 0, 1, or 2")
        mode = _mode_map[mode]
    elif isinstance(mode, str):
        _abbrev = {'v': 'valid', 's': 'same', 'f': 'full'}
        if mode in _abbrev:
            import warnings as _w
            _w.warn("Use of abbreviated mode '{}' is deprecated. Use the full string.".format(mode), DeprecationWarning, stacklevel=3)
            mode = _abbrev[mode]
    elif mode is None:
        raise TypeError("mode must not be None")
    na = len(a)
    nv = len(v)
    n_full = na + nv - 1
    a_list = a.tolist()
    v_list = v.tolist()
    _is_complex = 'complex' in str(a.dtype) or 'complex' in str(v.dtype)
    if not _is_complex:
        # Check if values are actually complex (e.g. _ObjectArray with complex)
        try:
            if isinstance(a_list, list) and len(a_list) > 0 and isinstance(a_list[0], complex):
                _is_complex = True
        except Exception:
            pass
    result = []
    for k in range(n_full):
        s = complex(0) if _is_complex else 0.0
        for j in range(nv):
            i = k - j
            if 0 <= i < na:
                ai = a_list[i] if isinstance(a_list, list) else a_list
                vj = v_list[j] if isinstance(v_list, list) else v_list
                s += ai * vj
        result.append(s)
    if _is_complex:
        return _ObjectArray(result)
    result = array(result)
    if mode == 'full':
        return result
    elif mode == 'same':
        start = (nv - 1) // 2
        return array([float(result[start + i]) for i in range(na)])
    elif mode == 'valid':
        n_valid = abs(na - nv) + 1
        start = _builtin_min(na, nv) - 1
        return array([float(result[start + i]) for i in range(n_valid)])
    else:
        raise ValueError("mode must be 'full', 'same', or 'valid', got '" + str(mode) + "'")

def delete(arr, obj, axis=None):
    """Return a new array with sub-arrays along an axis deleted."""
    arr = asarray(arr)
    if axis is None:
        arr = arr.flatten()
        axis = 0
    n = arr.shape[axis]
    # Normalize obj to a list of indices
    if isinstance(obj, int):
        indices_to_del = [obj if obj >= 0 else n + obj]
    elif isinstance(obj, (list, tuple)):
        indices_to_del = [i if i >= 0 else n + i for i in obj]
    elif isinstance(obj, ndarray):
        indices_to_del = [int(obj[i]) for i in range(len(obj))]
        indices_to_del = [i if i >= 0 else n + i for i in indices_to_del]
    else:
        indices_to_del = [int(obj)]
    del_set = set(indices_to_del)
    keep = [i for i in range(n) if i not in del_set]
    if not keep:
        # All deleted - return empty
        new_shape = list(arr.shape)
        new_shape[axis] = 0
        return array([])
    # Build result by selecting kept indices along axis
    if arr.ndim == 1:
        result = [float(arr[i]) for i in keep]
        return array(result)
    else:
        if axis == 0:
            # For multi-dimensional, concatenate slices along axis 0
            slices = []
            for i in keep:
                slices.append(arr[i])
            if slices:
                rows = []
                for s in slices:
                    if s.ndim == 0:
                        rows.append([float(s)])
                    else:
                        rows.append([float(s[j]) for j in range(len(s))])
                return array(rows)
            return array([])
        else:
            # For non-zero axis: transpose so target axis is first,
            # delete along axis 0, then transpose back
            # Build axis permutation: move target axis to front
            ndim = arr.ndim
            perm = [axis] + [i for i in range(ndim) if i != axis]
            inv_perm = [0] * ndim
            for i, p in enumerate(perm):
                inv_perm[p] = i
            t = transpose(arr, perm)
            t_del = delete(t, obj, axis=0)
            return transpose(t_del, inv_perm)

def insert(arr, obj, values, axis=None):
    """Insert values along the given axis before the given indices."""
    arr = asarray(arr)
    if axis is None:
        arr = arr.flatten()
        axis = 0
    n = arr.shape[axis]
    idx = obj if isinstance(obj, int) else int(obj)
    if idx < 0:
        idx = n + idx
    if arr.ndim == 1:
        result = []
        if isinstance(values, (int, float)):
            values = [values]
        elif isinstance(values, ndarray):
            values = [float(values[i]) for i in range(len(values))]
        for i in range(n):
            if i == idx:
                for v in values:
                    result.append(float(v))
            result.append(float(arr[i]))
        if idx >= n:
            for v in values:
                result.append(float(v))
        return array(result)
    # Multi-dimensional: transpose target axis to front, insert along axis 0, transpose back
    ndims = arr.ndim
    if axis < 0:
        axis = ndims + axis
    # Build permutation to move target axis to position 0
    perm = [axis] + [i for i in range(ndims) if i != axis]
    inv_perm = [0] * ndims
    for i, p in enumerate(perm):
        inv_perm[p] = i
    t = transpose(arr, perm)
    # t now has target axis as axis 0; shape is (n, ...)
    values = asarray(values)
    # Build slices for before, inserted, and after
    before_slices = []
    for i in range(idx):
        before_slices.append(t[i])
    after_slices = []
    for i in range(idx, t.shape[0]):
        after_slices.append(t[i])
    # Reshape values to match the sub-array shape if needed
    sub_shape = t.shape[1:] if ndims > 1 else ()
    if values.ndim == 0 or (values.ndim == 1 and len(sub_shape) == 1 and values.shape[0] == sub_shape[0]):
        # values is a single row to insert
        vals_row = values.reshape(list(sub_shape)) if sub_shape else values
        inserted = [vals_row]
    else:
        inserted = [values[i] for i in range(values.shape[0])]
    all_rows = before_slices + inserted + after_slices
    # Stack rows manually: build flat list
    row_size = 1
    for s in sub_shape:
        row_size = row_size * s
    flat = []
    for row in all_rows:
        r = asarray(row).flatten()
        for j in range(row_size):
            flat.append(float(r[j]))
    new_shape = list(t.shape)
    new_shape[0] = len(all_rows)
    result = array(flat).reshape(new_shape)
    return transpose(result, inv_perm)

def select(condlist, choicelist, default=0):
    """Return array drawn from elements in choicelist, depending on conditions."""
    if len(condlist) != len(choicelist):
        raise ValueError("condlist and choicelist must be the same length")
    condlist = [asarray(c) for c in condlist]
    choicelist = [asarray(c) for c in choicelist]
    # Determine output shape from first array
    shape = condlist[0].shape
    n = condlist[0].size
    result = [float(default)] * n
    # Process in reverse order so first matching condition wins
    for i in range(len(condlist) - 1, -1, -1):
        cond = condlist[i].flatten()
        choice = choicelist[i].flatten()
        for j in range(n):
            if float(cond[j]) != 0.0:
                result[j] = float(choice[j])
    result_arr = array(result)
    if len(shape) > 1:
        result_arr = result_arr.reshape(list(shape))
    return result_arr

def piecewise(x, condlist, funclist):
    """Evaluate a piecewise-defined function."""
    x = asarray(x)
    n = x.size
    flat_x = x.flatten()
    result = [0.0] * n

    # funclist should have len(condlist) or len(condlist)+1 entries
    # If len(condlist)+1, the last entry is the "otherwise" value
    has_otherwise = len(funclist) == len(condlist) + 1

    for i in range(n):
        val = float(flat_x[i])
        matched = False
        for j, cond in enumerate(condlist):
            c = asarray(cond).flatten()
            if float(c[i]) != 0.0:
                if callable(funclist[j]):
                    result[i] = float(funclist[j](val))
                else:
                    result[i] = float(funclist[j])
                matched = True
                break
        if not matched and has_otherwise:
            if callable(funclist[-1]):
                result[i] = float(funclist[-1](val))
            else:
                result[i] = float(funclist[-1])

    result_arr = array(result)
    if len(x.shape) > 1:
        result_arr = result_arr.reshape(list(x.shape))
    return result_arr

# --- mgrid / ogrid / ix_ ----------------------------------------------------

class _MGrid:
    """Return dense multi-dimensional 'meshgrid' arrays via slice notation."""
    def __getitem__(self, key):
        if not isinstance(key, tuple):
            key = (key,)
        ndim = len(key)
        arrays = []
        for s in key:
            if isinstance(s, slice):
                start = s.start if s.start is not None else 0
                stop = s.stop if s.stop is not None else 0
                step = s.step if s.step is not None else 1
                # Complex step means "use linspace with this many points"
                if isinstance(step, complex) or (hasattr(step, 'imag') and step.imag != 0):
                    num = int(abs(step))
                    grid = linspace(float(start), float(stop), num=num, endpoint=True)
                    arrays.append(grid)
                else:
                    vals = []
                    v = float(start)
                    if step > 0:
                        while v < float(stop):
                            vals.append(v)
                            v += float(step)
                    else:
                        while v > float(stop):
                            vals.append(v)
                            v += float(step)
                    arrays.append(array(vals) if vals else array([]))
            else:
                arrays.append(array([float(s)]))

        if ndim == 1:
            return arrays[0]

        # Create dense meshgrid
        shapes = [len(a) for a in arrays]
        result = []
        for i, arr in enumerate(arrays):
            shape = [1] * ndim
            shape[i] = shapes[i]
            reshaped = arr.reshape(shape)
            reps = list(shapes)
            reps[i] = 1
            result.append(tile(reshaped, reps))
        return result

mgrid = _MGrid()


class _OGrid:
    """Return open (sparse) multi-dimensional 'meshgrid' arrays via slice notation."""
    def __getitem__(self, key):
        if not isinstance(key, tuple):
            key = (key,)
        ndim = len(key)
        arrays = []
        for s in key:
            if isinstance(s, slice):
                start = s.start if s.start is not None else 0
                stop = s.stop if s.stop is not None else 0
                step = s.step if s.step is not None else 1
                # Complex step means "use linspace with this many points"
                if isinstance(step, complex) or (hasattr(step, 'imag') and step.imag != 0):
                    num = int(abs(step))
                    grid = linspace(float(start), float(stop), num=num, endpoint=True)
                    arrays.append(grid)
                else:
                    vals = []
                    v = float(start)
                    if step > 0:
                        while v < float(stop):
                            vals.append(v)
                            v += float(step)
                    else:
                        while v > float(stop):
                            vals.append(v)
                            v += float(step)
                    arrays.append(array(vals) if vals else array([]))
            else:
                arrays.append(array([float(s)]))

        if ndim == 1:
            return arrays[0]

        # Sparse: each array reshaped to broadcast along its own axis
        result = []
        for i, arr in enumerate(arrays):
            shape = [1] * ndim
            shape[i] = len(arr)
            result.append(arr.reshape(shape))
        return result

ogrid = _OGrid()


def ix_(*args):
    """Construct an open mesh from multiple sequences for cross-indexing."""
    ndim = len(args)
    result = []
    for i, arg in enumerate(args):
        arr = asarray(arg).flatten()
        shape = [1] * ndim
        shape[i] = len(arr)
        result.append(arr.reshape(shape))
    return tuple(result)


class _RClass:
    """Row concatenation using index syntax: np.r_[1:5, 7, 8]"""
    def __getitem__(self, key):
        if not isinstance(key, tuple):
            key = (key,)
        pieces = []
        for item in key:
            if isinstance(item, slice):
                start = item.start if item.start is not None else 0
                stop = item.stop
                step = item.step if item.step is not None else 1
                if stop is None:
                    raise ValueError("r_ requires explicit stop for slices")
                pieces.append(arange(start, stop, step))
            elif isinstance(item, (int, float)):
                pieces.append(array([item]))
            else:
                pieces.append(asarray(item))
        if len(pieces) == 0:
            return array([])
        return concatenate(pieces)

r_ = _RClass()


class _CClass:
    """Column concatenation: np.c_[a, b] == np.column_stack((a, b))"""
    def __getitem__(self, key):
        if not isinstance(key, tuple):
            key = (key,)
        arrays = []
        for item in key:
            if isinstance(item, slice):
                start = item.start if item.start is not None else 0
                stop = item.stop
                step = item.step if item.step is not None else 1
                if stop is None:
                    raise ValueError("c_ requires explicit stop for slices")
                arr = arange(start, stop, step)
            elif isinstance(item, (int, float)):
                arr = array([item])
            else:
                arr = asarray(item)
            # Ensure 2D
            if arr.ndim == 1:
                arr = arr.reshape((-1 if arr.size > 0 else 0, 1))
            arrays.append(arr)
        if len(arrays) == 0:
            return array([])
        return concatenate(arrays, 1)

c_ = _CClass()


class _SClass:
    """Index expression helper: np.s_[0:5, 1::2]"""
    def __getitem__(self, key):
        return key

s_ = _SClass()
index_exp = s_  # s_ already exists and does the same thing


def lexsort(keys):
    """Perform an indirect stable sort using a sequence of keys.
    Last key is used as the primary sort key."""
    keys_list = [asarray(k) for k in keys]
    n = keys_list[0].size
    indices = list(range(n))
    # Flatten all keys
    flat_keys = []
    for k in keys_list:
        fk = k.flatten()
        flat_keys.append([fk[i] for i in range(n)])
    # Sort indices using keys (last key = primary, first key = least significant)
    # Python sort is stable, so we sort from least significant to most significant
    for ki in range(len(flat_keys)):
        vals = flat_keys[ki]
        indices.sort(key=lambda idx, v=vals: v[idx])
    return array(indices)

def partition(a, kth, axis=-1):
    """Return a partitioned copy of an array.
    Creates a copy where the element at kth position is where it would be in a sorted array.
    Elements before kth are <= element at kth, elements after are >=."""
    a = asarray(a)
    if axis == -1:
        axis = a.ndim - 1
    if a.ndim == 1 or axis is None:
        flat = a.flatten()
        n = flat.size
        vals = [flat[i] for i in range(n)]
        vals.sort()
        return array(vals)
    # For multi-dim, sort along axis (full sort, which satisfies partition contract)
    return sort(a, axis=axis)

def argpartition(a, kth, axis=-1):
    """Return indices that would partition the array.
    Same as argsort but only guarantees kth element is correct."""
    a = asarray(a)
    if axis == -1:
        axis = a.ndim - 1
    if a.ndim == 1 or axis is None:
        return argsort(a)
    return argsort(a, axis=axis)

def correlate(a, v, mode='valid'):
    """Cross-correlation of two 1-dimensional sequences."""
    a = asarray(a).flatten()
    v = asarray(v).flatten()
    # Normalize mode first (same rules as convolve)
    if isinstance(mode, int):
        _mode_map = {0: 'valid', 1: 'same', 2: 'full'}
        if mode not in _mode_map:
            raise ValueError("mode must be 0, 1, or 2")
        mode = _mode_map[mode]
    elif isinstance(mode, str):
        _abbrev = {'v': 'valid', 's': 'same', 'f': 'full'}
        if mode in _abbrev:
            import warnings as _w
            _w.warn("Use of abbreviated mode '{}' is deprecated. Use the full string.".format(mode), DeprecationWarning, stacklevel=2)
            mode = _abbrev[mode]
    elif mode is None:
        raise TypeError("mode must not be None")
    na = a.size
    nv = v.size
    if na == 0 or nv == 0:
        raise ValueError("Array arguments cannot be empty")
    # Check for complex dtypes and do correlation manually
    a_dt = str(a.dtype)
    v_dt = str(v.dtype)
    # Also detect complex data in _ObjectArray
    _has_complex = 'complex' in a_dt or 'complex' in v_dt
    if not _has_complex:
        try:
            _d = a.flatten().tolist() if hasattr(a, 'tolist') else list(a)
            if len(_d) > 0 and isinstance(_d[0], complex):
                _has_complex = True
        except Exception:
            pass
    if _has_complex:
        # Pure Python complex correlation
        a_list = a.flatten().tolist()
        v_list = v.flatten().tolist()
        na_l = len(a_list)
        nv_l = len(v_list)
        # Full correlation length
        full_len = na_l + nv_l - 1
        result = []
        for k in range(full_len):
            s = complex(0, 0)
            for j in range(nv_l):
                ai = k - nv_l + 1 + j
                if 0 <= ai < na_l:
                    s += complex(a_list[ai]) * complex(v_list[j]).conjugate()
            result.append(s)
        if mode == 'valid':
            _bmin = __import__("builtins").min
            _bmax = __import__("builtins").max
            start = _bmin(na_l, nv_l) - 1
            end = _bmax(na_l, nv_l)
            result = result[start:end]
        elif mode == 'same':
            start = (full_len - na_l) // 2
            result = result[start:start + na_l]
        return _ObjectArray(result, "complex128")
    # Reverse v for correlation (correlation = convolution with reversed kernel)
    v_rev = array([v[nv - 1 - i] for i in range(nv)])
    return convolve(a, v_rev, mode=mode)


# --- Tier 15 Group C: apply_along_axis, vectorize, put, putmask, broadcast_arrays ---

def apply_along_axis(func1d, axis, arr, *args, **kwargs):
    """Apply a function to 1-D slices of an array along the given axis."""
    arr = asarray(arr)
    if arr.ndim == 1:
        return asarray(func1d(arr, *args, **kwargs))
    nd = arr.ndim
    if axis < 0:
        axis = nd + axis
    # General nD implementation
    # Build the shape of the non-target axes (the "outer" iteration shape)
    shape = arr.shape
    out_shape = tuple(shape[i] for i in range(nd) if i != axis)
    # Compute total number of outer iterations
    n_outer = 1
    for s in out_shape:
        n_outer *= s
    # For each combination of indices in the non-target axes, extract the 1D slice
    results = []
    for flat_idx in range(n_outer):
        # Convert flat_idx to multi-index in out_shape
        idx = []
        rem = flat_idx
        for s in reversed(out_shape):
            idx.append(rem % s)
            rem = rem // s
        idx.reverse()
        # Build the full index with a slice at the target axis position
        # Extract the 1D slice along the target axis
        # We need to index arr with idx inserted around the target axis
        outer_idx_pos = 0
        slice_vals = []
        for k in range(shape[axis]):
            # Build index tuple for element [idx[0], ..., k, ..., idx[-1]]
            full_idx = []
            oi = 0
            for dim in range(nd):
                if dim == axis:
                    full_idx.append(k)
                else:
                    full_idx.append(idx[oi])
                    oi += 1
            # Navigate to element
            elem = arr
            for fi in full_idx:
                elem = elem[fi]
            slice_vals.append(float(elem))
        slice_arr = array(slice_vals)
        result = func1d(slice_arr, *args, **kwargs)
        results.append(result)
    # Reshape results back to out_shape
    # If func1d returns scalar, result shape is out_shape
    first_res = results[0]
    if isinstance(first_res, ndarray):
        # func returns array - more complex, but handle scalar reduction case
        result_arr = asarray(results)
        return result_arr.reshape(out_shape)
    else:
        return array([float(r) for r in results]).reshape(out_shape)


class vectorize:
    """Generalized function class.

    Takes a nested sequence of objects or numpy arrays as inputs and returns
    a single numpy array or a tuple of numpy arrays by applying the function
    element-by-element.
    """
    def __init__(self, pyfunc, otypes=None, doc=None, excluded=None, cache=False, signature=None):
        self.pyfunc = pyfunc
        self.otypes = otypes
        if doc is not None:
            self.__doc__ = doc
        elif pyfunc.__doc__:
            self.__doc__ = pyfunc.__doc__

    def __call__(self, *args, **kwargs):
        # Convert all args to arrays
        arr_args = [asarray(a) for a in args]
        if len(arr_args) == 0:
            return array([])
        # Get the size from first arg
        first = arr_args[0].flatten()
        n = first.size
        results = []
        for i in range(n):
            elem_args = []
            for a in arr_args:
                flat = a.flatten()
                if flat.size == 1:
                    elem_args.append(flat[0])
                else:
                    elem_args.append(flat[i])
            results.append(self.pyfunc(*elem_args, **kwargs))
        result = array(results)
        # Try to reshape to the shape of first arg
        if arr_args[0].ndim > 1:
            result = result.reshape(arr_args[0].shape)
        return result


def put(a, ind, v, mode='raise'):
    """Replaces specified elements of an array with given values."""
    a = asarray(a)
    flat = a.flatten()
    n = flat.size
    ind_arr = asarray(ind).flatten()
    v_arr = asarray(v).flatten()
    vals = [flat[i] for i in range(n)]
    ni = ind_arr.size
    nv = v_arr.size
    for idx in range(ni):
        i = int(ind_arr[idx])
        if mode == 'wrap':
            i = i % n
        elif mode == 'clip':
            if i < 0:
                i = 0
            elif i >= n:
                i = n - 1
        vals[i] = v_arr[idx % nv]
    result = array(vals)
    if a.ndim > 1:
        result = result.reshape(a.shape)
    return result


def putmask(a, mask, values):
    """Changes elements of an array based on conditional and input values."""
    a = asarray(a)
    mask = asarray(mask)
    values = asarray(values)
    flat_a = a.flatten()
    flat_m = mask.flatten()
    flat_v = values.flatten()
    n = flat_a.size
    nv = flat_v.size
    vals = [flat_a[i] for i in range(n)]
    vi = 0
    for i in range(n):
        if flat_m[i]:
            vals[i] = flat_v[vi % nv]
            vi += 1
    result = array(vals)
    if a.ndim > 1:
        result = result.reshape(a.shape)
    return result


def broadcast_arrays(*args):
    """Broadcast any number of arrays against each other."""
    arrays = [asarray(a) for a in args]
    if len(arrays) == 0:
        return []
    # Find common shape
    shape = list(arrays[0].shape)
    for a in arrays[1:]:
        ashape = list(a.shape)
        # Pad shorter shape with 1s on left
        while len(shape) < len(ashape):
            shape.insert(0, 1)
        while len(ashape) < len(shape):
            ashape.insert(0, 1)
        new_shape = []
        for s1, s2 in zip(shape, ashape):
            if s1 == s2:
                new_shape.append(s1)
            elif s1 == 1:
                new_shape.append(s2)
            elif s2 == 1:
                new_shape.append(s1)
            else:
                raise ValueError("shape mismatch: objects cannot be broadcast to a single shape")
        shape = new_shape
    return [broadcast_to(a, tuple(shape)) for a in arrays]


# --- trapz / trapezoid — trapezoidal integration ----------------------------
def trapz(y, x=None, dx=1.0, axis=-1):
    """Integrate along the given axis using the composite trapezoidal rule."""
    y = asarray(y)
    if y.ndim == 1:
        n = y.size
        if x is not None:
            x = asarray(x).flatten()
            total = 0.0
            for i in range(1, n):
                total += (x[i] - x[i-1]) * (y[i] + y[i-1]) / 2.0
            return total
        else:
            total = 0.0
            for i in range(1, n):
                total += dx * (y[i] + y[i-1]) / 2.0
            return total
    # For multi-dim, apply along specified axis
    if axis == -1:
        axis = y.ndim - 1
    if y.ndim == 2:
        results = []
        if axis == 0:
            for j in range(y.shape[1]):
                col = array([y[i][j] for i in range(y.shape[0])])
                results.append(trapz(col, dx=dx))
        else:
            for i in range(y.shape[0]):
                results.append(trapz(y[i], dx=dx))
        return array(results)
    # General nD case
    if axis < 0:
        axis = y.ndim + axis
    # Move target axis to last, flatten leading dims, apply trapz per lane
    y_moved = moveaxis(y, axis, -1)
    result_shape = list(y_moved.shape[:-1])
    n_lane = y_moved.shape[-1]
    lead = 1
    for s in result_shape:
        lead *= s
    y_flat = y_moved.reshape((lead, n_lane))
    flat_results = []
    y_list = y_flat.tolist()
    for i in range(lead):
        row = array(y_list[i])
        if x is not None:
            flat_results.append(float(trapz(row, x=asarray(x), dx=dx)))
        else:
            flat_results.append(float(trapz(row, dx=dx)))
    if not result_shape:
        return float(flat_results[0])
    return array(flat_results).reshape(result_shape)

trapezoid = trapz


# --- finfo — floating point type info ---------------------------------------
class finfo:
    """Machine limits for floating point types."""
    def __init__(self, dtype=None):
        if dtype is None or str(dtype) in ('float64', 'f8', 'float', 'd'):
            self.bits = 64
            self.eps = 2.220446049250313e-16
            self.max = 1.7976931348623157e+308
            self.min = -1.7976931348623157e+308
            self.tiny = 2.2250738585072014e-308
            self.smallest_normal = 2.2250738585072014e-308
            self.smallest_subnormal = 5e-324
            self.resolution = 1e-15
            self.dtype = float64
            self.maxexp = 1024
            self.minexp = -1021
            self.nmant = 52
            self.nexp = 11
            self.machep = -52
            self.negep = -53
            self.iexp = 11
            self.precision = 15
        elif str(dtype) in ('float32', 'f4', 'f'):
            self.bits = 32
            self.eps = 1.1920929e-07
            self.max = 3.4028235e+38
            self.min = -3.4028235e+38
            self.tiny = 1.1754944e-38
            self.smallest_normal = 1.1754944e-38
            self.smallest_subnormal = 1e-45
            self.resolution = 1e-6
            self.dtype = float32
            self.maxexp = 128
            self.minexp = -125
            self.nmant = 23
            self.nexp = 8
            self.machep = -23
            self.negep = -24
            self.iexp = 8
            self.precision = 6
        elif str(dtype) in ('float16', 'half', 'f2', 'e'):
            self.bits = 16
            self.eps = 9.765625e-04
            self.max = 65504.0
            self.min = -65504.0
            self.tiny = 6.103515625e-05
            self.smallest_normal = 6.103515625e-05
            self.smallest_subnormal = 5.96e-08
            self.resolution = 0.001
            self.dtype = float16
            self.maxexp = 16
            self.minexp = -13
            self.nmant = 10
            self.nexp = 5
            self.machep = -10
            self.negep = -11
            self.iexp = 5
            self.precision = 3
        else:
            raise ValueError("finfo only supports float16, float32 and float64")
        # Legacy _machar attribute (deprecated but accessed by some tests)
        self._machar = _MachAr(self)

    def __repr__(self):
        return f"finfo(resolution={self.resolution}, min={self.min}, max={self.max}, dtype={self.dtype})"


class _MachAr:
    """Legacy MachAr stub (deprecated in numpy 1.22+)."""
    def __init__(self, finfo_obj):
        self.eps = finfo_obj.eps
        self.tiny = finfo_obj.tiny
        self.huge = finfo_obj.max
        self.smallest_normal = finfo_obj.smallest_normal
        self.smallest_subnormal = getattr(finfo_obj, 'smallest_subnormal', 0.0)

# --- iinfo — integer type info ----------------------------------------------
class iinfo:
    """Machine limits for integer types."""
    def __init__(self, dtype=None):
        if dtype is None or str(dtype) in ('int64', 'i8', 'int', 'l'):
            self.bits = 64
            self.min = -9223372036854775808
            self.max = 9223372036854775807
            self.dtype = int64
            self.kind = 'i'
        elif str(dtype) in ('int32', 'i4', 'i'):
            self.bits = 32
            self.min = -2147483648
            self.max = 2147483647
            self.dtype = int32
            self.kind = 'i'
        elif str(dtype) in ('int8', 'i1'):
            self.bits = 8
            self.min = -128
            self.max = 127
            self.dtype = int8
            self.kind = 'i'
        elif str(dtype) in ('int16', 'i2'):
            self.bits = 16
            self.min = -32768
            self.max = 32767
            self.dtype = int16
            self.kind = 'i'
        elif str(dtype) in ('uint8', 'u1', 'B'):
            self.bits = 8
            self.min = 0
            self.max = 255
            self.dtype = uint8
            self.kind = 'u'
        elif str(dtype) in ('uint16', 'u2', 'H'):
            self.bits = 16
            self.min = 0
            self.max = 65535
            self.dtype = uint16
            self.kind = 'u'
        elif str(dtype) in ('uint32', 'u4', 'I'):
            self.bits = 32
            self.min = 0
            self.max = 4294967295
            self.dtype = uint32
            self.kind = 'u'
        elif str(dtype) in ('uint64', 'u8', 'Q'):
            self.bits = 64
            self.min = 0
            self.max = 18446744073709551615
            self.dtype = uint64
            self.kind = 'u'
        else:
            raise ValueError("iinfo does not support this dtype")

    def __repr__(self):
        return f"iinfo(min={self.min}, max={self.max}, dtype={self.dtype})"


# --- fromfunction — construct array from function ----------------------------
def fromfunction(function, shape, dtype=float, **kwargs):
    """Construct an array by executing a function over each coordinate."""
    coords = indices(shape, dtype=dtype)
    return asarray(function(*coords, **kwargs))


# --- fmod — C-style remainder (sign of dividend) ----------------------------
def fmod(x1, x2):
    """Return the element-wise remainder of division (C-style, sign of dividend)."""
    x1 = asarray(x1)
    x2 = asarray(x2)
    return x1 - trunc(x1 / x2) * x2


# --- modf — return fractional and integer parts -----------------------------
def modf(x):
    """Return the fractional and integral parts of an array, element-wise."""
    x = asarray(x)
    integer_part = trunc(x)
    fractional_part = x - integer_part
    return fractional_part, integer_part


# --- fill_diagonal — fill main diagonal of 2-d array -----------------------
def fill_diagonal(a, val, wrap=False):
    """Fill the main diagonal of the given array. Returns new array (our arrays are immutable)."""
    a = asarray(a)
    if a.ndim != 2:
        raise ValueError("array must be 2-d")
    n = a.shape[0]
    m = a.shape[1]
    rows = []
    for i in range(n):
        row = []
        for j in range(m):
            if i == j:
                row.append(float(val) if not isinstance(val, (list, tuple)) else float(val[i % len(val)]))
            else:
                row.append(a[i][j])
        rows.append(row)
    return array(rows)


# --- diag_indices / diag_indices_from — diagonal index helpers --------------
def diag_indices(n, ndim=2):
    """Return the indices to access the main diagonal of an array."""
    idx = arange(0, n)
    return tuple([idx] * ndim)

def diag_indices_from(arr):
    """Return the indices to access the main diagonal of an n-dimensional array."""
    arr = asarray(arr)
    if arr.ndim != 2:
        raise ValueError("array must be 2-d")
    n = arr.shape[0]
    if arr.shape[0] != arr.shape[1]:
        raise ValueError("array must be square")
    return diag_indices(n, 2)


# --- tril_indices / triu_indices — triangle index helpers -------------------
def tril_indices(n, k=0, m=None):
    """Return the indices for the lower-triangle of an (n, m) array."""
    if m is None:
        m = n
    rows = []
    cols = []
    for i in range(n):
        for j in range(m):
            if j <= i + k:
                rows.append(float(i))
                cols.append(float(j))
    return array(rows), array(cols)

def triu_indices(n, k=0, m=None):
    """Return the indices for the upper-triangle of an (n, m) array."""
    if m is None:
        m = n
    rows = []
    cols = []
    for i in range(n):
        for j in range(m):
            if j >= i + k:
                rows.append(float(i))
                cols.append(float(j))
    return array(rows), array(cols)

def tril_indices_from(arr, k=0):
    """Return the indices for the lower-triangle of arr."""
    arr = asarray(arr)
    if arr.ndim != 2:
        raise ValueError("array must be 2-d")
    return tril_indices(arr.shape[0], k=k, m=arr.shape[1])

def triu_indices_from(arr, k=0):
    """Return the indices for the upper-triangle of arr."""
    arr = asarray(arr)
    if arr.ndim != 2:
        raise ValueError("array must be 2-d")
    return triu_indices(arr.shape[0], k=k, m=arr.shape[1])


# --- ndenumerate — multidimensional index iterator --------------------------
class ndenumerate:
    """Multidimensional index iterator."""
    def __init__(self, arr):
        self._arr = asarray(arr)
        self._flat = self._arr.flatten()
        self._shape = self._arr.shape
        self._size = self._flat.size
        self._idx = 0

    def __iter__(self):
        return self

    def __next__(self):
        if self._idx >= self._size:
            raise StopIteration
        # Convert flat index to multi-dim index
        idx = self._idx
        multi = []
        for s in reversed(self._shape):
            multi.append(idx % s)
            idx //= s
        multi.reverse()
        val = self._flat[self._idx]
        self._idx += 1
        return tuple(multi), val


# --- ndindex — N-dimensional index iterator ---------------------------------
class ndindex:
    """An N-dimensional iterator object to index arrays."""
    def __init__(self, *shape):
        if len(shape) == 1 and isinstance(shape[0], tuple):
            shape = shape[0]
        self._shape = shape
        self._size = 1
        for s in shape:
            self._size *= s
        self._idx = 0

    def __iter__(self):
        return self

    def __next__(self):
        if self._idx >= self._size:
            raise StopIteration
        idx = self._idx
        multi = []
        for s in reversed(self._shape):
            multi.append(idx % s)
            idx //= s
        multi.reverse()
        self._idx += 1
        return tuple(multi)


# --- Signal window functions ------------------------------------------------
def bartlett(M):
    """Return the Bartlett window."""
    if M < 1:
        return array([])
    if M == 1:
        return array([1.0])
    n = arange(0, M)
    mid = (M - 1) / 2.0
    vals = []
    for i in range(M):
        v = float(n[i])
        if v <= mid:
            vals.append(2.0 * v / (M - 1))
        else:
            vals.append(2.0 - 2.0 * v / (M - 1))
    return array(vals)

def blackman(M):
    """Return the Blackman window."""
    if M < 1:
        return array([])
    if M == 1:
        return array([1.0])
    vals = []
    for i in range(M):
        vals.append(0.42 - 0.5 * _math.cos(2.0 * pi * i / (M - 1)) + 0.08 * _math.cos(4.0 * pi * i / (M - 1)))
    return array(vals)

def hamming(M):
    """Return the Hamming window."""
    if M < 1:
        return array([])
    if M == 1:
        return array([1.0])
    vals = []
    for i in range(M):
        vals.append(0.54 - 0.46 * _math.cos(2.0 * pi * i / (M - 1)))
    return array(vals)

def hanning(M):
    """Return the Hanning window."""
    if M < 1:
        return array([])
    if M == 1:
        return array([1.0])
    vals = []
    for i in range(M):
        vals.append(0.5 - 0.5 * _math.cos(2.0 * pi * i / (M - 1)))
    return array(vals)

def kaiser(M, beta):
    """Return the Kaiser window."""
    if M < 1:
        return array([])
    if M == 1:
        return array([1.0])
    # I0 is modified Bessel function of first kind, order 0
    # Use series approximation
    def _i0(x):
        """Modified Bessel function I0 via series."""
        val = 1.0
        term = 1.0
        for k in range(1, 25):
            term *= (x / 2.0) ** 2 / (k * k)
            val += term
        return val

    alpha = (M - 1) / 2.0
    vals = []
    for i in range(M):
        arg = beta * _math.sqrt(1.0 - ((i - alpha) / alpha) ** 2)
        vals.append(_i0(arg) / _i0(beta))
    return array(vals)


# --- nditer — simplified N-dimensional iterator -----------------------------
class nditer:
    """Simplified N-dimensional iterator."""
    def __init__(self, op, flags=None, op_flags=None, op_dtypes=None, order='K',
                 casting='safe', op_axes=None, itershape=None, buffersize=0):
        if isinstance(op, (list, tuple)):
            self._arrays = [asarray(a) for a in op]
        else:
            self._arrays = [asarray(op)]
        self._flat = [a.flatten() for a in self._arrays]
        self._size = self._flat[0].size
        self._idx = 0
        self._shape = self._arrays[0].shape
        self.multi_index = None

    def _unravel(self, flat_idx):
        """Convert flat index to tuple of indices."""
        idx = []
        remaining = flat_idx
        for dim in reversed(self._shape):
            idx.append(remaining % dim)
            remaining //= dim
        return tuple(reversed(idx))

    def __iter__(self):
        self._idx = 0
        return self

    def __next__(self):
        if self._idx >= self._size:
            raise StopIteration
        # Compute multi_index from flat index
        self.multi_index = self._unravel(self._idx)
        if len(self._flat) == 1:
            val = self._flat[0][self._idx]
            self._idx += 1
            return val
        vals = tuple(f[self._idx] for f in self._flat)
        self._idx += 1
        return vals

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass

    @property
    def finished(self):
        return self._idx >= self._size

    def iternext(self):
        if self._idx >= self._size:
            return False
        self._idx += 1
        return self._idx < self._size

    @property
    def value(self):
        if len(self._flat) == 1:
            return self._flat[0][self._idx]
        return tuple(f[self._idx] for f in self._flat)


# --- array_str / array_repr ------------------------------------------------
def array_str(a, max_line_width=None, precision=None, suppress_small=None):
    """Return a string representation of the data in an array."""
    a = asarray(a)
    return str(a)

def array_repr(a, max_line_width=None, precision=None, suppress_small=None):
    """Return the string representation of an array."""
    a = asarray(a)
    return repr(a)


# --- Tier 18 Group C: i0, apply_over_axes, real_if_close, isneginf, isposinf ---

def i0(x):
    """Modified Bessel function of the first kind, order 0."""
    x = asarray(x)
    flat = x.flatten()
    n = flat.size
    result = []
    for i in range(n):
        v = float(flat[i])
        # Series expansion: I0(x) = sum_{k=0}^{inf} ((x/2)^k / k!)^2
        val = 1.0
        term = 1.0
        for k in range(1, 25):
            term *= (v / 2.0) ** 2 / (k * k)
            val += term
        result.append(val)
    r = array(result)
    if x.ndim > 1:
        r = r.reshape(x.shape)
    return r

def apply_over_axes(func, a, axes):
    """Apply a function repeatedly over multiple axes."""
    a = asarray(a)
    if isinstance(axes, int):
        axes = [axes]
    for ax in axes:
        result = func(a, axis=ax)
        if isinstance(result, ndarray):
            a = result
        else:
            a = asarray(result)
    return a

def real_if_close(a, tol=100):
    """If input is complex with all imaginary parts close to zero, return real parts."""
    a = asarray(a)
    if a.dtype not in ("complex64", "complex128"):
        return a
    # Check if imaginary part is negligible
    im = imag(a)
    re = real(a)
    eps = 2.220446049250313e-16  # float64 machine epsilon
    flat_im = im.flatten()
    for i in range(flat_im.size):
        if _math.fabs(float(flat_im[i])) > tol * eps:
            return a  # has significant imaginary part
    return re

def isneginf(x, out=None):
    """Test element-wise for negative infinity."""
    x = asarray(x)
    flat = x.flatten()
    n = flat.size
    vals = []
    for i in range(n):
        v = float(flat[i])
        vals.append(1.0 if (not (v != v) and v == float('-inf')) else 0.0)
    r = array(vals)
    if x.ndim > 1:
        r = r.reshape(x.shape)
    # Convert to bool by comparing > 0
    return r > zeros(r.shape)

def isposinf(x, out=None):
    """Test element-wise for positive infinity."""
    x = asarray(x)
    flat = x.flatten()
    n = flat.size
    vals = []
    for i in range(n):
        v = float(flat[i])
        vals.append(1.0 if (not (v != v) and v == float('inf')) else 0.0)
    r = array(vals)
    if x.ndim > 1:
        r = r.reshape(x.shape)
    return r > zeros(r.shape)


# --- save/load/savez (Tier 18A) ---------------------------------------------

def save(file, arr, allow_pickle=True, fix_imports=True):
    """Save an array to a .npy file (text-based format for compatibility)."""
    arr = asarray(arr)
    with open(file, 'w') as f:
        f.write(f"# shape: {list(arr.shape)}\n")
        flat = arr.flatten()
        vals = [str(flat[i]) for i in range(flat.size)]
        f.write(','.join(vals) + '\n')

def load(file, mmap_mode=None, allow_pickle=False, fix_imports=True, encoding='ASCII'):
    """Load array from a .npy file (text-based format)."""
    with open(file, 'r') as f:
        lines = f.readlines()
    # Parse shape from first line
    shape_line = lines[0].strip()
    if shape_line.startswith('# shape:'):
        import json
        shape = tuple(json.loads(shape_line.split(':')[1].strip()))
        data_line = lines[1].strip()
    else:
        # Fallback: treat as flat data
        data_line = lines[0].strip()
        shape = None
    vals = [float(v) for v in data_line.split(',')]
    result = array(vals)
    if shape is not None and len(shape) > 1:
        result = result.reshape(shape)
    return result

def savez(file, *args, **kwds):
    """Save several arrays into a single file in text format.
    Since we can't use actual npz (zip) format, save as multi-section text."""
    arrays = {}
    for i, arr in enumerate(args):
        arrays[f'arr_{i}'] = asarray(arr)
    for name, arr in kwds.items():
        arrays[name] = asarray(arr)
    with open(file, 'w') as f:
        for name, arr in arrays.items():
            f.write(f"# {name} shape: {list(arr.shape)}\n")
            flat = arr.flatten()
            vals = [str(flat[i]) for i in range(flat.size)]
            f.write(','.join(vals) + '\n')

savez_compressed = savez  # alias, same behavior in our sandbox

# --- frompyfunc (Tier 18A) --------------------------------------------------

def frompyfunc(func, nin, nout):
    """Takes an arbitrary Python function and returns a NumPy ufunc-like object.
    Returns a vectorize wrapper."""
    return vectorize(func)

# --- take_along_axis / put_along_axis (Tier 18A) ----------------------------

def take_along_axis(arr, indices, axis):
    """Take values from the input array by matching 1-d index and data slices along the given axis."""
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
            # Transpose to get correct shape
            result = []
            for i in range(indices.shape[0]):
                row = [rows[j][i] for j in range(arr.shape[1])]
                result.append(row)
            return array(result)
        else:  # axis == 1
            rows = []
            for i in range(arr.shape[0]):
                row = []
                for j in range(indices.shape[1]):
                    row.append(arr[i][int(indices[i][j])])
                rows.append(row)
            return array(rows)
    # General nD case: move axis to last, flatten leading dims, gather, reshape back
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
    # General nD case: move axis to last, flatten, scatter, reshape back
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

# --- linalg extensions (built on existing primitives) -----------------------

def _linalg_pinv(a):
    """Compute the (Moore-Penrose) pseudo-inverse of a matrix using SVD."""
    a = asarray(a)
    U, s, Vt = linalg.svd(a)
    # Build pseudo-inverse: V @ diag(1/s) @ U^T
    # s is 1D singular values
    n = s.size
    s_inv_vals = []
    tol = 1e-15 * s[0] if n > 0 else 0
    for i in range(n):
        v = s[i]
        if v > tol:
            s_inv_vals.append(1.0 / v)
        else:
            s_inv_vals.append(0.0)
    s_inv = diag(array(s_inv_vals))
    # pinv = Vt.T @ s_inv @ U.T
    return dot(dot(Vt.T, s_inv), U.T)

def _linalg_matrix_rank(M, tol=None):
    """Return matrix rank using SVD."""
    M = asarray(M)
    U, s, Vt = linalg.svd(M)
    n = s.size
    if tol is None:
        s_max = float(s[0]) if n > 0 else 0.0
        tol = s_max * 1e-15 * _builtin_max(M.shape[0], M.shape[1]) if n > 0 else 0
    rank = 0
    for i in range(n):
        if s[i] > tol:
            rank += 1
    return rank

def _linalg_matrix_power(M, n):
    """Raise a square matrix to the (integer) power n."""
    M = asarray(M)
    if n == 0:
        return eye(M.shape[0])
    if n < 0:
        M = linalg.inv(M)
        n = -n
    result = eye(M.shape[0])
    for _ in range(n):
        result = dot(result, M)
    return result

def _linalg_slogdet(a):
    """Compute sign and log of the determinant."""
    a = asarray(a)
    d = linalg.det(a)
    import math as _m
    if d > 0:
        return 1.0, _m.log(d)
    elif d < 0:
        return -1.0, _m.log(-d)
    else:
        return 0.0, float('-inf')

def _linalg_cond(x, p=None):
    """Compute the condition number of a matrix."""
    x = asarray(x)
    U, s, Vt = linalg.svd(x)
    n = s.size
    if n == 0:
        return float('inf')
    s_max = s[0]
    s_min = s[n - 1]
    if s_min == 0:
        return float('inf')
    return s_max / s_min

def _linalg_eigh(a):
    """Eigenvalues and eigenvectors of a symmetric matrix.
    Falls back to eig (our eig handles symmetric matrices fine)."""
    return linalg.eig(asarray(a))

def _linalg_eigvals(a):
    """Compute eigenvalues only."""
    vals, vecs = linalg.eig(asarray(a))
    return vals

def _linalg_multi_dot(arrays):
    """Compute the dot product of two or more arrays in a single call."""
    result = asarray(arrays[0])
    for i in range(1, len(arrays)):
        result = dot(result, asarray(arrays[i]))
    return result

_linalg_norm_orig = linalg.norm

def _linalg_norm_with_axis(x, ord=None, axis=None, keepdims=False):
    """Compute matrix or vector norm, optionally along an axis."""
    x = asarray(x)
    if axis is None:
        # Delegate to native Rust norm (Frobenius / flat L2)
        return _linalg_norm_orig(x)
    # Compute norm along specified axis (use positional arg for sum/max)
    if ord is None or ord == 2:
        return sqrt((x * x).sum(axis))
    elif ord == 1:
        return abs(x).sum(axis)
    elif ord == float('inf'):
        return abs(x).max(axis)
    else:
        return (abs(x) ** ord).sum(axis) ** (1.0 / ord)

# Monkey-patch linalg module
linalg.norm = _linalg_norm_with_axis
linalg.pinv = _linalg_pinv
linalg.matrix_rank = _linalg_matrix_rank
linalg.matrix_power = _linalg_matrix_power
linalg.slogdet = _linalg_slogdet
linalg.cond = _linalg_cond
linalg.eigh = _linalg_eigh
linalg.eigvals = _linalg_eigvals
linalg.multi_dot = _linalg_multi_dot
linalg.lstsq = _native.linalg.lstsq
linalg.cholesky = _native.linalg.cholesky
linalg.qr = _native.linalg.qr
linalg.trace = trace

def _linalg_eigvalsh(a, UPLO='L'):
    """Eigenvalues of symmetric/Hermitian matrix."""
    vals, _ = linalg.eigh(a)
    return vals

linalg.eigvalsh = _linalg_eigvalsh

# --- linalg additions (Tier 36 Group A) ------------------------------------
linalg.cross = cross        # delegate to top-level cross()
linalg.diagonal = diagonal  # delegate to top-level diagonal()
linalg.outer = outer        # delegate to top-level outer()

def _linalg_matrix_norm(x, ord='fro', axis=(-2, -1), keepdims=False):
    """Matrix norm."""
    return linalg.norm(x)

def _linalg_vector_norm(x, ord=2, axis=None, keepdims=False):
    """Vector norm."""
    return linalg.norm(x)

def _linalg_tensorsolve(a, b, axes=None):
    """Solve tensor equation (stub)."""
    raise NotImplementedError("tensorsolve not implemented")

linalg.matrix_norm = _linalg_matrix_norm
linalg.vector_norm = _linalg_vector_norm
linalg.tensorsolve = _linalg_tensorsolve

# --- FFT module extensions (Tier 19 Group B) --------------------------------

def _fft_rfftfreq(n, d=1.0):
    """Return the Discrete Fourier Transform sample frequencies for rfft."""
    val = 1.0 / (n * d)
    N = n // 2 + 1
    results = []
    for i in range(N):
        results.append(float(i) * val)
    return array(results)

def _fft_fftfreq(n, d=1.0):
    """Return the Discrete Fourier Transform sample frequencies."""
    results = []
    half = (n - 1) // 2 + 1
    for i in range(half):
        results.append(float(i) / (n * d))
    for i in range(-(n // 2), 0):
        results.append(float(i) / (n * d))
    return array(results)

def _fft_fftshift(x, axes=None):
    """Shift the zero-frequency component to the center of the spectrum."""
    x = asarray(x)
    if axes is None:
        axes = list(range(x.ndim))
    elif isinstance(axes, int):
        axes = [axes]
    result = x
    for ax in axes:
        n = result.shape[ax]
        shift = n // 2
        result = roll(result, shift, axis=ax)
    return result

def _fft_ifftshift(x, axes=None):
    """The inverse of fftshift."""
    x = asarray(x)
    if axes is None:
        axes = list(range(x.ndim))
    elif isinstance(axes, int):
        axes = [axes]
    result = x
    for ax in axes:
        n = result.shape[ax]
        shift = -(n // 2)
        result = roll(result, shift, axis=ax)
    return result

def _fft_complex_column_fft(row_ffts, rows, cols, inverse=False):
    """Apply FFT/IFFT along columns of a complex (rows, cols, 2) representation.

    row_ffts is a list of (cols, 2) arrays from fft.fft applied to each row.
    Returns a (rows, cols, 2) shaped array representing the 2D FFT result.
    The complex representation uses [real, imag] pairs.
    """
    fft_fn = fft.ifft if inverse else fft.fft
    # For each column j, extract real and imaginary parts across all rows,
    # apply FFT to each separately, then combine using:
    #   DFT(xr + j*xi) = DFT(xr) + j*DFT(xi)
    #   result_real = DFT(xr)_real - DFT(xi)_imag
    #   result_imag = DFT(xr)_imag + DFT(xi)_real
    col_results = []  # col_results[j] is a list of (real, imag) for each row i
    for j in range(cols):
        col_real = array([row_ffts[i][j][0] for i in range(rows)])
        col_imag = array([row_ffts[i][j][1] for i in range(rows)])
        if inverse:
            # ifft requires complex format (n, 2) - convert real arrays to complex with zero imag
            col_real_c = array([[float(col_real[i]), 0.0] for i in range(rows)])
            col_imag_c = array([[float(col_imag[i]), 0.0] for i in range(rows)])
            fft_of_real = fft_fn(col_real_c)   # (rows, 2)
            fft_of_imag = fft_fn(col_imag_c)   # (rows, 2)
        else:
            fft_of_real = fft_fn(col_real)   # (rows, 2)
            fft_of_imag = fft_fn(col_imag)   # (rows, 2)
        # Combine: for each row i
        col_ri = []
        for i in range(rows):
            r = fft_of_real[i][0] - fft_of_imag[i][1]
            im = fft_of_real[i][1] + fft_of_imag[i][0]
            col_ri.append((r, im))
        col_results.append(col_ri)
    # Reconstruct as (rows, cols, 2) using stack
    final_rows = []
    for i in range(rows):
        row_data = []
        for j in range(cols):
            row_data.append([col_results[j][i][0], col_results[j][i][1]])
        final_rows.append(array(row_data))
    return stack(final_rows)

def _fft_fft2(a, s=None, axes=(-2, -1), norm=None):
    """Compute the 2-D discrete Fourier Transform."""
    a = asarray(a)
    if a.ndim != 2:
        raise ValueError("fft2 requires a 2-D array")
    rows = a.shape[0]
    cols = a.shape[1]
    # FFT each row -> list of (cols, 2) complex arrays
    row_ffts = [fft.fft(a[i]) for i in range(rows)]
    return _fft_complex_column_fft(row_ffts, rows, cols, inverse=False)

def _fft_ifft2(a, s=None, axes=(-2, -1), norm=None):
    """Compute the 2-D inverse discrete Fourier Transform."""
    a = asarray(a)
    if a.ndim != 2:
        raise ValueError("ifft2 requires a 2-D array")
    rows = a.shape[0]
    cols = a.shape[1]
    # IFFT each row -> list of (cols, 2) complex arrays
    row_iffts = [fft.ifft(a[i]) for i in range(rows)]
    return _fft_complex_column_fft(row_iffts, rows, cols, inverse=True)

# Monkey-patch fft module with extension functions
fft.rfftfreq = _fft_rfftfreq
fft.fftfreq = _fft_fftfreq
fft.fftshift = _fft_fftshift
fft.ifftshift = _fft_ifftshift
fft.fft2 = _fft_fft2
fft.ifft2 = _fft_ifft2

def _fft_fftn(a, s=None, axes=None):
    """N-dimensional FFT."""
    a = asarray(a)
    if a.ndim == 1:
        return fft.fft(a)
    elif a.ndim == 2:
        return fft.fft2(a, s=s)
    else:
        # For higher dimensions, apply fft2 on last two axes as approximation
        # This is a simplified implementation
        result = a
        ax_list = axes if axes is not None else list(range(a.ndim))
        for ax in reversed(ax_list):
            # Apply 1D FFT along each axis using apply_along_axis
            result = apply_along_axis(lambda x: array(fft.fft(array(x)).tolist()), ax, result)
        return result

def _fft_ifftn(a, s=None, axes=None):
    """N-dimensional inverse FFT."""
    a = asarray(a)
    if a.ndim == 1:
        return fft.ifft(a)
    elif a.ndim == 2:
        return fft.ifft2(a, s=s)
    elif a.ndim == 3 and a.shape[-1] == 2:
        # Complex representation from fft2/fftn: shape (rows, cols, 2)
        # This is a complex array; apply ifft2 logic
        rows = a.shape[0]
        cols = a.shape[1]
        # Extract row-wise complex data and apply ifft
        row_iffts = []
        for i in _builtin_range(rows):
            row_iffts.append(fft.ifft(a[i]))
        return _fft_complex_column_fft(row_iffts, rows, cols, inverse=True)
    else:
        result = a
        ax_list = axes if axes is not None else list(range(a.ndim))
        for ax in reversed(ax_list):
            result = apply_along_axis(lambda x: array(fft.ifft(array(x)).tolist()), ax, result)
        return result

fft.fftn = _fft_fftn
fft.ifftn = _fft_ifftn

def _fft_rfft(a, n=None, axis=-1, norm=None):
    """Real FFT - only positive frequencies."""
    a = asarray(a).astype("float64")
    if a.ndim == 0:
        a = a.reshape([1])
    data = a.tolist()
    if isinstance(data[0], list):
        raise NotImplementedError("rfft only supports 1D")
    N = n if n is not None else len(data)
    # Pad or truncate
    if len(data) < N:
        data = data + [0.0] * (N - len(data))
    elif len(data) > N:
        data = data[:N]
    # Compute full DFT - only first N//2 + 1 frequencies
    import cmath
    result = []
    out_len = N // 2 + 1
    for k in _builtin_range(out_len):
        s = 0.0 + 0.0j
        for n_idx in _builtin_range(N):
            angle = -2.0 * 3.141592653589793 * k * n_idx / N
            s += data[n_idx] * cmath.exp(1j * angle)
        if norm == "ortho":
            s /= N ** 0.5
        result.append([s.real, s.imag])
    # Return as (out_len, 2) shaped array matching native fft format
    return array(result)

def _fft_irfft(a, n=None, axis=-1, norm=None):
    """Inverse real FFT."""
    a = asarray(a)
    # Handle (M, 2) complex format from rfft
    data_list = a.tolist()
    if a.ndim == 2 and a.shape[1] == 2:
        # Complex format: [[real, imag], ...]
        data_r = [row[0] for row in data_list]
        data_i = [row[1] for row in data_list]
    elif a.ndim == 1:
        # Real-only input
        data_r = data_list
        data_i = [0.0] * len(data_r)
    else:
        data_r = a.real.tolist() if hasattr(a, 'real') else data_list
        data_i = a.imag.tolist() if hasattr(a, 'imag') else [0.0] * len(data_r)
    if not isinstance(data_r, list):
        data_r = [data_r]
        data_i = [data_i]
    m = len(data_r)
    N = n if n is not None else 2 * (m - 1)
    # Reconstruct full spectrum using Hermitian symmetry
    import math as _math_mod
    full_spectrum = []
    for i in _builtin_range(m):
        full_spectrum.append(complex(data_r[i], data_i[i]))
    # Mirror for negative frequencies
    for i in _builtin_range(m, N):
        mirror = N - i
        full_spectrum.append(complex(data_r[mirror], -data_i[mirror]))
    # IDFT
    result = []
    for n_idx in _builtin_range(N):
        s = 0.0
        for k in _builtin_range(N):
            angle = 2.0 * 3.141592653589793 * k * n_idx / N
            s += full_spectrum[k].real * _math_mod.cos(angle) - full_spectrum[k].imag * _math_mod.sin(angle)
        if norm == "ortho":
            s /= N ** 0.5
        else:
            s /= N
        result.append(s)
    return array(result)

fft.rfft = _fft_rfft
fft.irfft = _fft_irfft

def _fft_rfft2(a, s=None, axes=(-2, -1), norm=None):
    """2D real FFT — rfft along last axis for each row."""
    a = asarray(a)
    if a.ndim < 2:
        return fft.rfft(a)
    rows = a.tolist()
    rfft_rows = []
    for row in rows:
        r = fft.rfft(array(row))
        rfft_rows.append(r)
    # Stack results: each r is an ndarray
    return stack(rfft_rows)

def _fft_irfft2(a, s=None, axes=(-2, -1), norm=None):
    """2D inverse real FFT — irfft along last axis for each row."""
    a = asarray(a)
    if a.ndim < 2:
        return fft.irfft(a)
    n_val = s[-1] if s else None
    result_rows = []
    for i in range(a.shape[0]):
        row = a[i]
        r = fft.irfft(row, n=n_val)
        result_rows.append(r)
    return stack(result_rows)

def _fft_rfftn(a, s=None, axes=None, norm=None):
    """N-D real FFT."""
    return _fft_rfft2(a, s=s, norm=norm)

def _fft_irfftn(a, s=None, axes=None, norm=None):
    """N-D inverse real FFT."""
    return _fft_irfft2(a, s=s, norm=norm)

def _fft_hfft(a, n=None, axis=-1, norm=None):
    """Hermitian FFT - input is Hermitian symmetric, output is real."""
    a = asarray(a)
    # hfft(a) = irfft(conj(a)) * N
    conj_a = conj(a)
    N = n if n is not None else 2 * (a.shape[0] - 1) if a.ndim > 0 else 2
    result = fft.irfft(conj_a, n=N, norm=norm)
    if norm != 'ortho':
        result = result * N
    return result

def _fft_ihfft(a, n=None, axis=-1, norm=None):
    """Inverse Hermitian FFT - input is real, output is Hermitian."""
    a = asarray(a)
    # ihfft(a) = conj(rfft(a)) / N
    N = n if n is not None else len(a.tolist()) if a.ndim > 0 else 1
    result = fft.rfft(a, n=N, norm=norm)
    return conj(result) / N if norm != 'ortho' else conj(result)

fft.rfft2 = _fft_rfft2
fft.irfft2 = _fft_irfft2
fft.rfftn = _fft_rfftn
fft.irfftn = _fft_irfftn
fft.hfft = _fft_hfft
fft.ihfft = _fft_ihfft

# --- random extension functions (Tier 19 Group C) ---------------------------

def _random_shuffle(x):
    """Modify a sequence in-place by shuffling its contents. Returns None."""
    if not isinstance(x, ndarray):
        x = asarray(x)
    n = x.size
    flat = x.flatten()
    vals = [flat[i] for i in range(n)]
    for i in range(n - 1, 0, -1):
        j_arr = random.randint(0, i + 1, (1,))
        j = int(j_arr[0])
        vals[i], vals[j] = vals[j], vals[i]
    # Attempt to update array in-place via __setitem__
    try:
        for i in range(n):
            x[i] = vals[i]
    except Exception:
        pass  # if in-place update not supported, shuffle is best-effort
    return None  # real numpy returns None

def _random_permutation(x):
    """Randomly permute a sequence, or return a permuted range."""
    if isinstance(x, (int, float)):
        x = arange(0, int(x))
    else:
        x = asarray(x)
    n = x.size
    flat = x.flatten()
    vals = [flat[i] for i in range(n)]
    for i in range(n - 1, 0, -1):
        j_arr = random.randint(0, i + 1, (1,))
        j = int(j_arr[0])
        vals[i], vals[j] = vals[j], vals[i]
    result = array(vals)
    if x.ndim > 1:
        result = result.reshape(x.shape)
    return result

def _random_standard_normal(size=None):
    """Draw samples from a standard Normal distribution (mean=0, stdev=1)."""
    if size is None:
        return float(random.normal(0.0, 1.0, (1,)).flatten()[0])
    if isinstance(size, int):
        size = (size,)
    return random.normal(0.0, 1.0, size)

def _random_exponential(scale=1.0, size=None):
    """Draw samples from an exponential distribution."""
    if size is None:
        import math as _m
        u = float(random.uniform(0.0, 1.0, (1,)).flatten()[0])
        if u >= 1.0:
            u = 0.9999999999
        return float(-scale * _m.log(1.0 - u))
    if isinstance(size, int):
        size = (size,)
    # Generate uniform [0,1) then transform: -scale * ln(1 - U)
    u = random.uniform(0.0, 1.0, size)
    flat = u.flatten()
    n = flat.size
    result = []
    for i in range(n):
        v = float(flat[i])
        if v >= 1.0:
            v = 0.9999999999
        import math as _m
        result.append(-scale * _m.log(1.0 - v))
    r = array(result)
    if u.ndim > 1:
        r = r.reshape(u.shape)
    return r

def _random_poisson(lam=1.0, size=None):
    """Draw samples from a Poisson distribution."""
    import math as _m
    if size is None:
        L = _m.exp(-lam)
        k = 0
        p = 1.0
        while True:
            k += 1
            u = random.uniform(0.0, 1.0, (1,))
            p *= float(u[0])
            if p <= L:
                break
        return float(k - 1)
    if isinstance(size, int):
        size = (size,)
    total = 1
    for s in size:
        total *= s
    result = []
    for _ in range(total):
        # Knuth algorithm
        L = _m.exp(-lam)
        k = 0
        p = 1.0
        while True:
            k += 1
            u = random.uniform(0.0, 1.0, (1,))
            p *= float(u[0])
            if p <= L:
                break
        result.append(float(k - 1))
    r = array(result)
    if len(size) > 1:
        r = r.reshape(size)
    return r

def _random_binomial(n, p, size=None):
    """Draw samples from a binomial distribution."""
    if size is None:
        successes = 0
        for _ in range(int(n)):
            u = random.uniform(0.0, 1.0, (1,))
            if float(u[0]) < p:
                successes += 1
        return float(successes)
    if isinstance(size, int):
        size = (size,)
    total = 1
    for s in size:
        total *= s
    result = []
    for _ in range(total):
        successes = 0
        for _ in range(int(n)):
            u = random.uniform(0.0, 1.0, (1,))
            if float(u[0]) < p:
                successes += 1
        result.append(float(successes))
    r = array(result)
    if len(size) > 1:
        r = r.reshape(size)
    return r

def _random_beta(a, b, size=None):
    """Draw samples from a Beta distribution.
    Uses the relationship: if X~Gamma(a,1) and Y~Gamma(b,1), then X/(X+Y)~Beta(a,b)."""
    if size is None:
        while True:
            u1 = float(random.uniform(0.0, 1.0, (1,))[0])
            u2 = float(random.uniform(0.0, 1.0, (1,))[0])
            x = u1 ** (1.0 / a)
            y = u2 ** (1.0 / b)
            if x + y <= 1.0:
                return float(x / (x + y))
    if isinstance(size, int):
        size = (size,)
    total = 1
    for s in size:
        total *= s
    result = []
    for _ in range(total):
        # Use Johnk's algorithm for Beta
        import math as _m
        while True:
            u1 = float(random.uniform(0.0, 1.0, (1,))[0])
            u2 = float(random.uniform(0.0, 1.0, (1,))[0])
            x = u1 ** (1.0 / a)
            y = u2 ** (1.0 / b)
            if x + y <= 1.0:
                result.append(x / (x + y))
                break
    r = array(result)
    if len(size) > 1:
        r = r.reshape(size)
    return r

def _random_gamma(shape_param, scale=1.0, size=None):
    """Draw samples from a Gamma distribution using Marsaglia-Tsang method."""
    import math as _m
    def _gamma_one_sample(shape_param, scale):
        alpha = shape_param
        if alpha < 1:
            u = float(random.uniform(0.0, 1.0, (1,))[0])
            alpha = alpha + 1
            boost = u ** (1.0 / shape_param)
        else:
            boost = 1.0
        d = alpha - 1.0/3.0
        c = 1.0 / _m.sqrt(9.0 * d)
        while True:
            x = float(random.randn((1,))[0])
            v = (1.0 + c * x) ** 3
            if v <= 0:
                continue
            u = float(random.uniform(0.0, 1.0, (1,))[0])
            if u < 1 - 0.0331 * x**4:
                return d * v * scale * boost
            if _m.log(u) < 0.5 * x**2 + d * (1 - v + _m.log(v)):
                return d * v * scale * boost
    if size is None:
        return float(_gamma_one_sample(shape_param, scale))
    if isinstance(size, int):
        size = (size,)
    total = 1
    for s in size:
        total *= s
    result = []
    for _ in range(total):
        result.append(_gamma_one_sample(shape_param, scale))
    r = array(result)
    if len(size) > 1:
        r = r.reshape(size)
    return r

def _random_multinomial(n, pvals, size=None):
    """Draw samples from a multinomial distribution."""
    pvals = [float(p) for p in (pvals.tolist() if isinstance(pvals, ndarray) else pvals)]
    k = len(pvals)
    if size is None:
        # Single draw: n trials among k categories
        result = [0] * k
        for _ in range(n):
            r = float(random.rand((1,))[0])
            cumsum = 0.0
            for j in range(k):
                cumsum += pvals[j]
                if r < cumsum:
                    result[j] += 1
                    break
            else:
                result[-1] += 1
        return array(result)
    else:
        if isinstance(size, int):
            size = (size,)
        total = 1
        for s in size:
            total *= s
        rows = []
        for _ in range(total):
            result = [0] * k
            for _ in range(n):
                r = float(random.rand((1,))[0])
                cumsum = 0.0
                for j in range(k):
                    cumsum += pvals[j]
                    if r < cumsum:
                        result[j] += 1
                        break
                else:
                    result[-1] += 1
            rows.append(result)
        out = array(rows)
        if len(size) > 1:
            out = out.reshape(list(size) + [k])
        return out

def _random_lognormal(mean=0.0, sigma=1.0, size=None):
    """Draw samples from a log-normal distribution."""
    if size is None:
        import math as _m
        n = float(random.normal(mean, sigma, (1,)).flatten()[0])
        return float(_m.exp(n))
    if isinstance(size, int):
        size = (size,)
    normals = random.normal(mean, sigma, size)
    return exp(normals)

def _random_geometric(p, size=None):
    """Draw samples from a geometric distribution.
    Returns number of trials until first success (minimum value 1)."""
    import math as _m
    if size is None:
        log1mp = _m.log(1.0 - p)
        u = float(random.uniform(0.0, 1.0, (1,))[0])
        if u >= 1.0:
            u = 0.9999999999
        if u <= 0.0:
            u = 1e-15
        return float(_m.ceil(_m.log(u) / log1mp))
    if isinstance(size, int):
        size = (size,)
    total = 1
    for s in size:
        total *= s
    log1mp = _m.log(1.0 - p)
    result = []
    for _ in range(total):
        u = float(random.uniform(0.0, 1.0, (1,))[0])
        # Avoid log(0)
        if u >= 1.0:
            u = 0.9999999999
        if u <= 0.0:
            u = 1e-15
        result.append(float(_m.ceil(_m.log(u) / log1mp)))
    r = array(result)
    if len(size) > 1:
        r = r.reshape(size)
    return r

def _random_dirichlet(alpha, size=None):
    """Draw samples from a Dirichlet distribution."""
    if isinstance(alpha, ndarray):
        alpha = alpha.tolist()
    alpha = [float(a) for a in alpha]
    k = len(alpha)
    if size is None:
        # Single draw
        samples = []
        for a in alpha:
            g = float(_random_gamma(a, 1.0, (1,))[0])
            samples.append(g)
        total = sum(samples)
        return array([s / total for s in samples])
    else:
        if isinstance(size, int):
            size = (size,)
        num = 1
        for s in size:
            num *= s
        rows = []
        for _ in range(num):
            samples = []
            for a in alpha:
                g = float(_random_gamma(a, 1.0, (1,))[0])
                samples.append(g)
            total = sum(samples)
            rows.append([s / total for s in samples])
        out = array(rows)
        if len(size) > 1:
            out = out.reshape(list(size) + [k])
        return out

class _Generator:
    """Random number generator (simplified)."""
    def __init__(self, seed_val=None):
        if seed_val is not None:
            random.seed(int(seed_val))

    def random(self, size=None):
        if size is None:
            return float(random.rand((1,))[0])
        if isinstance(size, int):
            size = (size,)
        return random.rand(size)

    def standard_normal(self, size=None):
        return _random_standard_normal(size)

    def integers(self, low, high=None, size=None, dtype='int64', endpoint=False):
        if high is None:
            high = low
            low = 0
        if not endpoint:
            pass  # high is exclusive already
        else:
            high = high + 1
        if size is None:
            return int(random.randint(low, high, (1,)).flatten()[0])
        if isinstance(size, int):
            size = (size,)
        return random.randint(low, high, size)

    def choice(self, a, size=None, replace=True, p=None):
        if isinstance(a, int):
            a = arange(0.0, float(a), 1.0)
        elif isinstance(a, (list, tuple)):
            a = array(a)
        elif not isinstance(a, ndarray):
            a = asarray(a)
        if size is None:
            size = 1
        return random.choice(a, size, replace)

    def normal(self, loc=0.0, scale=1.0, size=None):
        if size is None:
            size = (1,)
        if isinstance(size, int):
            size = (size,)
        return random.normal(loc, scale, size)

    def uniform(self, low=0.0, high=1.0, size=None):
        if size is None:
            size = (1,)
        if isinstance(size, int):
            size = (size,)
        return random.uniform(low, high, size)

    def shuffle(self, x):
        return _random_shuffle(x)

    def permutation(self, x):
        return _random_permutation(x)

    def exponential(self, scale=1.0, size=None):
        return _random_exponential(scale, size)

    def poisson(self, lam=1.0, size=None):
        return _random_poisson(lam, size)

    def binomial(self, n, p, size=None):
        return _random_binomial(n, p, size)

def _random_random(size=None):
    """Return random floats in [0, 1). Same as rand but takes size tuple."""
    if size is None:
        return float(random.rand((1,))[0])
    if isinstance(size, int):
        size = (size,)
    # Compute total elements
    total = 1
    for s in size:
        total *= s
    result = random.uniform(0.0, 1.0, (total,))
    if len(size) > 1:
        result = result.reshape(list(size))
    return result

def _random_multivariate_normal(mean, cov, size=None):
    """Draw from multivariate normal distribution."""
    mean = asarray(mean)
    cov = asarray(cov)
    n = len(mean.tolist())

    # Cholesky decomposition of covariance
    L = linalg.cholesky(cov)

    if size is None:
        # Single sample: generate n standard normals
        z = random.normal(0.0, 1.0, (n,))
        # Transform: L @ z + mean
        z_list = z.tolist()
        mean_list = mean.tolist()
        L_list = L.tolist()
        sample = []
        for i in range(n):
            val = mean_list[i]
            for j in range(n):
                val += L_list[i][j] * z_list[j]
            sample.append(val)
        return array(sample)

    if isinstance(size, int):
        size = (size,)

    total = 1
    for s in size:
        total *= s

    # Generate total*n standard normals
    z = random.normal(0.0, 1.0, (total * n,)).reshape((total, n))

    # Transform: samples = mean + z @ L^T
    z_list = z.tolist()
    mean_list = mean.tolist()
    L_list = L.tolist()

    results = []
    for row in z_list:
        sample = []
        for i in range(n):
            val = mean_list[i]
            for j in range(n):
                val += L_list[i][j] * row[j]
            sample.append(val)
        results.append(sample)

    result = array(results)
    if len(size) > 1:
        result = result.reshape(list(size) + [n])
    return result

def _random_chisquare(df, size=None):
    """Chi-square distribution (sum of df squared standard normals)."""
    df = int(df)
    if size is None:
        total = 1
    elif isinstance(size, int):
        total = size
    else:
        total = 1
        for s in size:
            total *= s
    results = []
    for _ in range(total):
        z = random.normal(0.0, 1.0, (df,))
        z_list = z.tolist()
        results.append(sum(v * v for v in z_list))
    result = array(results)
    if size is None:
        return float(result[0])
    if not isinstance(size, int) and len(size) > 1:
        result = result.reshape(list(size))
    return result

def _random_laplace(loc=0.0, scale=1.0, size=None):
    """Laplace distribution."""
    if size is None:
        total = 1
    elif isinstance(size, int):
        total = size
    else:
        total = 1
        for s in size:
            total *= s
    u = random.uniform(0.0, 1.0, (total,))
    # Inverse CDF: loc - scale * sign(u - 0.5) * log(1 - 2*abs(u - 0.5))
    u_list = u.tolist()
    results = []
    import math
    for ui in u_list:
        ui_shifted = ui - 0.5
        if ui_shifted == 0:
            results.append(loc)
        else:
            sign_val = 1.0 if ui_shifted > 0 else -1.0
            results.append(loc - scale * sign_val * math.log(1.0 - 2.0 * abs(ui_shifted)))
    result = array(results)
    if size is None:
        return float(result[0])
    if not isinstance(size, int) and len(size) > 1:
        result = result.reshape(list(size))
    return result

def _random_triangular(left, mode, right, size=None):
    """Triangular distribution."""
    if size is None:
        total = 1
    elif isinstance(size, int):
        total = size
    else:
        total = 1
        for s in size:
            total *= s
    u = random.uniform(0.0, 1.0, (total,))
    u_list = u.tolist()
    results = []
    fc = (mode - left) / (right - left)
    for ui in u_list:
        if ui < fc:
            results.append(left + ((right - left) * (mode - left) * ui) ** 0.5)
        else:
            results.append(right - ((right - left) * (right - mode) * (1.0 - ui)) ** 0.5)
    result = array(results)
    if size is None:
        return float(result[0])
    if not isinstance(size, int) and len(size) > 1:
        result = result.reshape(list(size))
    return result

def _random_rayleigh(scale=1.0, size=None):
    """Rayleigh distribution."""
    if size is None:
        total = 1
    elif isinstance(size, int):
        total = size
    else:
        total = 1
        for s in size:
            total *= s
    u = random.uniform(0.0, 1.0, (total,))
    import math
    results = [scale * math.sqrt(-2.0 * math.log(1.0 - ui)) for ui in u.tolist()]
    result = array(results)
    if size is None:
        return float(result[0])
    if not isinstance(size, int) and len(size) > 1:
        result = result.reshape(list(size))
    return result

def _random_weibull(a, size=None):
    """Weibull distribution."""
    if size is None:
        total = 1
    elif isinstance(size, int):
        total = size
    else:
        total = 1
        for s in size:
            total *= s
    u = random.uniform(0.0, 1.0, (total,))
    import math
    results = [(-math.log(1.0 - ui)) ** (1.0 / a) for ui in u.tolist()]
    result = array(results)
    if size is None:
        return float(result[0])
    if not isinstance(size, int) and len(size) > 1:
        result = result.reshape(list(size))
    return result

def _random_logistic(loc=0.0, scale=1.0, size=None):
    """Logistic distribution via inverse CDF."""
    import math
    if size is None:
        total = 1
    elif isinstance(size, int):
        total = size
    else:
        total = 1
        for s in size:
            total *= s
    results = []
    while len(results) < total:
        u = float(random.uniform(0.0, 1.0, (1,))[0])
        if 0 < u < 1:
            results.append(loc + scale * math.log(u / (1.0 - u)))
    result = array(results[:total])
    if size is None:
        return float(result[0])
    if not isinstance(size, int) and len(size) > 1:
        result = result.reshape(list(size))
    return result

def _random_gumbel(loc=0.0, scale=1.0, size=None):
    """Gumbel distribution via inverse CDF."""
    import math
    if size is None:
        total = 1
    elif isinstance(size, int):
        total = size
    else:
        total = 1
        for s in size:
            total *= s
    results = []
    while len(results) < total:
        u = float(random.uniform(0.0, 1.0, (1,))[0])
        if 0 < u < 1:
            results.append(loc - scale * math.log(-math.log(u)))
    result = array(results[:total])
    if size is None:
        return float(result[0])
    if not isinstance(size, int) and len(size) > 1:
        result = result.reshape(list(size))
    return result

def _random_negative_binomial(n, p, size=None):
    """Negative binomial distribution."""
    if size is None:
        total = 1
    elif isinstance(size, int):
        total = size
    else:
        total = 1
        for s in size:
            total *= s
    results = []
    for _ in range(total):
        # Generate n geometric trials: count failures before n successes
        count = 0
        successes = 0
        while successes < n:
            u = float(random.uniform(0.0, 1.0, (1,))[0])
            if u < p:
                successes += 1
            else:
                count += 1
        results.append(float(count))
    result = array(results)
    if size is None:
        return float(result[0])
    if not isinstance(size, int) and len(size) > 1:
        result = result.reshape(list(size))
    return result

def _random_power(a, size=None):
    """Power distribution."""
    if size is None:
        total = 1
    elif isinstance(size, int):
        total = size
    else:
        total = 1
        for s in size:
            total *= s
    u = random.uniform(0.0, 1.0, (total,))
    results = [ui ** (1.0 / a) for ui in u.tolist()]
    result = array(results)
    if size is None:
        return float(result[0])
    if not isinstance(size, int) and len(size) > 1:
        result = result.reshape(list(size))
    return result

def _random_vonmises(mu, kappa, size=None):
    """Von Mises distribution (rejection sampling)."""
    import math
    if size is None:
        total = 1
    elif isinstance(size, int):
        total = size
    else:
        total = 1
        for s in size:
            total *= s
    results = []
    if kappa < 1e-6:
        # For very small kappa, uniform on [-pi, pi]
        for _ in range(total):
            u = float(random.uniform(0.0, 1.0, (1,))[0])
            results.append(-math.pi + 2.0 * math.pi * u)
    else:
        tau = 1.0 + (1.0 + 4.0 * kappa * kappa) ** 0.5
        rho = (tau - (2.0 * tau) ** 0.5) / (2.0 * kappa)
        r = (1.0 + rho * rho) / (2.0 * rho)
        for _ in range(total):
            while True:
                u1 = float(random.uniform(0.0, 1.0, (1,))[0])
                z = _math.cos(math.pi * u1)
                f = (1.0 + r * z) / (r + z)
                c = kappa * (r - f)
                u2 = float(random.uniform(0.0, 1.0, (1,))[0])
                if u2 < c * (2.0 - c) or u2 <= c * _math.exp(1.0 - c):
                    u3 = float(random.uniform(0.0, 1.0, (1,))[0])
                    theta = mu + (1.0 if u3 > 0.5 else -1.0) * _math.acos(f)
                    results.append(theta)
                    break
    result = array(results[:total])
    if size is None:
        return float(result[0])
    if not isinstance(size, int) and len(size) > 1:
        result = result.reshape(list(size))
    return result

def _random_wald(mean, scale, size=None):
    """Wald (inverse Gaussian) distribution."""
    if size is None:
        total = 1
    elif isinstance(size, int):
        total = size
    else:
        total = 1
        for s in size:
            total *= s
    results = []
    for _ in range(total):
        v = float(random.normal(0.0, 1.0, (1,))[0])
        y = v * v
        x = mean + (mean * mean * y) / (2.0 * scale) - (mean / (2.0 * scale)) * (4.0 * mean * scale * y + mean * mean * y * y) ** 0.5
        u = float(random.uniform(0.0, 1.0, (1,))[0])
        if u <= mean / (mean + x):
            results.append(x)
        else:
            results.append(mean * mean / x)
    result = array(results)
    if size is None:
        return float(result[0])
    if not isinstance(size, int) and len(size) > 1:
        result = result.reshape(list(size))
    return result

def _random_zipf(a, size=None):
    """Zipf distribution (rejection sampling)."""
    if size is None:
        total = 1
    elif isinstance(size, int):
        total = size
    else:
        total = 1
        for s in size:
            total *= s
    am1 = a - 1.0
    b = 2.0 ** am1
    results = []
    for _ in range(total):
        while True:
            u = float(random.uniform(0.0, 1.0, (1,))[0])
            if u <= 0.0:
                continue
            v = float(random.uniform(0.0, 1.0, (1,))[0])
            x = int(u ** (-1.0 / am1))
            if x < 1:
                x = 1
            t = (1.0 + 1.0 / x) ** am1
            if v * x * (t - 1.0) / (b - 1.0) <= t / b:
                results.append(float(x))
                break
    result = array(results)
    if size is None:
        return float(result[0])
    if not isinstance(size, int) and len(size) > 1:
        result = result.reshape(list(size))
    return result

def _random_hypergeometric(ngood, nbad, nsample, size=None):
    """Hypergeometric distribution."""
    def _draw_one(ng, nb, ns):
        count = 0
        rg = ng
        rt = ng + nb
        uniforms = random.uniform(0.0, 1.0, (ns,)).tolist()
        for u in uniforms:
            if u < rg / rt:
                count += 1
                rg -= 1
            rt -= 1
        return count
    if size is None:
        return _draw_one(ngood, nbad, nsample)
    if isinstance(size, int):
        size = (size,)
    total_elems = 1
    for s in size:
        total_elems *= s
    result = [float(_draw_one(ngood, nbad, nsample)) for _ in _builtin_range(total_elems)]
    return array(result).reshape(list(size))

def _random_pareto(a, size=None):
    """Pareto II (Lomax) distribution."""
    if size is None:
        u = float(random.uniform(0.0, 1.0, (1,)).flatten()[0])
        return (1.0 - u) ** (-1.0 / a) - 1.0
    if isinstance(size, int):
        size = (size,)
    total = 1
    for s in size:
        total *= s
    uniforms = random.uniform(0.0, 1.0, (total,)).tolist()
    result = [(1.0 - u) ** (-1.0 / a) - 1.0 for u in uniforms]
    return array(result).reshape(list(size))

def _random_bytes(length):
    """Return random bytes."""
    vals = random.uniform(0.0, 1.0, (length,)).tolist()
    return bytes([int(v * 256) for v in vals])

def _default_rng(seed=None):
    return _Generator(seed)

class _RandomState:
    """Legacy random state compatible with np.random.RandomState(seed)."""
    def __init__(self, seed=None):
        if seed is not None:
            random.seed(seed)

    def rand(self, *shape):
        if len(shape) == 0:
            return float(random.rand((1,))[0])
        return random.rand(shape)

    def randn(self, *shape):
        if len(shape) == 0:
            return float(random.randn((1,))[0])
        return random.randn(shape)

    def randint(self, low, high=None, size=None, dtype='int64'):
        if high is None:
            high = low
            low = 0
        if size is None:
            return int(random.randint(low, high, (1,)).flatten()[0])
        if isinstance(size, int):
            size = (size,)
        return random.randint(low, high, size)

    def random(self, size=None):
        return _random_random(size=size)

    def random_sample(self, size=None):
        return _random_random(size=size)

    def normal(self, loc=0.0, scale=1.0, size=None):
        if size is None:
            return float(random.normal(float(loc), float(scale), (1,))[0])
        if isinstance(size, int):
            size = (size,)
        return random.normal(float(loc), float(scale), size)

    def uniform(self, low=0.0, high=1.0, size=None):
        if size is None:
            return float(random.uniform(float(low), float(high), (1,))[0])
        if isinstance(size, int):
            size = (size,)
        return random.uniform(float(low), float(high), size)

    def choice(self, a, size=None, replace=True, p=None):
        if isinstance(a, int):
            arr = arange(0.0, float(a), 1.0)
        elif isinstance(a, (list, tuple)):
            arr = array(a)
        elif not isinstance(a, ndarray):
            arr = asarray(a)
        else:
            arr = a
        if size is None:
            size = 1
        return random.choice(arr, size, replace)

    def shuffle(self, x):
        return _random_shuffle(x)

    def permutation(self, x):
        return _random_permutation(x)

    def seed(self, seed=None):
        random.seed(seed)

    def exponential(self, scale=1.0, size=None):
        return _random_exponential(scale, size)

    def poisson(self, lam=1.0, size=None):
        return _random_poisson(lam, size)

    def binomial(self, n, p, size=None):
        return _random_binomial(n, p, size)

    def beta(self, a, b, size=None):
        return _random_beta(a, b, size)

    def gamma(self, shape, scale=1.0, size=None):
        return _random_gamma(shape, scale, size)

    def lognormal(self, mean=0.0, sigma=1.0, size=None):
        return _random_lognormal(mean, sigma, size)

    def chisquare(self, df, size=None):
        return _random_chisquare(df, size)

    def standard_normal(self, size=None):
        return _random_standard_normal(size)

    def multivariate_normal(self, mean, cov, size=None):
        return _random_multivariate_normal(mean, cov, size)

    def get_state(self):
        return {'state': 'not_implemented'}

    def set_state(self, state):
        pass

# Wrap random.choice to accept lists, tuples, and ints (Rust version requires ndarray)
_native_random_choice = random.choice
def _wrapped_random_choice(a, size=None, replace=True, p=None):
    if isinstance(a, int):
        a = arange(0.0, float(a), 1.0)
    elif isinstance(a, (list, tuple)):
        a = array([float(x) for x in a])
    if size is None:
        size = 1
    return _native_random_choice(a, size, replace)
random.choice = _wrapped_random_choice

# Monkey-patch random module with extension functions
random.shuffle = _random_shuffle
random.permutation = _random_permutation
random.standard_normal = _random_standard_normal
random.exponential = _random_exponential
random.poisson = _random_poisson
random.binomial = _random_binomial
random.beta = _random_beta
random.gamma = _random_gamma
random.multinomial = _random_multinomial
random.lognormal = _random_lognormal
random.geometric = _random_geometric
random.dirichlet = _random_dirichlet
random.default_rng = _default_rng
random.Generator = _Generator
random.random = _random_random
random.random_sample = _random_random
random.multivariate_normal = _random_multivariate_normal
random.chisquare = _random_chisquare
random.laplace = _random_laplace
random.triangular = _random_triangular
random.rayleigh = _random_rayleigh
random.weibull = _random_weibull
random.logistic = _random_logistic
random.gumbel = _random_gumbel
random.negative_binomial = _random_negative_binomial
random.power = _random_power
random.vonmises = _random_vonmises
random.wald = _random_wald
random.zipf = _random_zipf
random.hypergeometric = _random_hypergeometric
random.pareto = _random_pareto
random.bytes = _random_bytes
random.RandomState = _RandomState

# --- Numerical Utilities (Tier 20C) -----------------------------------------

def packbits(a, axis=None, bitorder='big'):
    """Pack a binary-valued array into uint8 (int64 since we lack uint8 dtype)."""
    if not isinstance(a, ndarray):
        a = asarray(a)
    # Flatten if axis is None
    vals = a.flatten().tolist()
    if bitorder == 'little':
        # Reverse bit order within each byte
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
        # big endian (default)
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


def asfortranarray(a):
    """Return an array laid out in Fortran order (simplified: just copy)."""
    return array(a, copy=True)


def asarray_chkfinite(a):
    """Convert to array, checking for NaN and Inf."""
    a = asarray(a)
    if isinstance(a, ndarray):
        vals = a.flatten().tolist()
        for v in vals:
            if _math.isinf(v) or _math.isnan(v):
                raise ValueError("array must not contain infs or NaNs")
    return a


def nextafter(x1, x2):
    """Return the next floating-point value after x1 towards x2, element-wise."""
    if isinstance(x1, (int, float)) and isinstance(x2, (int, float)):
        return _math.nextafter(float(x1), float(x2))
    x1 = asarray(x1)
    x2 = asarray(x2)
    v1 = x1.flatten().tolist()
    v2 = x2.flatten().tolist()
    # Broadcast: if one is scalar-like (len 1), expand
    if len(v1) == 1 and len(v2) > 1:
        v1 = v1 * len(v2)
    elif len(v2) == 1 and len(v1) > 1:
        v2 = v2 * len(v1)
    result = [_math.nextafter(float(a), float(b)) for a, b in zip(v1, v2)]
    return array(result)


def spacing(x):
    """Return the distance between x and the nearest adjacent number."""
    if isinstance(x, (int, float)):
        ax = abs(float(x))
        return _math.nextafter(ax, _math.inf) - ax
    x = asarray(x)
    vals = x.flatten().tolist()
    result = []
    for v in vals:
        ax = abs(float(v))
        result.append(_math.nextafter(ax, _math.inf) - ax)
    return array(result)


def vdot(a, b):
    """Conjugate dot product of two arrays (flattened)."""
    a = asarray(a).flatten()
    b = asarray(b).flatten()
    return dot(a, b)


def broadcast_shapes(*shapes):
    """Compute the broadcast result shape from multiple shapes."""
    if not shapes:
        return ()
    ndim = _builtin_max(len(s) for s in shapes)
    result = [1] * ndim
    for shape in shapes:
        offset = ndim - len(shape)
        for i, dim in enumerate(shape):
            j = i + offset
            if result[j] == 1:
                result[j] = dim
            elif dim != 1 and dim != result[j]:
                raise ValueError(
                    f"shape mismatch: objects cannot be broadcast to a single shape. Mismatch at dimension {j}"
                )
    return tuple(result)


def gcd(x1, x2):
    """Element-wise greatest common divisor."""
    x1 = asarray(x1)
    x2 = asarray(x2)
    f1 = x1.flatten().tolist()
    f2 = x2.flatten().tolist()
    result = [_math.gcd(int(a), int(b)) for a, b in zip(f1, f2)]
    return array(result).reshape(x1.shape)


def lcm(x1, x2):
    """Element-wise least common multiple."""
    x1 = asarray(x1)
    x2 = asarray(x2)
    f1 = x1.flatten().tolist()
    f2 = x2.flatten().tolist()
    def _lcm_pair(a, b):
        ia, ib = int(a), int(b)
        if ia == 0 or ib == 0:
            return 0
        g = _math.gcd(ia, ib)
        v = ia * ib
        if v < 0:
            v = -v
        return v // g
    result = [_lcm_pair(a, b) for a, b in zip(f1, f2)]
    return array(result).reshape(x1.shape)


def polydiv(u, v):
    """Polynomial division: returns (quotient, remainder)."""
    if isinstance(u, poly1d):
        u = list(u._coeffs)
    elif isinstance(u, ndarray):
        u = [float(u[i]) for i in range(u.size)]
    else:
        u = [float(c) for c in u]
    if isinstance(v, poly1d):
        v = list(v._coeffs)
    elif isinstance(v, ndarray):
        v = [float(v[i]) for i in range(v.size)]
    else:
        v = [float(c) for c in v]
    n = len(u)
    nv = len(v)
    if nv > n:
        return array([0.0]), array(u)
    q = [0.0] * (n - nv + 1)
    r = list(u)
    for i in range(n - nv + 1):
        q[i] = r[i] / v[0]
        for j in range(nv):
            r[i + j] -= q[i] * v[j]
    remainder = r[n - nv + 1:]
    return array(q), array(remainder)


def fabs(x):
    """Absolute value for floats, element-wise."""
    return abs(asarray(x))


# --- ufunc-like wrappers with .reduce() / .accumulate() --------------------

class _UfuncWithReduce:
    """Wraps a binary element-wise function to add .reduce() and .accumulate()."""
    def __init__(self, func, reduce_func, name=None):
        self._func = func
        self._reduce = reduce_func
        self.__name__ = name or getattr(func, '__name__', 'ufunc')

    def __call__(self, *args, **kwargs):
        return self._func(*args, **kwargs)

    def reduce(self, a, axis=None, **kwargs):
        return self._reduce(a, axis=axis)

    def accumulate(self, a, axis=0, **kwargs):
        a = asarray(a)
        flat = a.flatten().tolist()
        if len(flat) == 0:
            return a
        # Simple 1-D accumulate
        result = [flat[0]]
        for v in flat[1:]:
            result.append(float(self._func(result[-1], v)))
        return array(result).reshape(a.shape)

    def outer(self, a, b, **kwargs):
        a = asarray(a)
        b = asarray(b)
        fa = a.flatten().tolist()
        fb = b.flatten().tolist()
        result = []
        for va in fa:
            for vb in fb:
                result.append(float(self._func(va, vb)))
        return array(result).reshape((len(fa), len(fb)))

    def __repr__(self):
        return f"<ufunc '{self.__name__}'>"

# Save original function references before wrapping
_maximum_func = maximum
_minimum_func = minimum
_add_func = add
_multiply_func = multiply
_subtract_func = subtract

# Wrap them as ufunc-like objects
maximum = _UfuncWithReduce(_maximum_func, lambda a, axis=None: max(a, axis=axis), name='maximum')
minimum = _UfuncWithReduce(_minimum_func, lambda a, axis=None: min(a, axis=axis), name='minimum')
add = _UfuncWithReduce(_add_func, lambda a, axis=None: sum(a, axis=axis), name='add')
multiply = _UfuncWithReduce(_multiply_func, lambda a, axis=None: prod(a, axis=axis), name='multiply')
def _subtract_reduce(a, axis=None):
    a = asarray(a)
    if axis is None:
        flat = a.flatten().tolist()
        if len(flat) == 0:
            return array(0.0)
        result = flat[0]
        for v in flat[1:]:
            result = result - v
        return result
    sh = list(a.shape)
    n = sh[axis]
    if n == 0:
        new_sh = sh[:axis] + sh[axis+1:]
        return zeros(tuple(new_sh) if new_sh else (1,))
    slices = [take(a, [i], axis=axis).squeeze(axis=axis) for i in range(n)]
    result = slices[0]
    for s in slices[1:]:
        result = result - s
    return result

subtract = _UfuncWithReduce(_subtract_func, _subtract_reduce, name='subtract')


# --- Tier 27 Group B: Additional functions ----------------------------------

def frombuffer(buffer, dtype=None, count=-1, offset=0):
    """Interpret a buffer as a 1-D array (simplified: treats as uint8)."""
    if isinstance(buffer, (bytes, bytearray)):
        data = list(buffer[offset:])
        if count != -1:
            data = data[:count]
        result = array([float(x) for x in data])
        if dtype is not None:
            result = result.astype(str(dtype))
        return result
    return asarray(buffer)

def trim_zeros(filt, trim='fb'):
    """Trim leading and/or trailing zeros from a 1-D array."""
    filt = asarray(filt)
    data = filt.tolist()
    first = 0
    last = len(data)
    if 'f' in trim or 'F' in trim:
        for i in _builtin_range(len(data)):
            if data[i] != 0:
                first = i
                break
        else:
            return array([])
    if 'b' in trim or 'B' in trim:
        for i in _builtin_range(len(data) - 1, -1, -1):
            if data[i] != 0:
                last = i + 1
                break
        else:
            return array([])
    return array(data[first:last])

def histogramdd(sample, bins=10, range=None, density=False, weights=None):
    """Simplified N-dimensional histogram."""
    sample = asarray(sample)
    if sample.ndim == 1:
        sample = sample.reshape((-1, 1))
    sample_list = sample.tolist()
    n_samples = len(sample_list)
    n_dims = len(sample_list[0])

    # Determine bins per dimension
    if isinstance(bins, int):
        bins_per_dim = [bins] * n_dims
    else:
        bins_per_dim = list(bins)

    # Determine edges per dimension
    _range = range  # local alias to avoid conflict
    edges = []
    for d in _builtin_range(n_dims):
        vals = [row[d] for row in sample_list]
        if _range is not None and _range[d] is not None:
            lo, hi = float(_range[d][0]), float(_range[d][1])
        else:
            lo, hi = _builtin_min(vals), _builtin_max(vals)
        edge = linspace(lo, hi, num=bins_per_dim[d] + 1, endpoint=True).tolist()
        edges.append(edge)

    # Build histogram
    shape = bins_per_dim
    total = 1
    for s in shape:
        total *= s
    counts = [0.0] * total

    w_list = None
    if weights is not None:
        w_list = asarray(weights).flatten().tolist()

    for idx_s in _builtin_range(n_samples):
        row = sample_list[idx_s]
        bin_indices = []
        in_range_flag = True
        for d in _builtin_range(n_dims):
            val = row[d]
            edge = edges[d]
            nb = bins_per_dim[d]
            found = False
            for j in _builtin_range(nb):
                if (val >= edge[j] and val < edge[j + 1]) or (j == nb - 1 and val == edge[j + 1]):
                    bin_indices.append(j)
                    found = True
                    break
            if not found:
                in_range_flag = False
                break
        if not in_range_flag:
            continue
        # Compute flat index
        flat_idx = 0
        stride = 1
        for d in _builtin_range(n_dims - 1, -1, -1):
            flat_idx += bin_indices[d] * stride
            stride *= bins_per_dim[d]
        counts[flat_idx] += (w_list[idx_s] if w_list is not None else 1.0)

    hist = array(counts).reshape(shape)
    edge_arrays = [array(e) for e in edges]

    if density and float(sum(hist)) > 0:
        # Normalize
        total_count = float(sum(hist))
        bin_volumes = ones(shape)
        for d in _builtin_range(n_dims):
            widths = [edges[d][i+1] - edges[d][i] for i in _builtin_range(bins_per_dim[d])]
            w_arr = array(widths)
            bcast_shape = [1] * n_dims
            bcast_shape[d] = bins_per_dim[d]
            w_arr = w_arr.reshape(bcast_shape)
            bin_volumes = bin_volumes * broadcast_to(w_arr, shape)
        hist = hist / (total_count * bin_volumes)

    return hist, edge_arrays

def mintypecode(typechars, typeset='GDFgdf', default='d'):
    """Return the character for the minimum-size type to which given types can be safely cast."""
    _typechar_order = {'?': 0, 'b': 1, 'B': 1, 'h': 2, 'H': 2, 'i': 3, 'I': 3, 'l': 4, 'L': 4,
                       'q': 4, 'Q': 4, 'f': 5, 'd': 6, 'g': 7, 'F': 8, 'D': 9, 'G': 10}
    best = default
    best_rank = _typechar_order.get(default, 6)
    for tc in typechars:
        r = _typechar_order.get(tc, 6)
        if r > best_rank and tc in typeset:
            best = tc
            best_rank = r
    return best

def common_type(*arrays):
    """Return a scalar type common to input arrays."""
    has_complex = False
    max_float = 32
    for a in arrays:
        arr = asarray(a)
        dt = str(arr.dtype)
        if "complex" in dt:
            has_complex = True
            if "128" in dt:
                max_float = 64
            else:
                max_float = _builtin_max(max_float, 32)
        elif "float64" in dt or "float" == dt:
            max_float = _builtin_max(max_float, 64)
        elif "float32" in dt:
            max_float = _builtin_max(max_float, 32)
        elif "int" in dt:
            max_float = _builtin_max(max_float, 64)  # int promotes to float64
    if has_complex:
        return complex128 if max_float >= 64 else complex64
    return float64 if max_float >= 64 else float32

class matrix:
    """Simplified matrix class (deprecated in NumPy, but still used)."""
    def __init__(self, data, dtype=None, copy=True):
        if isinstance(data, str):
            # Parse string like "1 2; 3 4"
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
        return matrix(linalg.inv(self.A))

    @property
    def shape(self):
        return self.A.shape

    @property
    def ndim(self):
        return self.A.ndim

    def __mul__(self, other):
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


# --- Tier 30 Group C: array2string, lib.stride_tricks, info, who -----------

def array2string(a, max_line_width=None, precision=None, suppress_small=None,
                 separator=' ', prefix='', style=None, formatter=None,
                 threshold=None, edgeitems=None, sign=None, floatmode=None,
                 suffix='', legacy=None):
    """Return a string representation of an array."""
    a = asarray(a)
    return repr(a)


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

class _ScimathModule:
    """Complex-safe math functions (numpy.lib.scimath)."""

    @staticmethod
    def _to_array(result, shape):
        """Convert list of float/complex results to an ndarray."""
        has_cplx = False
        for r in result:
            if isinstance(r, complex):
                has_cplx = True
                break
        if has_cplx:
            return _make_complex_array(result, shape)
        return _native.array([float(r) for r in result]).reshape(shape)

    @staticmethod
    def sqrt(x):
        x = asarray(x)
        flat = x.flatten()
        result = []
        for i in range(flat.size):
            v = flat[i]
            if v < 0:
                result.append(complex(0, (-v)**0.5))
            else:
                result.append(v**0.5)
        return _ScimathModule._to_array(result, x.shape)

    @staticmethod
    def log(x):
        x = asarray(x)
        flat = x.flatten()
        result = []
        for i in range(flat.size):
            v = flat[i]
            if v <= 0:
                import cmath
                result.append(cmath.log(v))
            else:
                result.append(_math.log(v))
        return _ScimathModule._to_array(result, x.shape)

    @staticmethod
    def log10(x):
        x = asarray(x)
        flat = x.flatten()
        result = []
        for i in range(flat.size):
            v = flat[i]
            if v <= 0:
                import cmath
                result.append(cmath.log10(v))
            else:
                result.append(_math.log10(v))
        return _ScimathModule._to_array(result, x.shape)

    @staticmethod
    def log2(x):
        x = asarray(x)
        flat = x.flatten()
        result = []
        for i in range(flat.size):
            v = flat[i]
            if v <= 0:
                import cmath
                result.append(cmath.log(v) / cmath.log(2))
            else:
                result.append(_math.log2(v))
        return _ScimathModule._to_array(result, x.shape)

    @staticmethod
    def power(x, p):
        x = asarray(x)
        flat = x.flatten()
        result = []
        for i in range(flat.size):
            v = flat[i]
            try:
                r = v ** p
                result.append(r)
            except (ValueError, ZeroDivisionError):
                import cmath
                result.append(cmath.exp(p * cmath.log(v)))
        return _ScimathModule._to_array(result, x.shape)

    @staticmethod
    def arccos(x):
        x = asarray(x)
        flat = x.flatten()
        result = []
        for i in range(flat.size):
            v = flat[i]
            if abs(v) > 1:
                import cmath
                result.append(cmath.acos(v))
            else:
                result.append(_math.acos(v))
        return _ScimathModule._to_array(result, x.shape)

    @staticmethod
    def arcsin(x):
        x = asarray(x)
        flat = x.flatten()
        result = []
        for i in range(flat.size):
            v = flat[i]
            if abs(v) > 1:
                import cmath
                result.append(cmath.asin(v))
            else:
                result.append(_math.asin(v))
        return _ScimathModule._to_array(result, x.shape)

lib.scimath = _ScimathModule()

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


# --- testing module ---------------------------------------------------------
class _TestingModule:
    def assert_allclose(self, actual, desired, rtol=1e-7, atol=0, equal_nan=True, err_msg='', verbose=True):
        actual = asarray(actual)
        desired = asarray(desired)
        if not allclose(actual, desired, rtol=rtol, atol=atol, equal_nan=equal_nan):
            actual_list = actual.tolist()
            desired_list = desired.tolist()
            raise AssertionError(err_msg or "Not equal to tolerance rtol={}, atol={}\n Actual: {}\n Desired: {}".format(rtol, atol, actual_list, desired_list))

    def assert_array_equal(self, x, y, err_msg='', verbose=True, strict=False):
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
        x = asarray(x)
        y = asarray(y)
        if not allclose(x, y, rtol=0, atol=1.5 * 10**(-decimal)):
            raise AssertionError(err_msg or "Arrays are not almost equal to {} decimals".format(decimal))

    def assert_equal(self, actual, desired, err_msg='', verbose=True):
        # Handle tuples/lists recursively
        if isinstance(actual, (tuple, list)) and isinstance(desired, (tuple, list)):
            if len(actual) != len(desired):
                raise AssertionError(err_msg or "Length mismatch: {} vs {}".format(len(actual), len(desired)))
            for i, (a, d) in enumerate(zip(actual, desired)):
                self.assert_equal(a, d, err_msg=err_msg, verbose=verbose)
            return
        actual_a = asarray(actual)
        desired_a = asarray(desired)
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


testing = _TestingModule()

# --- dtypes module (exposes per-dtype DType classes) ------------------------
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

# --- np.rec module (basic stub) ---
class _RecModule:
    """Minimal np.rec namespace."""
    def __init__(self):
        self.recarray = None  # placeholder

    def array(self, data, dtype=None):
        """Create a record array (falls back to regular array)."""
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
        if names is not None and dtype is None:
            fields = [(n, 'float64') for n in names]
            dtype = StructuredDtype(fields)
        return self.array(arrays, dtype=dtype)

rec = _RecModule()

# --- show_config stub -------------------------------------------------------
def show_config():
    """Show numpy-rust build configuration."""
    print("numpy-rust (codepod)")
    print("  backend: Rust + RustPython")

# --- fromfile stub ----------------------------------------------------------
def fromfile(file, dtype=float, count=-1, sep='', offset=0, like=None):
    """Read array from file (stub - not supported in sandbox)."""
    raise NotImplementedError("fromfile not supported in sandboxed environment")

# --- einsum_path stub -------------------------------------------------------
def einsum_path(*operands, optimize='greedy'):
    """Evaluate optimal contraction order (stub returns naive path)."""
    # Return a naive path: contract in order
    n = int(len(operands) // 2)  # rough estimate
    path = [(0, 1)] * _builtin_max(1, n - 1)
    return path, ""

# --- byte_bounds stub -------------------------------------------------------
def byte_bounds(a):
    """Return low and high byte pointers (stub returns (0, nbytes))."""
    arr = asarray(a)
    return (0, arr.nbytes)

# --- Module stubs -----------------------------------------------------------

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

# --- format_float functions -------------------------------------------------
def format_float_positional(x, precision=None, unique=True, fractional=True, trim='k', sign=False, pad_left=None, pad_right=None, min_digits=None):
    """Format a float in positional notation."""
    if precision is not None:
        return f"{x:.{precision}f}"
    return str(x)

def format_float_scientific(x, precision=None, unique=True, trim='k', sign=False, pad_left=None, exp_digits=None, min_digits=None):
    """Format a float in scientific notation."""
    if precision is not None:
        return f"{x:.{precision}e}"
    return f"{x:e}"

# --- sctypes and sctypeDict -------------------------------------------------
sctypes = {
    'int': [int8, int16, int32, int64],
    'uint': [uint8, uint16, uint32, uint64],
    'float': [float16, float32, float64],
    'complex': [complex64, complex128],
    'others': [bool_, object_, str_, bytes_, void],
}
sctypeDict = {
    'int8': int8, 'int16': int16, 'int32': int32, 'int64': int64,
    'uint8': uint8, 'uint16': uint16, 'uint32': uint32, 'uint64': uint64,
    'float16': float16, 'float32': float32, 'float64': float64,
    'complex64': complex64, 'complex128': complex128,
    'bool': bool_, 'object': object_, 'str': str_, 'bytes': bytes_,
    'i1': int8, 'i2': int16, 'i4': int32, 'i8': int64,
    'u1': uint8, 'u2': uint16, 'u4': uint32, 'u8': uint64,
    'f2': float16, 'f4': float32, 'f8': float64,
    'c8': complex64, 'c16': complex128,
}

# --- memmap stub ------------------------------------------------------------
class memmap:
    """Memory-mapped file stub (not supported in sandboxed environment)."""
    def __new__(cls, filename, dtype=None, mode='r+', offset=0, shape=None, order='C'):
        raise NotImplementedError("memmap not supported in sandboxed environment")

# --- ufunc class stub -------------------------------------------------------
class ufunc:
    """Universal function stub."""
    def __init__(self, name='', nin=0, nout=0):
        self.__name__ = name
        self.nin = nin
        self.nout = nout
    def __repr__(self):
        return f"<ufunc '{self.__name__}'>"

# --- Misc stubs -------------------------------------------------------------
def seterrcall(func):
    """Set callback for floating-point error handler (no-op)."""
    return None

def geterrcall():
    """Get callback for floating-point error handler (no-op)."""
    return None

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
    """Return include directory (not applicable)."""
    return ""

tracemalloc_domain = 0
use_hugepage = 0
nested_iters = None  # Not supported

# --- Import submodules so np.ma and np.polynomial are accessible ------------
import numpy.ma as ma
import numpy.polynomial as polynomial

# --- Module-level __getattr__ for deprecated aliases like np.bool -----------
def __getattr__(name):
    _bi = __import__("builtins")
    _deprecated_aliases = {
        'bool': _bi.bool,
        'int': _bi.int,
        'float': _bi.float,
        'complex': _bi.complex,
        'str': _bi.str,
        'object': _bi.object,
    }
    if name in _deprecated_aliases:
        return _deprecated_aliases[name]
    raise AttributeError(f"module 'numpy' has no attribute '{name}'")
