"""Internal helpers used by multiple numpy submodules."""
import sys as _sys
import math as _math
import _numpy_native as _native

# Builtin aliases (these names get shadowed by numpy functions later)
_builtin_min = __builtins__["min"] if isinstance(__builtins__, dict) else __import__("builtins").min
_builtin_max = __builtins__["max"] if isinstance(__builtins__, dict) else __import__("builtins").max
_builtin_range = __builtins__["range"] if isinstance(__builtins__, dict) else __import__("builtins").range

__all__ = [
    'AxisError', '_ArrayFlags', '_ObjectArray', '_ComplexResultArray',
    '_copy_into', '_apply_order', '_is_temporal_dtype', '_temporal_dtype_info',
    '_make_temporal_array', '_infer_shape', '_flatten_nested', '_to_float_list',
    '_unsupported_numeric_dtype', '_CLIP_UNSET',
    '_builtin_min', '_builtin_max', '_builtin_range',
]

from _numpy_native import ndarray


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
    def dtype(self):
        # Return a proper dtype object so callers can use .str, .kind, etc.
        if isinstance(self._dtype, str):
            try:
                import numpy as _np
                return _np.dtype(self._dtype)
            except Exception:
                pass
        return self._dtype
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
    def astype(self, dtype):
        dtype_str = str(dtype)
        # Convert temporal types to numeric
        if dtype_str in ('int64', 'int32', 'int16', 'int8', 'uint64', 'uint32') or dtype_str.startswith('int') or dtype_str.startswith('uint'):
            if self._data and getattr(self._data[0], '_is_timedelta64', False) or getattr(self._data[0] if self._data else None, '_is_datetime64', False):
                vals = [v._value if hasattr(v, '_value') else int(v) for v in self._data]
                result = _native.array([float(v) for v in vals]).astype(dtype_str)
                if len(self._shape) > 1:
                    result = result.reshape(list(self._shape))
                return result
        # Convert to temporal types
        if _is_temporal_dtype(dtype_str):
            return _make_temporal_array(self, dtype_str)
        return _ObjectArray(list(self._data), dtype_str, shape=self._shape, is_fortran=self._is_fortran)
    def flatten(self): return self
    def ravel(self): return self
    def tolist(self): return list(self._data)
    def all(self): return all(self._data)
    def any(self): return any(self._data)
    def __len__(self): return len(self._data)
    def _get_structured_field_names(self):
        """Parse structured dtype string to ordered list of (name, dtype_str) tuples."""
        import re
        dt = self._dtype
        if not isinstance(dt, str):
            return None
        # Match patterns like ('name', 'dtype') in a list
        fields = re.findall(r"\(\s*['\"](\w+)['\"],\s*['\"]([^'\"]+)['\"]", dt)
        return fields if fields else None

    def __getitem__(self, key):
        if isinstance(key, str):
            # Structured dtype field access: x['field']
            fields = self._get_structured_field_names()
            if fields is not None:
                names = [f[0] for f in fields]
                if key in names:
                    idx = names.index(key)
                    field_dtype = fields[idx][1]
                    import numpy as _np
                    nd = _np._normalize_dtype(field_dtype)
                    # Detect 1D vs 2D structured array
                    # 1D: _data = [(t1), (t2), ...] where each t is a record tuple
                    # 2D: _data = [[row1], [row2], ...] where each row is list of record tuples
                    if len(self._data) > 0 and isinstance(self._data[0], list):
                        # 2D structured array
                        out_shape = (len(self._data), len(self._data[0]))
                        flat = [self._data[i][j][idx]
                                for i in range(out_shape[0])
                                for j in range(out_shape[1])]
                    else:
                        # 1D structured array
                        flat = [row[idx] for row in self._data]
                        out_shape = (len(flat),)
                    try:
                        result = _native.array([float(v) for v in flat]).astype(nd)
                        result = result.reshape(list(out_shape))
                        # Structured field views are unaligned (interleaved memory semantics)
                        result._set_unaligned()
                        return result
                    except Exception:
                        return _ObjectArray(flat, field_dtype)
            raise IndexError("field %r not found in structured dtype" % key)
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
        import numpy as _np
        return _np.dtype("complex128")

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


def _is_temporal_dtype(s):
    """Return True if dtype string represents datetime64 or timedelta64."""
    if not isinstance(s, str):
        return False
    # Strip byte-order prefix
    bare = s.lstrip('<>=|')
    return (bare.startswith('m8') or bare.startswith('M8') or
            bare.startswith('timedelta64') or bare.startswith('datetime64'))


def _temporal_dtype_info(s):
    """Parse a temporal dtype string and return (kind, unit, canonical_name, str_form).
    kind: 'm' for timedelta64, 'M' for datetime64.
    unit: 'ns', 'us', 's', 'ms', 'D', 'generic', etc. (None = no unit specified)
    canonical_name: e.g. 'timedelta64[ns]' or 'datetime64[ns]'
    str_form: e.g. '<m8[ns]' or '<M8[ns]'
    """
    bare = s.lstrip('<>=|')
    if bare.startswith('m8') or bare.startswith('timedelta64'):
        kind = 'm'
        prefix = 'm8'
        long_prefix = 'timedelta64'
        rest = bare[2:] if bare.startswith('m8') else bare[len('timedelta64'):]
    else:  # M8 / datetime64
        kind = 'M'
        prefix = 'M8'
        long_prefix = 'datetime64'
        rest = bare[2:] if bare.startswith('M8') else bare[len('datetime64'):]
    # rest is like '[ns]', '[us]', '' etc.
    unit = None
    if rest.startswith('[') and rest.endswith(']'):
        unit = rest[1:-1]
    if unit:
        canonical = '{}[{}]'.format(long_prefix, unit)
        str_form = '<{}[{}]'.format(prefix, unit)
    else:
        canonical = long_prefix
        str_form = '<{}'.format(prefix)
    return kind, unit, canonical, str_form


def _make_temporal_array(data, dtype_str):
    """Create an _ObjectArray with datetime64/timedelta64 elements.
    dtype_str is the canonical form e.g. 'timedelta64[ns]' or 'datetime64[us]'.
    """
    import numpy as _np
    _datetime64_cls = _np.datetime64
    _timedelta64_cls = _np.timedelta64
    kind, unit, canonical, _ = _temporal_dtype_info(dtype_str)
    is_td = (kind == 'm')
    unit = unit or 'generic'

    def _convert_element(x):
        if isinstance(x, str):
            s = x.strip()
            if s == 'NaT':
                return _timedelta64_cls('NaT', unit) if is_td else _datetime64_cls('NaT', unit)
            # Date/time string for datetime64
            if not is_td:
                return _datetime64_cls(s, unit)
            # Timedelta: try int parse, else NaT
            try:
                return _timedelta64_cls(int(s), unit)
            except (ValueError, TypeError):
                return _timedelta64_cls('NaT', unit)
        if getattr(x, '_is_datetime64', False) or getattr(x, '_is_timedelta64', False):
            return x
        if isinstance(x, (int, float)):
            return _timedelta64_cls(int(x), unit) if is_td else _datetime64_cls(int(x), unit)
        # Unknown: return as-is
        return x

    def _flatten_temporal(d):
        if isinstance(d, (list, tuple)):
            result = []
            for item in d:
                if isinstance(item, (list, tuple)):
                    result.extend(_flatten_temporal(item))
                else:
                    result.append(_convert_element(item))
            return result
        return [_convert_element(d)]

    def _infer_temporal_shape(d):
        if isinstance(d, (list, tuple)):
            if len(d) == 0:
                return (0,)
            inner = _infer_temporal_shape(d[0])
            return (len(d),) + inner
        return ()

    if isinstance(data, (list, tuple)):
        shape = _infer_temporal_shape(data)
        flat = _flatten_temporal(data)
    elif isinstance(data, _ObjectArray):
        shape = data._shape
        flat = [_convert_element(v) for v in data._data]
    elif isinstance(data, ndarray):
        shape = data.shape
        flat = [_convert_element(v) for v in data.flatten().tolist()]
    else:
        shape = ()
        flat = [_convert_element(data)]

    return _ObjectArray(flat, canonical, shape=shape if shape != () else (len(flat),))


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

_CLIP_UNSET = object()
