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
    '_make_temporal_array', '_infer_shape', '_flatten_nested', '_all_bools_nested',
    '_to_float_list', '_unsupported_numeric_dtype', '_CLIP_UNSET',
    '_coerce_native_boxed_operand',
    '_builtin_min', '_builtin_max', '_builtin_range',
]

from _numpy_native import ndarray
import cmath as _cmath


def _c99_complex_mul(a, b):
    """C99-compliant complex multiplication handling inf/nan special cases."""
    ac = a.real * b.real
    bd = a.imag * b.imag
    ad = a.real * b.imag
    bc = a.imag * b.real
    re = ac - bd
    im = ad + bc
    if _math.isnan(re) or _math.isnan(im):
        a_inf = _math.isinf(a.real) or _math.isinf(a.imag)
        b_inf = _math.isinf(b.real) or _math.isinf(b.imag)
        if a_inf:
            ar = 1.0 if _math.isinf(a.real) else 0.0
            ai = 1.0 if _math.isinf(a.imag) else 0.0
            if a.real < 0 and _math.isinf(a.real): ar = -1.0
            if a.imag < 0 and _math.isinf(a.imag): ai = -1.0
            br = 0.0 if _math.isnan(b.real) else b.real
            bi = 0.0 if _math.isnan(b.imag) else b.imag
            re = float('inf') * (ar * br - ai * bi)
            im = float('inf') * (ar * bi + ai * br)
        elif b_inf:
            br = 1.0 if _math.isinf(b.real) else 0.0
            bi = 1.0 if _math.isinf(b.imag) else 0.0
            if b.real < 0 and _math.isinf(b.real): br = -1.0
            if b.imag < 0 and _math.isinf(b.imag): bi = -1.0
            ar = 0.0 if _math.isnan(a.real) else a.real
            ai = 0.0 if _math.isnan(a.imag) else a.imag
            re = float('inf') * (ar * br - ai * bi)
            im = float('inf') * (ar * bi + ai * br)
    return complex(re, im)


def _complex_pow(base, exp):
    """Complex power that handles special cases RustPython gets wrong."""
    exp_real = exp
    if isinstance(exp, complex):
        if exp.imag == 0:
            exp_real = exp.real
        else:
            exp_real = None
    if exp_real is not None and isinstance(exp_real, (int, float)):
        try:
            if exp_real == int(exp_real) and abs(exp_real) <= 100:
                n = int(exp_real)
                if n == 0:
                    return complex(1, 0)
                base = complex(base) if not isinstance(base, complex) else base
                if n < 0:
                    base = complex(1, 0) / base
                    n = -n
                result = complex(1, 0)
                b = base
                while n > 0:
                    if n % 2 == 1:
                        result = _c99_complex_mul(result, b)
                    b = _c99_complex_mul(b, b)
                    n //= 2
                return result
        except (OverflowError, ValueError, ZeroDivisionError):
            pass
    try:
        return base ** exp
    except (OverflowError, ValueError):
        return complex(float('nan'), float('nan'))


class AxisError(ValueError, IndexError):
    """Exception for invalid axis."""
    def __init__(self, axis=None, ndim=None, msg_prefix=None):
        # If axis is a string and ndim is not provided, treat as plain message
        if isinstance(axis, str) and ndim is None:
            msg = axis
            if msg_prefix:
                msg = "{}: {}".format(msg_prefix, msg)
            super().__init__(msg)
            self.axis = None
            self.ndim = None
            return
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
        # fnc = Fortran non-contiguous (Fortran-order but not contiguous)
        # For plain Python arrays this is always False
        self.fnc = False
    def __getitem__(self, key):
        k = key.upper()
        if k in ('C_CONTIGUOUS', 'C', 'CONTIGUOUS'):
            return self.c_contiguous
        if k in ('F_CONTIGUOUS', 'F', 'FORTRAN'):
            return self.f_contiguous
        if k == 'WRITEABLE':
            return self.writeable
        if k == 'FNC':
            return self.fnc
        return False

def _coerce_native_boxed_operand(x):
    if not isinstance(x, _ObjectArray):
        return x
    try:
        arr = _native.array_with_dtype(x._data, x._dtype)
        if arr.shape != x._shape:
            arr = arr.reshape(x._shape)
        return arr
    except (TypeError, ValueError):
        return x

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
        elif (isinstance(self._data, list) and len(self._data) > 0
              and all(isinstance(x, (list, tuple)) for x in self._data)
              and len(set(len(x) for x in self._data)) == 1):
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
        elif (
            isinstance(self._dtype, str)
            and self._dtype.lstrip("<>=|").startswith("S")
            and self._dtype.lstrip("<>=|")[1:].isdigit()
        ):
            self._itemsize = int(self._dtype.lstrip("<>=|")[1:])
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
                d = _np.dtype(self._dtype)
                # Ensure unicode string dtypes have correct .str with length
                if d.name == 'str' and d.str == '<U':
                    raw = self._dtype
                    stripped = raw.lstrip('<>=|')
                    if stripped.startswith('U') and len(stripped) > 1 and stripped[1:].isdigit():
                        ulen = int(stripped[1:])
                        d.str = '<U' + str(ulen)
                        d.itemsize = 4 * ulen
                        d.kind = 'U'
                        d.char = 'U'
                if d.name == 'bytes' and d.str == '|S0':
                    raw = self._dtype
                    stripped = raw.lstrip('<>=|')
                    if stripped.startswith('S') and len(stripped) > 1 and stripped[1:].isdigit():
                        blen = int(stripped[1:])
                        d.str = '|S' + str(blen)
                        d.itemsize = blen
                        d.kind = 'S'
                        d.char = 'S'
                return d
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
    def real(self):
        """Return real part of each element."""
        dt = str(self._dtype) if hasattr(self._dtype, '__str__') else self._dtype
        if 'complex' in str(dt):
            vals = [v.real if isinstance(v, complex) else float(v) for v in self._data]
            return _native.array(vals).reshape(list(self._shape)) if len(self._shape) > 1 else _native.array(vals)
        return self
    @real.setter
    def real(self, value):
        """Set real part of each element."""
        dt = str(self._dtype) if hasattr(self._dtype, '__str__') else self._dtype
        if 'complex' in str(dt):
            if hasattr(value, 'flat'):
                reals = [float(v) for v in value.flat]
            elif hasattr(value, '__iter__'):
                reals = [float(v) for v in value]
            else:
                reals = [float(value)] * len(self._data)
            for i, r in enumerate(reals):
                v = self._data[i]
                im = v.imag if isinstance(v, complex) else 0.0
                self._data[i] = complex(r, im)
        else:
            # For real arrays, copy values in
            if hasattr(value, 'flat'):
                vals = [float(v) for v in value.flat]
            elif hasattr(value, '__iter__'):
                vals = [float(v) for v in value]
            else:
                vals = [float(value)] * len(self._data)
            for i, v in enumerate(vals):
                self._data[i] = v
    @property
    def imag(self):
        """Return imaginary part of each element."""
        dt = str(self._dtype) if hasattr(self._dtype, '__str__') else self._dtype
        if 'complex' in str(dt):
            vals = [v.imag if isinstance(v, complex) else 0.0 for v in self._data]
            return _native.array(vals).reshape(list(self._shape)) if len(self._shape) > 1 else _native.array(vals)
        import numpy as _np
        return _np.zeros(self._shape)
    @imag.setter
    def imag(self, value):
        """Set imaginary part of each element."""
        dt = str(self._dtype) if hasattr(self._dtype, '__str__') else self._dtype
        if 'complex' in str(dt):
            if hasattr(value, 'flat'):
                imags = [float(v) for v in value.flat]
            elif hasattr(value, '__iter__'):
                imags = [float(v) for v in value]
            else:
                imags = [float(value)] * len(self._data)
            for i, im in enumerate(imags):
                v = self._data[i]
                re = v.real if isinstance(v, complex) else float(v)
                self._data[i] = complex(re, im)
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

    def copy(self):
        native_self = self._native_boxed()
        if isinstance(native_self, ndarray):
            try:
                return native_self.copy()
            except Exception:
                pass
        return _ObjectArray(list(self._data), self._dtype, shape=self._shape, is_fortran=self._is_fortran, itemsize=self._itemsize)
    def reshape(self, *shape):
        if len(shape) == 1 and isinstance(shape[0], (list, tuple)):
            shape = tuple(shape[0])
        else:
            shape = tuple(shape)
        n = 1
        for s in shape:
            n *= s
        if n != len(self._data):
            raise ValueError("cannot reshape array of size {} into shape {}".format(len(self._data), shape))
        native_self = self._native_boxed()
        if isinstance(native_self, ndarray):
            try:
                return native_self.reshape(shape)
            except Exception:
                pass
        return _ObjectArray(list(self._data), self._dtype, shape=shape, itemsize=self._itemsize)
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
        native_self = self._native_boxed()
        if isinstance(native_self, ndarray):
            try:
                return native_self.astype(dtype_str)
            except Exception:
                pass
        return _ObjectArray(list(self._data), dtype_str, shape=self._shape, is_fortran=self._is_fortran)
    def flatten(self):
        native_self = self._native_boxed()
        if isinstance(native_self, ndarray):
            try:
                return native_self.flatten()
            except Exception:
                pass
        return self
    def ravel(self):
        native_self = self._native_boxed()
        if isinstance(native_self, ndarray):
            try:
                return native_self.ravel()
            except Exception:
                pass
        return self
    def tolist(self):
        if self._shape == ():
            return self._data[0] if self._data else None
        return list(self._data)
    def all(self):
        native_self = self._native_boxed()
        if isinstance(native_self, ndarray):
            try:
                return native_self.all()
            except Exception:
                pass
        return all(self._data)
    def any(self):
        native_self = self._native_boxed()
        if isinstance(native_self, ndarray):
            try:
                return native_self.any()
            except Exception:
                pass
        return any(self._data)
    def __len__(self): return len(self._data)
    def sort(self, axis=-1, kind=None, order=None):
        """Sort in-place. For complex arrays, sort by real part then imag part."""
        native_self = self._native_boxed()
        if isinstance(native_self, ndarray):
            try:
                native_self.sort(axis=axis, kind=kind, order=order)
                self._data = native_self.flatten().tolist()
                self._shape = native_self.shape
                self._ndim = len(self._shape)
                return None
            except Exception:
                pass
        def _sort_key(v):
            if isinstance(v, (tuple, list)) and len(v) == 2:
                return (v[0], v[1])
            if isinstance(v, complex):
                return (v.real, v.imag)
            try:
                return (float(v), 0.0)
            except (TypeError, ValueError):
                return (0.0, 0.0)
        self._data.sort(key=_sort_key)
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
        native_self = self._native_boxed()
        if isinstance(native_self, ndarray):
            try:
                return native_self[key]
            except Exception:
                pass
        if isinstance(key, tuple):
            _ball = __import__("builtins").all
            if _ball(isinstance(k, int) for k in key):
                if len(key) == len(self._shape):
                    # Full indexing: return scalar
                    idx = self._flat_index(key)
                    return self._wrap_element(self._data[idx])
                else:
                    # Partial indexing: apply keys one at a time
                    result = self
                    for k in key:
                        result = result[k]
                    return result
            # Handle tuple with slices (N-D indexing)
            return self._nd_getitem(key)
        if isinstance(key, int) and len(self._shape) > 1:
            # Integer key on N-D array: return sub-array with shape shape[1:]
            if key < 0:
                key += self._shape[0]
            sub_shape = self._shape[1:]
            sub_size = 1
            for s in sub_shape:
                sub_size *= s
            sub_data = self._data[key * sub_size : (key + 1) * sub_size]
            return _ObjectArray(sub_data, self._dtype, shape=sub_shape, itemsize=self._itemsize)
        result = self._data[key]
        if isinstance(key, slice):
            return _ObjectArray(result, self._dtype)
        return self._wrap_element(result)
    def _wrap_element(self, val):
        """Wrap scalar element in numpy scalar type when appropriate."""
        dt = str(self._dtype) if not isinstance(self._dtype, str) else self._dtype
        if dt == 'void' or dt.startswith('V'):
            from numpy._core_types import _NumpyVoidScalar
            return _NumpyVoidScalar(val)
        if isinstance(val, complex) and 'complex' in dt:
            from numpy._core_types import _NumpyComplexScalar
            dn = 'complex64' if '64' in dt else 'complex128'
            return _NumpyComplexScalar(val, dn)
        return val
    def __setitem__(self, key, value):
        if key is ...:
            # Ellipsis: set all elements to value
            n = len(self._data)
            if isinstance(value, (list, tuple)):
                self._data[:] = list(value)
            elif isinstance(value, _ObjectArray):
                self._data[:] = value._data[:]
            else:
                self._data[:] = [value] * n
            return
        if isinstance(key, tuple):
            _ball = __import__("builtins").all
            if _ball(isinstance(k, int) for k in key):
                idx = self._flat_index(key)
                self._data[idx] = value
                return
            # Handle tuple with slices for __setitem__
            self._nd_setitem(key, value)
            return
        if isinstance(key, slice):
            if isinstance(value, (list, tuple)):
                self._data[key] = value
            elif isinstance(value, _ObjectArray):
                self._data[key] = value._data
            else:
                self._data[key] = [value] * len(range(*key.indices(len(self._data))))
        else:
            self._data[key] = value
    def _flat_index(self, idx):
        if len(idx) != self._ndim:
            raise IndexError("index has wrong number of dimensions")
        flat = 0
        mult = 1
        for i in range(self._ndim - 1, -1, -1):
            k = idx[i]
            dim = self._shape[i]
            if k < 0:
                k += dim
            if k < 0 or k >= dim:
                raise IndexError("index out of bounds")
            flat += k * mult
            mult *= dim
        return flat

    def _nd_getitem(self, key_tuple):
        """N-D indexing with a tuple of ints/slices."""
        import itertools as _it
        ndim = self._ndim
        shape = list(self._shape)

        # Pad key_tuple with slice(None) for missing trailing dims
        key_list = list(key_tuple)
        while len(key_list) < ndim:
            key_list.append(slice(None))

        # Compute per-dimension index lists and output shape
        dim_indices = []
        out_shape = []
        for i, k in enumerate(key_list):
            if isinstance(k, int):
                d = k if k >= 0 else shape[i] + k
                dim_indices.append([d])
                # int key: dimension removed from output
            elif isinstance(k, slice):
                idxs = list(range(*k.indices(shape[i])))
                dim_indices.append(idxs)
                out_shape.append(len(idxs))
            else:
                raise TypeError("unsupported index type: %s" % type(k))

        # Gather output data
        out_data = []
        for multi_idx in _it.product(*dim_indices):
            flat_idx = 0
            mult = 1
            for d in range(ndim - 1, -1, -1):
                flat_idx += multi_idx[d] * mult
                mult *= shape[d]
            out_data.append(self._data[flat_idx])

        if not out_shape:
            return self._wrap_element(out_data[0]) if out_data else None
        return _ObjectArray(out_data, self._dtype, shape=tuple(out_shape))
    def _nd_setitem(self, key_tuple, value):
        """N-D __setitem__ with a tuple of ints/slices."""
        import itertools as _it
        ndim = self._ndim
        shape = list(self._shape)

        key_list = list(key_tuple)
        while len(key_list) < ndim:
            key_list.append(slice(None))

        dim_indices = []
        for i, k in enumerate(key_list):
            if isinstance(k, int):
                d = k if k >= 0 else shape[i] + k
                dim_indices.append([d])
            elif isinstance(k, slice):
                dim_indices.append(list(range(*k.indices(shape[i]))))
            else:
                raise TypeError("unsupported index type: %s" % type(k))

        # Get value as a flat list
        if isinstance(value, _ObjectArray):
            vdata = value._data
        elif isinstance(value, (list, tuple)):
            vdata = list(value)
        else:
            vdata = None  # scalar

        all_combos = list(_it.product(*dim_indices))
        for vi, multi_idx in enumerate(all_combos):
            flat_idx = 0
            mult = 1
            for d in range(ndim - 1, -1, -1):
                flat_idx += multi_idx[d] * mult
                mult *= shape[d]
            if vdata is not None:
                self._data[flat_idx] = vdata[vi]
            else:
                self._data[flat_idx] = value

    def _to_bool_array(self, data):
        return _native.array([1.0 if x else 0.0 for x in data]).astype("bool")
    def _broadcast_other(self, other):
        """Get other's data list, broadcast scalar/single-element to match self size."""
        if isinstance(other, _ObjectArray):
            odata = other._data
            if len(odata) == 1 and len(self._data) > 1:
                odata = odata * len(self._data)
            return odata
        if isinstance(other, ndarray):
            odata = other.flatten().tolist()
            if len(odata) == 1 and len(self._data) > 1:
                odata = odata * len(self._data)
            return odata
        return None
    def _native_boxed(self):
        return _coerce_native_boxed_operand(self)
    def _sync_from_native(self, native_arr):
        self._data = native_arr.flatten().tolist()
        self._shape = native_arr.shape
        self._ndim = len(self._shape)
        self._dtype = str(native_arr.dtype)
        return self
    def _supports_native_boxed_ops(self):
        return (
            isinstance(self._dtype, str)
            and (
                self._dtype == "object"
                or self._dtype.startswith("datetime64")
                or self._dtype.startswith("timedelta64")
            )
        )
    def _native_binary(self, other, op_name):
        if not self._supports_native_boxed_ops():
            return None
        native_self = self._native_boxed()
        if not isinstance(native_self, ndarray):
            return None
        native_other = _coerce_native_boxed_operand(other)
        try:
            return getattr(native_self, op_name)(native_other)
        except Exception:
            return None
    def __eq__(self, other):
        native_self = self._native_boxed()
        if isinstance(native_self, ndarray):
            return native_self == _coerce_native_boxed_operand(other)
        odata = self._broadcast_other(other)
        if odata is not None:
            return self._to_bool_array([a == b for a, b in zip(self._data, odata)])
        if other is None or isinstance(other, (int, float, complex, str, bytes)):
            return self._to_bool_array([x == other for x in self._data])
        return NotImplemented
    def __ne__(self, other):
        native_self = self._native_boxed()
        if isinstance(native_self, ndarray):
            return native_self != _coerce_native_boxed_operand(other)
        odata = self._broadcast_other(other)
        if odata is not None:
            return self._to_bool_array([a != b for a, b in zip(self._data, odata)])
        if other is None or isinstance(other, (int, float, complex, str, bytes)):
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
        native_self = self._native_boxed()
        if isinstance(native_self, ndarray):
            return native_self <= _coerce_native_boxed_operand(other)
        if isinstance(other, _ObjectArray):
            return self._to_bool_array([self._cmp_le(a, b) for a, b in zip(self._data, other._data)])
        if isinstance(other, ndarray):
            return self._to_bool_array([self._cmp_le(a, b) for a, b in zip(self._data, other.flatten().tolist())])
        return self._to_bool_array([self._cmp_le(x, other) for x in self._data])
    def __lt__(self, other):
        native_self = self._native_boxed()
        if isinstance(native_self, ndarray):
            return native_self < _coerce_native_boxed_operand(other)
        if isinstance(other, _ObjectArray):
            return self._to_bool_array([self._cmp_lt(a, b) for a, b in zip(self._data, other._data)])
        if isinstance(other, ndarray):
            return self._to_bool_array([self._cmp_lt(a, b) for a, b in zip(self._data, other.flatten().tolist())])
        return self._to_bool_array([self._cmp_lt(x, other) for x in self._data])
    def __ge__(self, other):
        native_self = self._native_boxed()
        if isinstance(native_self, ndarray):
            return native_self >= _coerce_native_boxed_operand(other)
        if isinstance(other, _ObjectArray):
            return self._to_bool_array([self._cmp_ge(a, b) for a, b in zip(self._data, other._data)])
        if isinstance(other, ndarray):
            return self._to_bool_array([self._cmp_ge(a, b) for a, b in zip(self._data, other.flatten().tolist())])
        return self._to_bool_array([self._cmp_ge(x, other) for x in self._data])
    def __gt__(self, other):
        native_self = self._native_boxed()
        if isinstance(native_self, ndarray):
            return native_self > _coerce_native_boxed_operand(other)
        if isinstance(other, _ObjectArray):
            return self._to_bool_array([self._cmp_gt(a, b) for a, b in zip(self._data, other._data)])
        if isinstance(other, ndarray):
            return self._to_bool_array([self._cmp_gt(a, b) for a, b in zip(self._data, other.flatten().tolist())])
        return self._to_bool_array([self._cmp_gt(x, other) for x in self._data])
    def __sub__(self, other):
        native = self._native_binary(other, "__sub__")
        if native is not None:
            return native
        if isinstance(other, _ObjectArray):
            return _ObjectArray([a - b for a, b in zip(self._data, other._data)], self._dtype)
        return _ObjectArray([x - other for x in self._data], self._dtype)
    def __rsub__(self, other):
        if not self._supports_native_boxed_ops():
            return _ObjectArray([other - x for x in self._data], self._dtype)
        native_self = self._native_boxed()
        if isinstance(native_self, ndarray):
            native_other = _coerce_native_boxed_operand(other)
            try:
                return native_other - native_self
            except Exception:
                pass
        return _ObjectArray([other - x for x in self._data], self._dtype)
    def __mul__(self, other):
        native = self._native_binary(other, "__mul__")
        if native is not None:
            return native
        if isinstance(other, _ObjectArray):
            return _ObjectArray([a * b for a, b in zip(self._data, other._data)], self._dtype)
        if isinstance(other, int) and self._dtype == "object":
            return _ObjectArray(self._data * other, self._dtype)
        return _ObjectArray([x * other for x in self._data], self._dtype)
    def __rmul__(self, other):
        if not self._supports_native_boxed_ops():
            return self.__mul__(other)
        native_self = self._native_boxed()
        if isinstance(native_self, ndarray):
            native_other = _coerce_native_boxed_operand(other)
            try:
                return native_other * native_self
            except Exception:
                pass
        return self.__mul__(other)
    def __add__(self, other):
        native = self._native_binary(other, "__add__")
        if native is not None:
            return native
        if isinstance(other, _ObjectArray):
            return _ObjectArray([a + b for a, b in zip(self._data, other._data)], self._dtype)
        return _ObjectArray([x + other for x in self._data], self._dtype)
    def __radd__(self, other):
        if not self._supports_native_boxed_ops():
            return self.__add__(other)
        native_self = self._native_boxed()
        if isinstance(native_self, ndarray):
            native_other = _coerce_native_boxed_operand(other)
            try:
                return native_other + native_self
            except Exception:
                pass
        return self.__add__(other)
    def conjugate(self):
        if not self._supports_native_boxed_ops():
            return _ObjectArray([x.conjugate() if hasattr(x, 'conjugate') else x for x in self._data], self._dtype)
        native_self = self._native_boxed()
        if isinstance(native_self, ndarray):
            try:
                return native_self.conj()
            except Exception:
                pass
        return _ObjectArray([x.conjugate() if hasattr(x, 'conjugate') else x for x in self._data], self._dtype)
    def conj(self):
        return self.conjugate()
    def sum(self, axis=None, keepdims=False, **kwargs):
        if not self._supports_native_boxed_ops():
            _bsum = __import__("builtins").sum
            return _bsum(self._data)
        native_self = self._native_boxed()
        if isinstance(native_self, ndarray):
            try:
                return native_self.sum(axis=axis, keepdims=keepdims, **kwargs)
            except Exception:
                pass
        _bsum = __import__("builtins").sum
        return _bsum(self._data)
    def prod(self, axis=None, keepdims=False, **kwargs):
        if not self._supports_native_boxed_ops():
            r = 1
            for x in self._data:
                r *= x
            return r
        native_self = self._native_boxed()
        if isinstance(native_self, ndarray):
            try:
                return native_self.prod(axis=axis, keepdims=keepdims, **kwargs)
            except Exception:
                pass
        r = 1
        for x in self._data:
            r *= x
        return r
    def mean(self, axis=None, keepdims=False, **kwargs):
        if not self._supports_native_boxed_ops():
            _bsum = __import__("builtins").sum
            return _bsum(self._data) / len(self._data)
        native_self = self._native_boxed()
        if isinstance(native_self, ndarray):
            try:
                return native_self.mean(axis=axis, keepdims=keepdims, **kwargs)
            except Exception:
                pass
        _bsum = __import__("builtins").sum
        return _bsum(self._data) / len(self._data)
    def var(self, axis=None, ddof=0, keepdims=False, **kwargs):
        if not self._supports_native_boxed_ops():
            m = self.mean()
            _babs = __import__("builtins").abs
            _bsum = __import__("builtins").sum
            return _bsum(_babs(x - m) ** 2 for x in self._data) / (len(self._data) - ddof)
        native_self = self._native_boxed()
        if isinstance(native_self, ndarray):
            try:
                return native_self.var(axis=axis, ddof=ddof, keepdims=keepdims, **kwargs)
            except Exception:
                pass
        m = self.mean()
        _babs = __import__("builtins").abs
        _bsum = __import__("builtins").sum
        return _bsum(_babs(x - m) ** 2 for x in self._data) / (len(self._data) - ddof)
    def std(self, axis=None, ddof=0, keepdims=False, **kwargs):
        if not self._supports_native_boxed_ops():
            return self.var(axis, ddof, keepdims) ** 0.5
        native_self = self._native_boxed()
        if isinstance(native_self, ndarray):
            try:
                return native_self.std(axis=axis, ddof=ddof, keepdims=keepdims, **kwargs)
            except Exception:
                pass
        return self.var(axis, ddof, keepdims) ** 0.5
    def __abs__(self):
        native_self = self._native_boxed()
        if isinstance(native_self, ndarray):
            try:
                return abs(native_self)
            except Exception:
                pass
        _babs = __import__("builtins").abs
        result = [_babs(x) for x in self._data]
        # abs of complex returns float, so return a real ndarray
        if self._dtype in ('complex64', 'complex128'):
            from ._creation import array as _array
            import numpy as _np
            arr = _array(result, dtype='float64')
            if len(self._shape) > 1:
                arr = arr.reshape(self._shape)
            return arr
        return _ObjectArray(result, self._dtype)
    def __pow__(self, other):
        native = self._native_binary(other, "__pow__")
        if native is not None:
            return native
        if isinstance(other, _ObjectArray):
            return _ObjectArray([_complex_pow(a, b) for a, b in zip(self._data, other._data)], self._dtype)
        if isinstance(other, ndarray):
            if other.ndim == 0:
                scalar = other.item() if hasattr(other, 'item') else float(other)
                return _ObjectArray([_complex_pow(x, scalar) for x in self._data], self._dtype)
            other_list = other.flatten().tolist()
            return _ObjectArray([_complex_pow(a, b) for a, b in zip(self._data, other_list)], self._dtype)
        if hasattr(other, '__iter__') and not isinstance(other, str):
            other_list = list(other) if not isinstance(other, list) else other
            return _ObjectArray([_complex_pow(a, b) for a, b in zip(self._data, other_list)], self._dtype)
        return _ObjectArray([_complex_pow(x, other) for x in self._data], self._dtype)
    def __rpow__(self, other):
        if not self._supports_native_boxed_ops():
            if isinstance(other, _ObjectArray):
                return _ObjectArray([_complex_pow(b, a) for a, b in zip(self._data, other._data)], self._dtype)
            if isinstance(other, ndarray):
                if other.ndim == 0:
                    scalar = other.item() if hasattr(other, 'item') else float(other)
                    return _ObjectArray([_complex_pow(scalar, x) for x in self._data], self._dtype)
                other_list = other.flatten().tolist()
                return _ObjectArray([_complex_pow(b, a) for a, b in zip(self._data, other_list)], self._dtype)
            return _ObjectArray([_complex_pow(other, x) for x in self._data], self._dtype)
        native_self = self._native_boxed()
        if isinstance(native_self, ndarray):
            native_other = _coerce_native_boxed_operand(other)
            try:
                return native_other ** native_self
            except Exception:
                pass
        if isinstance(other, _ObjectArray):
            return _ObjectArray([_complex_pow(b, a) for a, b in zip(self._data, other._data)], self._dtype)
        if isinstance(other, ndarray):
            if other.ndim == 0:
                scalar = other.item() if hasattr(other, 'item') else float(other)
                return _ObjectArray([_complex_pow(scalar, x) for x in self._data], self._dtype)
            other_list = other.flatten().tolist()
            return _ObjectArray([_complex_pow(b, a) for a, b in zip(self._data, other_list)], self._dtype)
        return _ObjectArray([_complex_pow(other, x) for x in self._data], self._dtype)
    def __truediv__(self, other):
        native = self._native_binary(other, "__truediv__")
        if native is not None:
            return native
        if isinstance(other, _ObjectArray):
            return _ObjectArray([a / b for a, b in zip(self._data, other._data)], self._dtype)
        return _ObjectArray([x / other for x in self._data], self._dtype)
    def clip(self, a_min=None, a_max=None, out=None, **kwargs):
        _valid_castings = ('no', 'equiv', 'safe', 'same_kind', 'unsafe')
        if 'casting' in kwargs:
            c = kwargs['casting']
            if c not in _valid_castings:
                raise ValueError("casting must be one of 'no', 'equiv', 'safe', 'same_kind', or 'unsafe'")
        native_self = self._native_boxed()
        if isinstance(native_self, ndarray):
            return native_self.clip(
                _coerce_native_boxed_operand(a_min),
                _coerce_native_boxed_operand(a_max),
                out=out,
            )
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
    def view(self, dtype=None):
        """View the array as a different type. Supports chararray."""
        if dtype is None:
            return self
        # Check if dtype is the chararray class
        type_name = getattr(dtype, '__name__', '')
        if type_name == 'chararray':
            # Convert _ObjectArray to a proper ndarray of strings, then wrap
            import numpy as _np

            # Flatten nested data to 1D list of scalars
            def _flatten(d):
                if isinstance(d, (list, tuple)):
                    result = []
                    for item in d:
                        result.extend(_flatten(item))
                    return result
                return [d]

            flat = _flatten(self._data)
            # Create a 1D ndarray from flat items, then reshape to original shape
            arr = _np.array(flat)
            if self._ndim > 1 and len(flat) > 0:
                arr = arr.reshape(list(self._shape))
            return dtype._from_array(arr)
        native_self = self._native_boxed()
        if isinstance(native_self, ndarray):
            try:
                return native_self.view(dtype)
            except Exception:
                pass
        # Default: try to return self with new dtype
        return _ObjectArray(list(self._data), str(dtype) if not isinstance(dtype, str) else dtype, shape=self._shape)

    def item(self, *args):
        """Return element at given index as a Python scalar."""
        native_self = self._native_boxed()
        if isinstance(native_self, ndarray):
            try:
                return native_self.item(*args)
            except Exception:
                pass
        if not args:
            if self.size != 1:
                raise ValueError("can only convert an array of size 1 to a Python scalar")
            return self._data[0]
        if len(args) == 1:
            idx = args[0]
            if isinstance(idx, tuple):
                flat = self._flat_index(idx)
                return self._data[flat]
            idx = int(idx)
            if idx < 0:
                idx += self.size
            if idx < 0 or idx >= self.size:
                raise IndexError("index {} is out of bounds for size {}".format(args[0], self.size))
            return self._data[idx]
        raise ValueError("incorrect number of indices for array")

    def take(self, indices, axis=None, out=None, mode='raise'):
        """Take elements from array along an axis."""
        native_self = self._native_boxed()
        if isinstance(native_self, ndarray):
            try:
                return native_self.take(indices, axis=axis, out=out, mode=mode)
            except Exception:
                pass
        import numpy as _np
        if not isinstance(indices, (list, tuple)):
            indices = [int(indices)]
        else:
            indices = [int(i) for i in indices]
        if axis is not None:
            dim_size = self._shape[axis]
        else:
            dim_size = self.size
        result = []
        for idx in indices:
            if mode == 'wrap':
                if dim_size == 0:
                    raise IndexError("cannot take from a 0-length dimension")
                idx = idx % dim_size
            elif mode == 'clip':
                if dim_size == 0:
                    raise IndexError("cannot take from a 0-length dimension")
                idx = max(0, min(idx, dim_size - 1))
            else:
                if idx < 0:
                    idx += dim_size
                if idx < 0 or idx >= dim_size:
                    raise IndexError("index {} is out of bounds for axis with size {}".format(idx, dim_size))
            result.append(self._data[idx])
        return _ObjectArray(result, self._dtype)

    def put(self, indices, values, mode='raise'):
        """Replace specified elements of the array."""
        if not isinstance(indices, (list, tuple)):
            indices = [int(indices)]
        if not isinstance(values, (list, tuple)):
            values = [values]
        n = self.size
        nv = len(values)
        for i_idx, idx in enumerate(indices):
            idx = int(idx)
            if mode == 'wrap':
                if n == 0:
                    return
                idx = idx % n
            elif mode == 'clip':
                if n == 0:
                    return
                idx = max(0, min(idx, n - 1))
            else:
                if idx < 0:
                    idx += n
                if idx < 0 or idx >= n:
                    raise IndexError("index {} is out of bounds for axis 0 with size {}".format(indices[i_idx], n))
            self._data[idx] = values[i_idx % nv]

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
    if isinstance(data, _ObjectArray):
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
    if isinstance(data, _ObjectArray):
        return None  # ObjectArrays (e.g. complex) need special handling
    if isinstance(data, ndarray):
        flat = data.flatten().tolist()
        # Check if any element is a tuple (complex number from Rust backend)
        if flat and isinstance(flat[0], tuple):
            return None  # Let _array_core handle complex arrays via stack path
        return flat
    if isinstance(data, (int, float, bool)):
        return [float(data)]
    if isinstance(data, complex):
        return None  # Complex data needs special handling
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


def _all_bools_nested(data):
    """Check if all leaf elements in a nested list/tuple are bools."""
    if isinstance(data, bool):
        return True
    if isinstance(data, ndarray):
        return str(data.dtype) == 'bool'
    if isinstance(data, (int, float)):
        return False
    if isinstance(data, (list, tuple)):
        return all(_all_bools_nested(x) for x in data)
    return False


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
    if dt in ('bytes', 'str', 'void', 'S1'):
        return True
    if isinstance(dt, str):
        s = dt.lstrip('<>=|')
        if s.startswith('V'):
            return True
        if s.startswith('S') or s.startswith('U'):
            return True
    return False

_CLIP_UNSET = object()
