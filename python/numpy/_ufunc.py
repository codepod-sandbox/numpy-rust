"""NumPy ufunc class and function wrapping registration."""
from _numpy_native import ndarray
from ._helpers import _copy_into
from ._creation import asarray, array
from ._math import (
    add, subtract, multiply, divide, true_divide, floor_divide,
    power, remainder, fmod, maximum, minimum, fmax, fmin,
    greater, less, equal, not_equal, greater_equal, less_equal,
    arctan2, hypot, copysign, ldexp, heaviside, nextafter,
    sin, cos, tan, arcsin, arccos, arctan,
    sinh, cosh, tanh,
    exp, exp2, log, log2, log10,
    sqrt, cbrt, square, reciprocal, negative, positive, absolute,
    sign, floor, ceil, rint, trunc,
    deg2rad, rad2deg, signbit,
    isnan, isinf, isfinite,
    modf,
)
from ._bitwise import (
    logical_and, logical_or, logical_xor, logical_not,
    bitwise_and, bitwise_or, bitwise_xor, bitwise_not,
    left_shift, right_shift,
)
from ._reductions import sum, cumsum, prod, cumprod, max, min, all, any
from ._manipulation import expand_dims, take, squeeze, stack

__all__ = [
    'ufunc',
    '_ufunc_reconstruct',
    # Wrapped binary ufuncs (overwrite plain function names with ufunc objects)
    'add', 'subtract', 'multiply', 'divide', 'true_divide', 'floor_divide',
    'power', 'remainder', 'mod', 'fmod', 'maximum', 'minimum', 'fmax', 'fmin',
    'logical_and', 'logical_or', 'logical_xor',
    'bitwise_and', 'bitwise_or', 'bitwise_xor', 'left_shift', 'right_shift',
    'greater', 'less', 'equal', 'not_equal', 'greater_equal', 'less_equal',
    'arctan2', 'hypot', 'copysign', 'ldexp', 'heaviside', 'nextafter', 'modf',
    # Wrapped unary ufuncs
    'sin', 'cos', 'tan', 'arcsin', 'arccos', 'arctan',
    'sinh', 'cosh', 'tanh',
    'exp', 'exp2', 'log', 'log2', 'log10',
    'sqrt', 'cbrt', 'square', 'reciprocal', 'negative', 'positive',
    'absolute', 'abs', 'sign',
    'floor', 'ceil', 'rint', 'trunc',
    'deg2rad', 'rad2deg', 'signbit',
    'logical_not', 'isnan', 'isinf', 'isfinite',
    'bitwise_not', 'invert',
    'bitwise_count',
    'gcd', 'lcm', 'divmod',
]

# Type signature constants for ufunc.types attribute
_FLOAT_BINARY_TYPES   = ['ff->f', 'dd->d']
_FLOAT_UNARY_TYPES    = ['f->f', 'd->d']
_INT_BINARY_TYPES     = ['bb->b', 'BB->B', 'hh->h', 'HH->H',
                          'ii->i', 'II->I', 'll->l', 'LL->L']
_INT_UNARY_TYPES      = ['b->b', 'B->B', 'h->h', 'H->H',
                          'i->i', 'I->I', 'l->l', 'L->L']
_NUMERIC_BINARY_TYPES = _INT_BINARY_TYPES + _FLOAT_BINARY_TYPES
_NUMERIC_UNARY_TYPES  = _INT_UNARY_TYPES + _FLOAT_UNARY_TYPES
_CMP_BINARY_TYPES     = ['bb->?', 'BB->?', 'hh->?', 'HH->?',
                          'ii->?', 'II->?', 'll->?', 'LL->?',
                          'ff->?', 'dd->?']

_UFUNC_SENTINEL = object()  # private token — only _make_ufunc passes it
_REDUCE_NOVALUE = object()  # sentinel distinguishes "no initial" from initial=None


def _apply_where_mask(result, where_arr, fill, existing=None):
    """Apply boolean mask: keep result at True, fill/existing at False."""
    flat_r = result.ravel().tolist()
    if where_arr.ndim == 0:
        flat_w = [bool(where_arr.flat[0])] * len(flat_r)
    else:
        flat_w = where_arr.ravel().tolist()
    if len(flat_w) < len(flat_r):
        repeats = (len(flat_r) + len(flat_w) - 1) // len(flat_w)
        flat_w = (flat_w * repeats)[:len(flat_r)]
    if existing is not None:
        flat_e = asarray(existing).ravel().tolist()
        flat_out = [r if w else e for r, w, e in zip(flat_r, flat_w, flat_e)]
    else:
        flat_out = [r if w else fill for r, w in zip(flat_r, flat_w)]
    return array(flat_out, dtype=result.dtype).reshape(result.shape)


def _ufunc_reconstruct(module_name, ufunc_name):
    """Deserialize a ufunc by module + name lookup."""
    import importlib
    for legacy in ('numpy.core', 'numpy._core.umath', 'numpy._core'):
        if module_name.startswith(legacy):
            module_name = 'numpy'
            break
    mod = importlib.import_module(module_name)
    return getattr(mod, ufunc_name)


def _check_out_shape(out, result):
    """Raise ValueError if out.shape is incompatible with result.shape.

    We allow: exact match, or result is scalar/0-d and out has size 1
    (matches NumPy's behaviour of allowing e.g. out=array([0.]) for a scalar
    reduction result).
    """
    if not hasattr(out, 'shape') or not hasattr(result, 'shape'):
        return
    if out.shape == result.shape:
        return
    # Allow scalar result into size-1 out (NumPy broadcasts in this case)
    r_size = getattr(result, 'size', 1) if result.shape != () else 1
    o_size = getattr(out, 'size', 1)
    if r_size == 1 and o_size == 1:
        return
    raise ValueError(
        "out array has wrong shape: expected {}, got {}".format(
            result.shape, out.shape))


def _set_at(a, idx, result):
    """Write result into a[idx], preserving a's dtype."""
    if hasattr(a, 'dtype'):
        try:
            if result.ndim == 0 or result.size == 1:
                a[idx] = a.dtype.type(result.flat[0])
            else:
                a[idx] = result.astype(a.dtype)
        except (TypeError, ValueError):
            if result.ndim == 0 or result.size == 1:
                a[idx] = result.flat[0]
            else:
                a[idx] = result
    else:
        a[idx] = result.flat[0] if result.size == 1 else result


# Mapping from dtype strings / type chars / Python types to numpy type codes
_DTYPE_TO_TYPECODE = {
    # Integer types (signed)
    'int8': 'b', 'int16': 'h', 'int32': 'i', 'int64': 'l', 'intp': 'l',
    # Integer types (unsigned)
    'uint8': 'B', 'uint16': 'H', 'uint32': 'I', 'uint64': 'L',
    # Float types
    'float16': 'e', 'float32': 'f', 'float64': 'd', 'float128': 'g',
    'longdouble': 'g',
    # Complex types
    'complex64': 'F', 'complex128': 'D', 'complex256': 'G',
    # Bool, object, bytes
    'bool': '?', 'bool_': '?', 'object': 'O', 'object_': 'O',
    'bytes_': 'S', 'str_': 'U', 'bytes': 'S', 'str': 'U',
    # Short codes
    'e': 'e', 'f': 'f', 'd': 'd', 'g': 'g',
    'b': 'b', 'h': 'h', 'i': 'i', 'l': 'l',
    'B': 'B', 'H': 'H', 'I': 'I', 'L': 'L', 'Q': 'L',
    '?': '?', 'O': 'O', 'F': 'F', 'D': 'D', 'G': 'G',
}

# Python builtin types to numpy type codes
_PYTYPE_TO_TYPECODE = {
    bool: '?',
    int: 'l',
    float: 'd',
    complex: 'D',
    object: 'O',
    bytes: 'S',
    str: 'U',
}


def _type_to_code(t):
    """Convert a type specifier to a numpy type code char, or None if any."""
    if t is None:
        return None
    # Python builtin types
    if t in _PYTYPE_TO_TYPECODE:
        return _PYTYPE_TO_TYPECODE[t]
    # numpy dtype-like objects
    if hasattr(t, 'name'):
        return _DTYPE_TO_TYPECODE.get(t.name)
    # Single character string
    if isinstance(t, str):
        # Could be 'float64', 'e', etc.
        return _DTYPE_TO_TYPECODE.get(t)
    return None


def _check_signature_types(ufunc_obj, sig_spec):
    """Validate signature type tuple against available loops.

    sig_spec is a tuple of (nin+nout) type specifiers, where None means "any".
    Raises TypeError if no compatible loop exists.
    """
    if not isinstance(sig_spec, (list, tuple)):
        return  # string signatures are not validated here
    if not ufunc_obj.types:
        return  # no type info, can't validate

    # Convert specifiers to codes (None = wildcard)
    codes = [_type_to_code(t) for t in sig_spec]
    nin = ufunc_obj.nin
    nout = ufunc_obj.nout

    # Check if len matches nin+nout
    if len(codes) != nin + nout:
        return  # wrong length, let numpy handle it

    # Try to find a matching loop
    for loop in ufunc_obj.types:
        # Parse loop: e.g. 'edd->d' or 'bb->b'
        if '->' in loop:
            parts = loop.split('->')
            in_codes = parts[0]
            out_codes = parts[1]
        else:
            continue

        if len(in_codes) != nin or len(out_codes) != nout:
            continue

        all_codes = list(in_codes) + list(out_codes)
        match = True
        for wanted, available in zip(codes, all_codes):
            if wanted is None:
                continue  # wildcard - matches anything
            if wanted != available:
                match = False
                break
        if match:
            return  # found a matching loop

    # No matching loop found
    raise TypeError(
        "ufunc '{}' does not support the signature {}".format(
            ufunc_obj.__name__, sig_spec
        )
    )


class ufunc:
    """Universal function wrapper with reduce/accumulate/outer/reduceat/at."""

    def __init__(self, *args, **kwargs):
        # If _create already initialized us (has _func set), nothing to do.
        if hasattr(self, '_func'):
            return
        # Public constructor path.
        if args and callable(args[0]):
            func = args[0]
            nin  = int(kwargs.get('nin', 1))
            nout = int(kwargs.get('nout', 1))
            name = kwargs.get('name', getattr(func, '__name__', 'ufunc'))
        elif args and isinstance(args[0], str):
            _n   = args[0]
            nin  = int(args[1]) if len(args) > 1 else int(kwargs.get('nin', 1))
            nout = int(args[2]) if len(args) > 2 else int(kwargs.get('nout', 1))
            name = _n
            def func(*a, **kw):
                raise TypeError(
                    "ufunc '{}' is not available in this environment".format(_n))
        else:
            raise TypeError("cannot create 'numpy.ufunc' instances")
        self._func            = func
        self.nin              = nin
        self.nout             = nout
        self.nargs            = nin + nout
        self.identity         = kwargs.get('identity', None)
        self.__name__         = name
        self._reduce_fast     = None
        self._accumulate_fast = None
        _types                = kwargs.get('types', None)
        self.types            = list(_types) if _types is not None else []
        self.ntypes           = len(self.types)
        self.signature        = kwargs.get('signature', None)

    @classmethod
    def _create(cls, func, nin, nout=1, *, name=None, identity=None,
                reduce_fast=None, accumulate_fast=None, types=None):
        """Internal factory — use _make_ufunc() instead of ufunc(...)."""
        obj = cls.__new__(cls)
        obj._func             = func
        obj.nin               = nin
        obj.nout              = nout
        obj.nargs             = nin + nout
        obj.identity          = identity
        obj.__name__          = name or getattr(func, '__name__', 'ufunc')
        obj._reduce_fast      = reduce_fast
        obj._accumulate_fast  = accumulate_fast
        obj.types             = list(types) if types is not None else []
        obj.ntypes            = len(obj.types)
        obj.signature         = None
        return obj

    def __call__(self, *args, **kwargs):
        out = kwargs.pop('out', None)
        _dtype = kwargs.pop('dtype', None)
        _where = kwargs.pop('where', True)
        _casting = kwargs.pop('casting', None)
        kwargs.pop('subok', None)
        kwargs.pop('order', None)
        _sig = kwargs.pop('sig', None)
        _signature = kwargs.pop('signature', None)
        # Detect conflicting keyword arguments (mirrors NumPy behavior)
        if _sig is not None and _signature is not None:
            raise TypeError(
                "cannot specify both 'sig' and 'signature'"
            )
        if _sig is not None and _dtype is not None:
            raise TypeError(
                "cannot specify both 'sig' and 'dtype'"
            )
        if _signature is not None and _dtype is not None:
            raise TypeError(
                "cannot specify both 'signature' and 'dtype'"
            )

        # Validate signature type tuple against available loops
        _effective_sig = _sig if _sig is not None else _signature
        if isinstance(_effective_sig, (list, tuple)):
            _check_signature_types(self, _effective_sig)

        # Check for __array_ufunc__ on inputs — NEP 13 protocol
        # Collect all inputs and outputs that might implement __array_ufunc__
        _all_args = list(args)
        if out is not None:
            _out_tuple = out if isinstance(out, tuple) else (out,)
            _all_args.extend(list(_out_tuple))
        else:
            _out_tuple = ()

        # Find the argument with highest __array_ufunc__ priority
        # Skip ndarray and basic Python types
        _au_candidates = []
        for a in _all_args:
            if isinstance(a, ndarray) or isinstance(a, (int, float, bool, complex)):
                continue
            au = getattr(type(a), '__array_ufunc__', NotImplemented)
            if au is not NotImplemented:
                if au is None:
                    # Opts out of ufunc dispatch
                    return NotImplemented
                _au_candidates.append(a)

        if _au_candidates:
            # Sort by __array_ufunc_priority__ (higher priority first)
            # Subclasses take priority
            _au_candidates.sort(
                key=lambda x: getattr(type(x), '__array_ufunc_priority__', 0),
                reverse=True)
            # Try each candidate
            for candidate in _au_candidates:
                # Rebuild kwargs with 'out' if needed
                _au_kwargs = dict(kwargs)
                if _out_tuple:
                    _au_kwargs['out'] = _out_tuple
                result = candidate.__array_ufunc__(self, '__call__', *args, **_au_kwargs)
                if result is not NotImplemented:
                    return result
            # If all returned NotImplemented, raise TypeError
            raise TypeError(
                f"operand type(s) all returned NotImplemented from "
                f"__array_ufunc__({type(self).__name__}, '__call__', ...)")

        # Check for MaskedArray inputs — delegate to numpy.ma and preserve type
        _has_masked = any(
            hasattr(a, 'filled') and hasattr(a, 'mask') and not isinstance(a, ndarray)
            for a in args
        )

        try:
            result = self._func(*args, **kwargs)
        except TypeError as e:
            if any(a is None for a in args):
                raise TypeError(
                    "loop of ufunc does not support argument 0 of type NoneType "
                    "which has no callable {} method".format(self.__name__)) from None
            raise

        # Don't asarray() MaskedArray results — preserve masked type
        if _has_masked and hasattr(result, 'filled'):
            pass  # keep as MaskedArray
        elif _dtype is not None:
            result = asarray(result).astype(str(_dtype))
        else:
            result = asarray(result)

        # Handle where=
        if _where is not True and not (isinstance(_where, bool) and _where):
            import warnings
            if out is None:
                warnings.warn(
                    "'where' used without 'out': elements at False positions "
                    "are undefined",
                    UserWarning, stacklevel=2)
                where_arr = asarray(_where, dtype='bool')
                result = _apply_where_mask(result, where_arr, fill=0)
            else:
                _out = out[0] if isinstance(out, tuple) else out
                where_arr = asarray(_where, dtype='bool')
                result = _apply_where_mask(result, where_arr, fill=None, existing=_out)
                _copy_into(_out, result)
                return _out

        # If all inputs were scalars (0-d or Python scalars), return a scalar
        _all_scalar = True
        from ._helpers import _ObjectArray
        for a in args:
            if isinstance(a, _ObjectArray):
                _all_scalar = False
                break
            if isinstance(a, (ndarray,)):
                if a.ndim != 0:
                    _all_scalar = False
                    break
            elif isinstance(a, (list, tuple)):
                _all_scalar = False
                break
        if _all_scalar and isinstance(result, ndarray) and result.ndim == 0:
            val = result.item() if hasattr(result, 'item') else float(result)
            if isinstance(val, tuple) and len(val) == 2:
                val = complex(val[0], val[1])
            result = val

        # Check errstate for raise/callback on invalid/divide operations
        import numpy as _np_mod
        _err = _np_mod.geterr()
        _invalid_mode = _err.get('invalid', 'warn')
        _divide_mode = _err.get('divide', 'warn')
        if _invalid_mode in ('raise', 'call') or _divide_mode in ('raise', 'call'):
            import math as _m
            _has_nan = False
            _has_inf = False
            if isinstance(result, ndarray):
                try:
                    flat = result.flatten()
                    for _idx in range(flat.size):
                        v = float(flat[_idx])
                        if _m.isnan(v):
                            _has_nan = True
                        if _m.isinf(v):
                            _has_inf = True
                        if _has_nan and _has_inf:
                            break
                except (TypeError, ValueError):
                    pass
            elif isinstance(result, (int, float)):
                try:
                    if _m.isnan(float(result)):
                        _has_nan = True
                    elif _m.isinf(float(result)):
                        _has_inf = True
                except (TypeError, ValueError):
                    pass
            if _has_nan and _invalid_mode == 'raise':
                raise FloatingPointError("invalid value encountered in " + self.__name__)
            if _has_nan and _invalid_mode == 'call':
                _errcall = _np_mod.geterrcall()
                if _errcall is not None:
                    _errcall("invalid value encountered in " + self.__name__, 8)
            if _has_inf and _divide_mode == 'raise':
                raise FloatingPointError("divide by zero encountered in " + self.__name__)
            if _has_inf and _divide_mode == 'call':
                _errcall = _np_mod.geterrcall()
                if _errcall is not None:
                    _errcall("divide by zero encountered in " + self.__name__, 1)

        if out is not None:
            out = out[0] if isinstance(out, tuple) else out
            if _casting == "equiv":
                res_dt = getattr(result, 'dtype', None)
                out_dt = getattr(out, 'dtype', None)
                if res_dt is not None and out_dt is not None:
                    if str(res_dt) != str(out_dt):
                        raise TypeError(
                            "Cannot cast ufunc '{}' output from dtype('{}') "
                            "to dtype('{}') with casting rule 'equiv'".format(
                                self.__name__, res_dt, out_dt
                            )
                        )
            _copy_into(out, result)
            return out
        return result

    def __reduce__(self):
        return (_ufunc_reconstruct, ('numpy', self.__name__))

    def __repr__(self):
        return f"<ufunc '{self.__name__}'>"

    def reduce(self, a, axis=0, dtype=None, out=None, keepdims=False,
               initial=_REDUCE_NOVALUE, where=True):
        if self.nin != 2:
            raise ValueError("reduce only supported for binary functions")
        a = asarray(a)
        # Validate dtype
        if dtype is not None:
            if isinstance(dtype, str):
                try:
                    a = a.astype(dtype)
                except Exception:
                    raise TypeError("Cannot cast to dtype {!r}".format(dtype))
            else:
                try:
                    a = a.astype(str(dtype))
                except Exception:
                    raise TypeError("Invalid dtype {!r}".format(dtype))
        # Validate out
        if out is not None:
            if isinstance(out, tuple):
                if len(out) == 1:
                    out = out[0]
                else:
                    raise TypeError("out must be a single array, not a tuple")
            if not hasattr(out, 'shape'):
                raise TypeError("out must be an array, not {!r}".format(type(out).__name__))
        # Validate axis
        if axis is not None and not isinstance(axis, (int, tuple)):
            raise TypeError(
                "axis must be None, int, or tuple of ints, not {!r}".format(
                    type(axis).__name__))
        # axis=() — reduce over zero axes → return copy
        if isinstance(axis, tuple) and len(axis) == 0:
            result = a.copy()
            if out is not None:
                _check_out_shape(out, result)
                _copy_into(out, result)
                return out
            return result
        # Handle where= (anything other than bare True)
        if where is not True and not (isinstance(where, bool) and where):
            result = self._reduce_with_where(a, axis, keepdims, initial, where)
            result = asarray(result)
            if out is not None:
                _check_out_shape(out, result)
                _copy_into(out, result)
                return out
            return result
        # Normal reduction
        _no_init = initial is _REDUCE_NOVALUE
        _use_fast = (self._reduce_fast is not None
                     and _no_init
                     and str(getattr(a, 'dtype', '')) != 'object')
        if _use_fast:
            result = self._reduce_fast(a, axis=axis, keepdims=keepdims)
        else:
            _init = None if _no_init else initial
            result = self._generic_reduce(a, axis=axis, keepdims=keepdims, initial=_init)
        result = asarray(result)
        if out is not None:
            _check_out_shape(out, result)
            _copy_into(out, result)
            return out
        return result

    def _reduce_with_where(self, a, axis, keepdims, initial, where):
        """Reduction applying a boolean where mask."""
        _no_init = initial is _REDUCE_NOVALUE
        identity = None if _no_init else initial
        if identity is None and self.identity is not None:
            identity = self.identity
        if identity is not None:
            where_arr = asarray(where, dtype='bool') if not isinstance(where, bool) else where
            if hasattr(where_arr, 'shape') and where_arr.ndim > 0:
                flat_a = a.ravel().tolist()
                flat_w = where_arr.ravel().tolist()
                if len(flat_w) < len(flat_a):
                    repeats = (len(flat_a) + len(flat_w) - 1) // len(flat_w)
                    flat_w = (flat_w * repeats)[:len(flat_a)]
                flat_m = [v if w else identity for v, w in zip(flat_a, flat_w)]
                masked = array(flat_m, dtype=a.dtype).reshape(a.shape)
            else:
                masked = a
            _init = None if _no_init else initial
            return self._generic_reduce(masked, axis=axis, keepdims=keepdims, initial=_init)
        else:
            _init = None if _no_init else initial
            return self._generic_reduce(a, axis=axis, keepdims=keepdims, initial=_init)

    def accumulate(self, a, axis=0, dtype=None, out=None):
        if self.nin != 2:
            raise ValueError("accumulate only supported for binary functions")
        a = asarray(a)
        if dtype is not None:
            a = a.astype(str(dtype))
        if self._accumulate_fast is not None:
            result = self._accumulate_fast(a, axis=axis)
        else:
            result = self._generic_accumulate(a, axis=axis)
        if out is not None:
            out = out[0] if isinstance(out, tuple) else out
            _check_out_shape(out, asarray(result))
            _copy_into(out, asarray(result))
            return out
        return result

    def outer(self, a, b, **kwargs):
        if self.nin != 2:
            raise ValueError("outer only supported for binary functions")
        a = asarray(a).ravel()
        b = asarray(b).ravel()
        result = self._func(a.reshape((-1, 1)), b.reshape((1, -1)))
        out = kwargs.get('out', None)
        if out is not None:
            out = out[0] if isinstance(out, tuple) else out
            _copy_into(out, asarray(result))
            return out
        return result

    def reduceat(self, a, indices, axis=0, dtype=None, out=None):
        if self.nin != 2:
            raise ValueError("reduceat only supported for binary functions")
        a = asarray(a)
        if dtype is not None:
            a = a.astype(str(dtype))
        indices = [int(x) for x in indices]
        n = a.shape[axis]
        results = []
        for k in range(len(indices)):
            i = indices[k]
            j = indices[k + 1] if k + 1 < len(indices) else n
            if j <= i:
                # When next index <= current, result is a[i]
                sl = [slice(None)] * a.ndim
                sl[axis] = i
                results.append(a[tuple(sl)])
            else:
                sl = [slice(None)] * a.ndim
                sl[axis] = slice(i, j)
                segment = a[tuple(sl)]
                results.append(self.reduce(segment, axis=axis))
        # Ensure all results are arrays for stacking
        results = [asarray(r) for r in results]
        result = stack(results, axis=axis)
        if out is not None:
            out = out[0] if isinstance(out, tuple) else out
            _check_out_shape(out, asarray(result))
            _copy_into(out, asarray(result))
            return out
        return result

    def at(self, a, indices, b=None):
        # Check for __array_ufunc__ dispatch
        au = getattr(type(a), '__array_ufunc__', NotImplemented)
        if au is not NotImplemented and au is not None and not isinstance(a, ndarray):
            result = a.__array_ufunc__(self, 'at', a, indices)
            if result is not NotImplemented:
                return result
        # Reject nout > 1
        if self.nout > 1:
            raise ValueError(
                "ufunc '{}' does not support at() — "
                "nout must be 1".format(self.__name__))
        # Reject gufuncs
        if self.signature is not None:
            raise TypeError(
                "ufunc '{}' with a non-trivial signature cannot be used "
                "with at()".format(self.__name__))
        # Validate b presence
        if self.nin == 1:
            if b is not None:
                raise ValueError(
                    "ufunc '{}' does not take a second operand in "
                    ".at()".format(self.__name__))
        else:
            if b is None:
                raise ValueError(
                    "ufunc '{}' requires a second operand in "
                    ".at()".format(self.__name__))

        n = len(a)
        # Normalize indices to a flat Python list of ints
        if isinstance(indices, slice):
            idx_list = list(range(*indices.indices(n)))
        elif hasattr(indices, 'tolist'):
            idx_list = [int(i) for i in indices.flatten().tolist()]
        else:
            idx_list = [int(i) for i in indices]
        # Resolve negative indices
        idx_list = [i if i >= 0 else n + i for i in idx_list]

        if self.nin == 1:
            for idx in idx_list:
                result = self._func(asarray(a[idx]))
                result = asarray(result)
                _set_at(a, idx, result)
        else:
            b_arr = asarray(b)
            if b_arr.ndim == 0:
                b_list = [b_arr.flat[0]] * len(idx_list)
            else:
                b_flat = b_arr.ravel().tolist()
                if len(b_flat) == 1:
                    b_list = b_flat * len(idx_list)
                elif len(b_flat) != len(idx_list):
                    raise ValueError(
                        "operands could not be broadcast together: "
                        "indices has {} elements but b has {}".format(
                            len(idx_list), len(b_flat)))
                else:
                    b_list = b_flat
            for idx, bv in zip(idx_list, b_list):
                result = self._func(asarray(a[idx]), asarray(bv))
                result = asarray(result)
                _set_at(a, idx, result)

    def _generic_reduce(self, a, axis, keepdims, initial):
        # Handle tuple axis
        if isinstance(axis, tuple):
            result = a
            _init = initial
            for ax in sorted(axis, reverse=True):
                result = asarray(self._generic_reduce(result, ax, keepdims=False,
                                                      initial=_init))
                _init = None  # apply initial only once
            if keepdims:
                result = asarray(result)
                for ax in sorted(axis):
                    result = expand_dims(result, axis=ax)
            return result
        if axis is None:
            flat = a.ravel()
            n = flat.size
            if initial is not None:
                acc = asarray(initial)
                for i in range(n):
                    acc = self._func(acc, flat[i])
            elif self.identity is not None:
                acc = asarray(self.identity)
                for i in range(n):
                    acc = self._func(acc, flat[i])
            else:
                if n == 0:
                    raise ValueError(
                        "zero-size array to reduction operation '{}' "
                        "which has no identity".format(self.__name__))
                acc = asarray(flat[0])
                for i in range(1, n):
                    acc = self._func(acc, flat[i])
            return acc
        # Axis-specific
        n = a.shape[axis]
        if n == 0:
            if self.identity is not None or initial is not None:
                seed = initial if initial is not None else self.identity
                shape = list(a.shape)
                shape.pop(axis)
                if keepdims:
                    shape.insert(axis, 1)
                result_shape = shape if shape else ()
                if result_shape:
                    return array([seed] * (1 if not result_shape else
                                  __import__('math').prod(result_shape)),
                                 dtype=a.dtype).reshape(result_shape)
                else:
                    return asarray(seed)
            raise ValueError(
                "zero-size array to reduction operation '{}' "
                "which has no identity".format(self.__name__))
        slices = [squeeze(take(a, [i], axis=axis), axis=axis) for i in range(n)]
        if initial is not None:
            acc = asarray(initial)
            for s in slices:
                acc = self._func(acc, s)
        else:
            acc = slices[0]
            for s in slices[1:]:
                acc = self._func(acc, s)
        if keepdims:
            acc = expand_dims(asarray(acc), axis=axis)
        return acc

    def _generic_accumulate(self, a, axis):
        if a.ndim == 0:
            return a.copy()
        n = a.shape[axis]
        slices = [squeeze(take(a, [i], axis=axis), axis=axis) for i in range(n)]
        results = [slices[0]]
        for s in slices[1:]:
            results.append(self._func(results[-1], s))
        return stack(results, axis=axis)


# --- Wrap element-wise functions as proper ufunc objects ---------------------
# Save original function references before rebinding names.
# Imported plain functions are referenced here; after this block, the
# module-level names become ufunc instances whose __call__ delegates to the
# saved reference.

_add_func = add
_subtract_func = subtract
_multiply_func = multiply
_divide_func = divide
_true_divide_func = true_divide
_floor_divide_func = floor_divide
_power_func = power
_remainder_func = remainder
_fmod_func = fmod
_maximum_func = maximum
_minimum_func = minimum
_fmax_func = fmax
_fmin_func = fmin
_logical_and_func = logical_and
_logical_or_func = logical_or
_logical_xor_func = logical_xor
_bitwise_and_func = bitwise_and
_bitwise_or_func = bitwise_or
_bitwise_xor_func = bitwise_xor
_left_shift_func = left_shift
_right_shift_func = right_shift
_greater_func = greater
_less_func = less
_equal_func = equal
_not_equal_func = not_equal
_greater_equal_func = greater_equal
_less_equal_func = less_equal
_arctan2_func = arctan2
_hypot_func = hypot
_copysign_func = copysign
_ldexp_func = ldexp
_heaviside_func = heaviside
_nextafter_func = nextafter

# Binary ufuncs with fast-path reduce/accumulate
add = ufunc._create(_add_func, 2, name='add', identity=0,
            reduce_fast=lambda a, axis=0, keepdims=False: sum(a, axis=axis, keepdims=keepdims),
            accumulate_fast=lambda a, axis=0: cumsum(a, axis=axis),
            types=_NUMERIC_BINARY_TYPES + ['OO->O'])
multiply = ufunc._create(_multiply_func, 2, name='multiply', identity=1,
                 reduce_fast=lambda a, axis=0, keepdims=False: prod(a, axis=axis, keepdims=keepdims),
                 accumulate_fast=lambda a, axis=0: cumprod(a, axis=axis),
                 types=_NUMERIC_BINARY_TYPES + ['OO->O'])
maximum = ufunc._create(_maximum_func, 2, name='maximum',
                reduce_fast=lambda a, axis=0, keepdims=False: max(a, axis=axis, keepdims=keepdims),
                types=_NUMERIC_BINARY_TYPES)
minimum = ufunc._create(_minimum_func, 2, name='minimum',
                reduce_fast=lambda a, axis=0, keepdims=False: min(a, axis=axis, keepdims=keepdims),
                types=_NUMERIC_BINARY_TYPES)
logical_and = ufunc._create(_logical_and_func, 2, name='logical_and', identity=True,
                    reduce_fast=lambda a, axis=0, keepdims=False: all(a, axis=axis, keepdims=keepdims),
                    types=['??->?', 'OO->O'])
logical_or = ufunc._create(_logical_or_func, 2, name='logical_or', identity=False,
                   reduce_fast=lambda a, axis=0, keepdims=False: any(a, axis=axis, keepdims=keepdims),
                   types=['??->?', 'OO->O'])

# Binary ufuncs with generic reduce only
subtract = ufunc._create(_subtract_func, 2, name='subtract',
                         types=_NUMERIC_BINARY_TYPES + ['OO->O'])
divide = ufunc._create(_divide_func, 2, name='divide',
                       types=_FLOAT_BINARY_TYPES)
true_divide = ufunc._create(_true_divide_func, 2, name='true_divide',
                            types=_FLOAT_BINARY_TYPES)
floor_divide = ufunc._create(_floor_divide_func, 2, name='floor_divide',
                             types=_NUMERIC_BINARY_TYPES)
power = ufunc._create(_power_func, 2, name='power',
                      types=_NUMERIC_BINARY_TYPES)
remainder = ufunc._create(_remainder_func, 2, name='remainder',
                          types=_NUMERIC_BINARY_TYPES)
mod = remainder
fmod = ufunc._create(_fmod_func, 2, name='fmod',
                     types=_FLOAT_BINARY_TYPES)
fmax = ufunc._create(_fmax_func, 2, name='fmax',
                     types=_FLOAT_BINARY_TYPES)
fmin = ufunc._create(_fmin_func, 2, name='fmin',
                     types=_FLOAT_BINARY_TYPES)
logical_xor = ufunc._create(_logical_xor_func, 2, name='logical_xor', identity=False,
                            types=['??->?', 'OO->O'])
bitwise_and = ufunc._create(_bitwise_and_func, 2, name='bitwise_and',
                            types=_INT_BINARY_TYPES)
bitwise_or = ufunc._create(_bitwise_or_func, 2, name='bitwise_or',
                           types=_INT_BINARY_TYPES)
bitwise_xor = ufunc._create(_bitwise_xor_func, 2, name='bitwise_xor',
                            types=_INT_BINARY_TYPES)
left_shift = ufunc._create(_left_shift_func, 2, name='left_shift',
                           types=_INT_BINARY_TYPES)
right_shift = ufunc._create(_right_shift_func, 2, name='right_shift',
                            types=_INT_BINARY_TYPES)
greater = ufunc._create(_greater_func, 2, name='greater',
                        types=_CMP_BINARY_TYPES)
less = ufunc._create(_less_func, 2, name='less',
                     types=_CMP_BINARY_TYPES)
equal = ufunc._create(_equal_func, 2, name='equal',
                      types=_CMP_BINARY_TYPES)
not_equal = ufunc._create(_not_equal_func, 2, name='not_equal',
                          types=_CMP_BINARY_TYPES)
greater_equal = ufunc._create(_greater_equal_func, 2, name='greater_equal',
                              types=_CMP_BINARY_TYPES)
less_equal = ufunc._create(_less_equal_func, 2, name='less_equal',
                           types=_CMP_BINARY_TYPES)
arctan2 = ufunc._create(_arctan2_func, 2, name='arctan2',
                        types=_FLOAT_BINARY_TYPES)
hypot = ufunc._create(_hypot_func, 2, name='hypot',
                      types=_FLOAT_BINARY_TYPES)
copysign = ufunc._create(_copysign_func, 2, name='copysign',
                         types=_FLOAT_BINARY_TYPES)
ldexp = ufunc._create(_ldexp_func, 2, name='ldexp',
                      types=['fi->f', 'di->d'])
heaviside = ufunc._create(_heaviside_func, 2, name='heaviside',
                          types=_FLOAT_BINARY_TYPES)
nextafter = ufunc._create(_nextafter_func, 2, name='nextafter',
                          types=_FLOAT_BINARY_TYPES)

# nout=2 ufuncs (can call, but .at() raises ValueError)
_modf_func = modf
modf = ufunc._create(_modf_func, 1, nout=2, name='modf',
                     types=['f->ff', 'd->dd'])

# Unary ufuncs (nin=1) — callable, but reduce/accumulate/outer raise ValueError
_sin_func = sin
_cos_func = cos
_tan_func = tan
_arcsin_func = arcsin
_arccos_func = arccos
_arctan_func = arctan
_sinh_func = sinh
_cosh_func = cosh
_tanh_func = tanh
_exp_func = exp
_exp2_func = exp2
_log_func = log
_log2_func = log2
_log10_func = log10
_sqrt_func = sqrt
_cbrt_func = cbrt
_square_func = square
_reciprocal_func = reciprocal
_negative_func = negative
_positive_func = positive
_absolute_func = absolute
_sign_func = sign
_floor_func = floor
_ceil_func = ceil
_rint_func = rint
_trunc_func = trunc
_deg2rad_func = deg2rad
_rad2deg_func = rad2deg
_signbit_func = signbit
_logical_not_func = logical_not
_isnan_func = isnan
_isinf_func = isinf
_isfinite_func = isfinite

sin = ufunc._create(_sin_func, 1, name='sin', types=_FLOAT_UNARY_TYPES)
cos = ufunc._create(_cos_func, 1, name='cos', types=_FLOAT_UNARY_TYPES)
tan = ufunc._create(_tan_func, 1, name='tan', types=_FLOAT_UNARY_TYPES)
arcsin = ufunc._create(_arcsin_func, 1, name='arcsin', types=_FLOAT_UNARY_TYPES)
arccos = ufunc._create(_arccos_func, 1, name='arccos', types=_FLOAT_UNARY_TYPES)
arctan = ufunc._create(_arctan_func, 1, name='arctan', types=_FLOAT_UNARY_TYPES)
sinh = ufunc._create(_sinh_func, 1, name='sinh', types=_FLOAT_UNARY_TYPES)
cosh = ufunc._create(_cosh_func, 1, name='cosh', types=_FLOAT_UNARY_TYPES)
tanh = ufunc._create(_tanh_func, 1, name='tanh', types=_FLOAT_UNARY_TYPES)
exp = ufunc._create(_exp_func, 1, name='exp', types=_FLOAT_UNARY_TYPES)
exp2 = ufunc._create(_exp2_func, 1, name='exp2', types=_FLOAT_UNARY_TYPES)
log = ufunc._create(_log_func, 1, name='log', types=_FLOAT_UNARY_TYPES)
log2 = ufunc._create(_log2_func, 1, name='log2', types=_FLOAT_UNARY_TYPES)
log10 = ufunc._create(_log10_func, 1, name='log10', types=_FLOAT_UNARY_TYPES)
sqrt = ufunc._create(_sqrt_func, 1, name='sqrt', types=_FLOAT_UNARY_TYPES)
cbrt = ufunc._create(_cbrt_func, 1, name='cbrt', types=_FLOAT_UNARY_TYPES)
square = ufunc._create(_square_func, 1, name='square', types=_NUMERIC_UNARY_TYPES)
reciprocal = ufunc._create(_reciprocal_func, 1, name='reciprocal', types=_FLOAT_UNARY_TYPES)
negative = ufunc._create(_negative_func, 1, name='negative',
                         types=_NUMERIC_UNARY_TYPES + ['O->O'])
positive = ufunc._create(_positive_func, 1, name='positive',
                         types=_NUMERIC_UNARY_TYPES + ['O->O'])
absolute = ufunc._create(_absolute_func, 1, name='absolute',
                         types=_NUMERIC_UNARY_TYPES + ['O->O'])
abs = absolute
sign = ufunc._create(_sign_func, 1, name='sign', types=_NUMERIC_UNARY_TYPES)
floor = ufunc._create(_floor_func, 1, name='floor', types=_FLOAT_UNARY_TYPES)
ceil = ufunc._create(_ceil_func, 1, name='ceil', types=_FLOAT_UNARY_TYPES)
rint = ufunc._create(_rint_func, 1, name='rint', types=_FLOAT_UNARY_TYPES)
trunc = ufunc._create(_trunc_func, 1, name='trunc', types=_FLOAT_UNARY_TYPES)
deg2rad = ufunc._create(_deg2rad_func, 1, name='deg2rad', types=_FLOAT_UNARY_TYPES)
rad2deg = ufunc._create(_rad2deg_func, 1, name='rad2deg', types=_FLOAT_UNARY_TYPES)
signbit = ufunc._create(_signbit_func, 1, name='signbit', types=['f->?', 'd->?'])
logical_not = ufunc._create(_logical_not_func, 1, name='logical_not', types=['?->?', 'O->?'])
isnan = ufunc._create(_isnan_func, 1, name='isnan', types=['f->?', 'd->?'])
isinf = ufunc._create(_isinf_func, 1, name='isinf', types=['f->?', 'd->?'])
isfinite = ufunc._create(_isfinite_func, 1, name='isfinite', types=['f->?', 'd->?'])

_bitwise_not_func = bitwise_not
bitwise_not = ufunc._create(_bitwise_not_func, 1, name='bitwise_not',
                            types=_INT_UNARY_TYPES)
invert = bitwise_not

# bitwise_count (popcount)
from ._bitwise import bitwise_count as _bitwise_count_func
bitwise_count = ufunc._create(_bitwise_count_func, 1, name='bitwise_count')
bitwise_count.types = ['b->B', 'B->B', 'h->B', 'H->B',
                       'i->B', 'I->B', 'l->B', 'L->B',
                       'q->B', 'Q->B', 'O->O']
bitwise_count.ntypes = len(bitwise_count.types)

# gcd / lcm / divmod
try:
    from ._math import gcd as _gcd_func
    gcd = ufunc._create(_gcd_func, 2, name='gcd', types=_INT_BINARY_TYPES)
except (ImportError, AttributeError):
    pass

try:
    from ._math import lcm as _lcm_func
    lcm = ufunc._create(_lcm_func, 2, name='lcm', types=_INT_BINARY_TYPES)
except (ImportError, AttributeError):
    pass

try:
    from ._math import divmod_ as _divmod_func
    divmod = ufunc._create(_divmod_func, 2, nout=2, name='divmod',
                           types=['ll->ll', 'qq->qq', 'ff->ff', 'dd->dd'])
except (ImportError, AttributeError):
    pass
