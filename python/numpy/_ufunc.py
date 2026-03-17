"""NumPy ufunc class and function wrapping registration."""
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
    # Wrapped binary ufuncs (overwrite plain function names with ufunc objects)
    'add', 'subtract', 'multiply', 'divide', 'true_divide', 'floor_divide',
    'power', 'remainder', 'mod', 'fmod', 'maximum', 'minimum', 'fmax', 'fmin',
    'logical_and', 'logical_or', 'logical_xor',
    'bitwise_and', 'bitwise_or', 'bitwise_xor', 'left_shift', 'right_shift',
    'greater', 'less', 'equal', 'not_equal', 'greater_equal', 'less_equal',
    'arctan2', 'hypot', 'copysign', 'ldexp', 'heaviside', 'nextafter',
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
]

_UFUNC_SENTINEL = object()  # private token — only _make_ufunc passes it

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
                reduce_fast=None, accumulate_fast=None):
        """Internal factory — use _make_ufunc() instead of ufunc(...)."""
        obj = cls.__new__(cls)
        obj._func = func
        obj.nin = nin
        obj.nout = nout
        obj.nargs = nin + nout
        obj.identity = identity
        obj.__name__ = name or getattr(func, '__name__', 'ufunc')
        obj._reduce_fast = reduce_fast
        obj._accumulate_fast = accumulate_fast
        obj.ntypes = 0
        obj.types = []
        obj.signature = None
        return obj

    def __call__(self, *args, **kwargs):
        out = kwargs.pop('out', None)
        _dtype = kwargs.pop('dtype', None)
        # Pop silently-ignored params so they don't reach the wrapped func
        kwargs.pop('casting', None)
        kwargs.pop('where', None)
        kwargs.pop('subok', None)
        kwargs.pop('order', None)
        result = self._func(*args, **kwargs)
        if _dtype is not None:
            result = asarray(result).astype(str(_dtype))
        if out is not None:
            out = out[0] if isinstance(out, tuple) else out
            _copy_into(out, asarray(result))
            return out
        return result

    def __repr__(self):
        return f"<ufunc '{self.__name__}'>"

    def reduce(self, a, axis=0, dtype=None, out=None, keepdims=False,
               initial=None, where=True):
        if self.nin != 2:
            raise ValueError("reduce only supported for binary functions")
        a = asarray(a)
        if dtype is not None:
            a = a.astype(str(dtype))
        if self._reduce_fast is not None and initial is None:
            result = self._reduce_fast(a, axis=axis, keepdims=keepdims)
        else:
            result = self._generic_reduce(a, axis=axis, keepdims=keepdims,
                                          initial=initial)
        if out is not None:
            out = out[0] if isinstance(out, tuple) else out
            _copy_into(out, asarray(result))
            return out
        return result

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
        indices = list(indices)
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
            _copy_into(out, asarray(result))
            return out
        return result

    def at(self, a, indices, b=None):
        if self.nin != 2:
            raise ValueError("at only supported for binary functions")
        if b is None:
            raise ValueError("second operand required for binary ufunc")
        indices = list(indices) if not isinstance(indices, list) else indices
        b = asarray(b)
        b_flat = b.flatten().tolist() if b.ndim > 0 else [float(b)]
        for k, idx in enumerate(indices):
            bv = b_flat[k] if k < len(b_flat) else b_flat[-1]
            result = self._func(a[idx], bv)
            # Extract scalar from result (func may return array)
            result = asarray(result)
            a[idx] = float(result) if result.ndim == 0 or result.size == 1 else result

    def _generic_reduce(self, a, axis, keepdims, initial):
        if axis is None:
            flat = a.ravel()
            if initial is not None:
                result = asarray(initial)
                start = 0
            elif self.identity is not None:
                result = asarray(self.identity)
                start = 0
            else:
                if len(flat) == 0:
                    raise ValueError("zero-size array with no identity")
                result = flat[0]
                start = 1
            for i in range(start, len(flat)):
                result = self._func(result, flat[i])
            return result
        # Axis-specific: fold slices along axis
        n = a.shape[axis]
        if n == 0:
            raise ValueError("zero-size array with no identity")
        slices = [squeeze(take(a, [i], axis=axis), axis=axis) for i in range(n)]
        if initial is not None:
            result = asarray(initial)
            for s in slices:
                result = self._func(result, s)
        else:
            result = slices[0]
            for s in slices[1:]:
                result = self._func(result, s)
        if keepdims:
            result = expand_dims(asarray(result), axis=axis)
        return result

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
            accumulate_fast=lambda a, axis=0: cumsum(a, axis=axis))
multiply = ufunc._create(_multiply_func, 2, name='multiply', identity=1,
                 reduce_fast=lambda a, axis=0, keepdims=False: prod(a, axis=axis, keepdims=keepdims),
                 accumulate_fast=lambda a, axis=0: cumprod(a, axis=axis))
maximum = ufunc._create(_maximum_func, 2, name='maximum',
                reduce_fast=lambda a, axis=0, keepdims=False: max(a, axis=axis, keepdims=keepdims))
minimum = ufunc._create(_minimum_func, 2, name='minimum',
                reduce_fast=lambda a, axis=0, keepdims=False: min(a, axis=axis, keepdims=keepdims))
logical_and = ufunc._create(_logical_and_func, 2, name='logical_and', identity=True,
                    reduce_fast=lambda a, axis=0, keepdims=False: all(a, axis=axis, keepdims=keepdims))
logical_or = ufunc._create(_logical_or_func, 2, name='logical_or', identity=False,
                   reduce_fast=lambda a, axis=0, keepdims=False: any(a, axis=axis, keepdims=keepdims))

# Binary ufuncs with generic reduce only
subtract = ufunc._create(_subtract_func, 2, name='subtract')
divide = ufunc._create(_divide_func, 2, name='divide')
true_divide = ufunc._create(_true_divide_func, 2, name='true_divide')
floor_divide = ufunc._create(_floor_divide_func, 2, name='floor_divide')
power = ufunc._create(_power_func, 2, name='power')
remainder = ufunc._create(_remainder_func, 2, name='remainder')
mod = remainder
fmod = ufunc._create(_fmod_func, 2, name='fmod')
fmax = ufunc._create(_fmax_func, 2, name='fmax')
fmin = ufunc._create(_fmin_func, 2, name='fmin')
logical_xor = ufunc._create(_logical_xor_func, 2, name='logical_xor', identity=False)
bitwise_and = ufunc._create(_bitwise_and_func, 2, name='bitwise_and')
bitwise_or = ufunc._create(_bitwise_or_func, 2, name='bitwise_or')
bitwise_xor = ufunc._create(_bitwise_xor_func, 2, name='bitwise_xor')
left_shift = ufunc._create(_left_shift_func, 2, name='left_shift')
right_shift = ufunc._create(_right_shift_func, 2, name='right_shift')
greater = ufunc._create(_greater_func, 2, name='greater')
less = ufunc._create(_less_func, 2, name='less')
equal = ufunc._create(_equal_func, 2, name='equal')
not_equal = ufunc._create(_not_equal_func, 2, name='not_equal')
greater_equal = ufunc._create(_greater_equal_func, 2, name='greater_equal')
less_equal = ufunc._create(_less_equal_func, 2, name='less_equal')
arctan2 = ufunc._create(_arctan2_func, 2, name='arctan2')
hypot = ufunc._create(_hypot_func, 2, name='hypot')
copysign = ufunc._create(_copysign_func, 2, name='copysign')
ldexp = ufunc._create(_ldexp_func, 2, name='ldexp')
heaviside = ufunc._create(_heaviside_func, 2, name='heaviside')
nextafter = ufunc._create(_nextafter_func, 2, name='nextafter')

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

sin = ufunc._create(_sin_func, 1, name='sin')
cos = ufunc._create(_cos_func, 1, name='cos')
tan = ufunc._create(_tan_func, 1, name='tan')
arcsin = ufunc._create(_arcsin_func, 1, name='arcsin')
arccos = ufunc._create(_arccos_func, 1, name='arccos')
arctan = ufunc._create(_arctan_func, 1, name='arctan')
sinh = ufunc._create(_sinh_func, 1, name='sinh')
cosh = ufunc._create(_cosh_func, 1, name='cosh')
tanh = ufunc._create(_tanh_func, 1, name='tanh')
exp = ufunc._create(_exp_func, 1, name='exp')
exp2 = ufunc._create(_exp2_func, 1, name='exp2')
log = ufunc._create(_log_func, 1, name='log')
log2 = ufunc._create(_log2_func, 1, name='log2')
log10 = ufunc._create(_log10_func, 1, name='log10')
sqrt = ufunc._create(_sqrt_func, 1, name='sqrt')
cbrt = ufunc._create(_cbrt_func, 1, name='cbrt')
square = ufunc._create(_square_func, 1, name='square')
reciprocal = ufunc._create(_reciprocal_func, 1, name='reciprocal')
negative = ufunc._create(_negative_func, 1, name='negative')
positive = ufunc._create(_positive_func, 1, name='positive')
absolute = ufunc._create(_absolute_func, 1, name='absolute')
abs = absolute
sign = ufunc._create(_sign_func, 1, name='sign')
floor = ufunc._create(_floor_func, 1, name='floor')
ceil = ufunc._create(_ceil_func, 1, name='ceil')
rint = ufunc._create(_rint_func, 1, name='rint')
trunc = ufunc._create(_trunc_func, 1, name='trunc')
deg2rad = ufunc._create(_deg2rad_func, 1, name='deg2rad')
rad2deg = ufunc._create(_rad2deg_func, 1, name='rad2deg')
signbit = ufunc._create(_signbit_func, 1, name='signbit')
logical_not = ufunc._create(_logical_not_func, 1, name='logical_not')
isnan = ufunc._create(_isnan_func, 1, name='isnan')
isinf = ufunc._create(_isinf_func, 1, name='isinf')
isfinite = ufunc._create(_isfinite_func, 1, name='isfinite')

_bitwise_not_func = bitwise_not
bitwise_not = ufunc._create(_bitwise_not_func, 1, name='bitwise_not')
invert = bitwise_not

# bitwise_count (popcount)
from ._bitwise import bitwise_count as _bitwise_count_func
bitwise_count = ufunc._create(_bitwise_count_func, 1, name='bitwise_count')
bitwise_count.types = ['b->B', 'B->B', 'h->B', 'H->B',
                       'i->B', 'I->B', 'l->B', 'L->B',
                       'q->B', 'Q->B', 'O->O']
bitwise_count.ntypes = len(bitwise_count.types)
