"""numpy.ma - masked array support.

Provides a nearly complete surface-compatible stub of NumPy's masked-array
module.  Every public name that real ``numpy.ma`` exposes (227 symbols as of
NumPy 2.4) is present here so that user code doing ``from numpy.ma import X``
never gets an ``ImportError``.
"""

import sys as _sys


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------

class MAError(Exception):
    """Class for masked array related errors."""

class MaskError(MAError):
    """Class for mask related errors."""


# ---------------------------------------------------------------------------
# Constants / sentinel objects
# ---------------------------------------------------------------------------

nomask = False

MaskType = bool


class _MaskedPrintOption:
    """Controls how masked values are printed."""
    def __init__(self, display):
        self._display = display
        self._enabled = True

    def display(self):
        return self._display

    def set_display(self, s):
        self._display = s

    def enabled(self):
        return self._enabled

    def enable(self, shrink=True):
        self._enabled = shrink

    def __str__(self):
        return str(self._display)

    def __repr__(self):
        return str(self._display)


masked_print_option = _MaskedPrintOption('--')


# ---------------------------------------------------------------------------
# MaskedConstant
# ---------------------------------------------------------------------------

class MaskedConstant:
    """The masked constant (singleton)."""
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __repr__(self):
        return '--'

    def __str__(self):
        return '--'

    def __bool__(self):
        return False

    def __eq__(self, other):
        return isinstance(other, MaskedConstant)

    def __ne__(self, other):
        return not isinstance(other, MaskedConstant)

    def __hash__(self):
        return hash('--')


masked = MaskedConstant()
masked_singleton = masked


# ---------------------------------------------------------------------------
# MaskedArray
# ---------------------------------------------------------------------------

class MaskedArray:
    """Simplified masked array."""

    def __init__(self, data, mask=None, dtype=None, fill_value=None,
                 keep_mask=True, hard_mask=False, shrink=True,
                 copy=False, subok=True, ndmin=0, order=None):
        import numpy as np
        if dtype is not None:
            self.data = np.asarray(data, dtype=dtype)
        else:
            self.data = np.asarray(data)
        if mask is None or mask is nomask:
            self.mask = np.zeros(self.data.shape, dtype="bool")
        elif isinstance(mask, bool) and not mask:
            self.mask = np.zeros(self.data.shape, dtype="bool")
        elif isinstance(mask, bool) and mask:
            self.mask = np.ones(self.data.shape, dtype="bool")
        else:
            self.mask = np.asarray(mask, dtype="bool")
            # broadcast mask to data shape if needed
            if self.mask.shape != self.data.shape:
                self.mask = np.broadcast_to(self.mask, self.data.shape).copy()
        self._fill_value = fill_value if fill_value is not None else _default_fill_value_for(self.data)
        self._hardmask = hard_mask

    # -- properties --

    @property
    def shape(self):
        return self.data.shape

    @shape.setter
    def shape(self, new_shape):
        self.data = self.data.reshape(new_shape)
        self.mask = self.mask.reshape(new_shape)

    @property
    def ndim(self):
        return self.data.ndim

    @property
    def size(self):
        return self.data.size

    @property
    def dtype(self):
        return self.data.dtype

    @property
    def T(self):
        import numpy as np
        return MaskedArray(np.transpose(self.data), mask=np.transpose(self.mask),
                           fill_value=self._fill_value)

    @property
    def flat(self):
        return self.flatten()

    @property
    def fill_value(self):
        return self._fill_value

    @fill_value.setter
    def fill_value(self, value):
        self._fill_value = value

    @property
    def baseclass(self):
        import numpy as np
        return np.ndarray

    @property
    def hardmask(self):
        return self._hardmask

    @property
    def _data(self):
        return self.data

    @property
    def sharedmask(self):
        return False

    @property
    def real(self):
        return self.get_real()

    @property
    def imag(self):
        return self.get_imag()

    def get_real(self):
        import numpy as np
        return MaskedArray(np.real(self.data), mask=self.mask, fill_value=self._fill_value)

    def get_imag(self):
        import numpy as np
        return MaskedArray(np.imag(self.data), mask=self.mask, fill_value=self._fill_value)

    def get_fill_value(self):
        return self._fill_value

    def set_fill_value(self, value):
        self._fill_value = value

    # -- core methods --

    def filled(self, fill_value=None):
        import numpy as np
        fv = fill_value if fill_value is not None else self._fill_value
        result = self.data.copy()
        if not _any_true(self.mask):
            return result
        mask_flat = self.mask.flatten().tolist()
        result_flat = result.flatten().tolist()
        for i in range(len(result_flat)):
            if mask_flat[i]:
                result_flat[i] = fv
        return np.array(result_flat, dtype=self.data.dtype).reshape(self.data.shape)

    def compressed(self):
        import numpy as np
        data_list = self.data.flatten().tolist()
        mask_list = self.mask.flatten().tolist()
        return np.array([d for d, m in zip(data_list, mask_list) if not m],
                        dtype=self.data.dtype)

    def count(self, axis=None, keepdims=False):
        import numpy as np
        not_masked = np.logical_not(self.mask).astype("int64")
        return int(np.sum(not_masked, axis=axis, keepdims=keepdims))

    def nonzero(self):
        import numpy as np
        filled = self.filled(0)
        return np.nonzero(filled)

    def fill(self, value):
        import numpy as np
        self.data = np.full(self.data.shape, value, dtype=self.data.dtype)

    # -- aggregation --

    def sum(self, axis=None, dtype=None, out=None, keepdims=False):
        import numpy as np
        filled = self.filled(0)
        return np.sum(filled, axis=axis, keepdims=keepdims)

    def prod(self, axis=None, dtype=None, out=None, keepdims=False):
        import numpy as np
        filled = self.filled(1)
        return np.prod(filled, axis=axis, keepdims=keepdims)

    product = prod

    def mean(self, axis=None, dtype=None, out=None, keepdims=False):
        import numpy as np
        filled = self.filled(0.0)
        not_mask = np.logical_not(self.mask).astype("float64")
        s = np.sum(filled * not_mask, axis=axis, keepdims=keepdims)
        c = np.sum(not_mask, axis=axis, keepdims=keepdims)
        return s / c

    def var(self, axis=None, dtype=None, out=None, ddof=0, keepdims=False, mean=None):
        import numpy as np
        m = self.mean(axis=axis, keepdims=True) if mean is None else mean
        filled = self.filled(0.0)
        not_mask = np.logical_not(self.mask).astype("float64")
        diff = (filled - m) * not_mask
        v = np.sum(diff * diff, axis=axis, keepdims=keepdims)
        c = np.sum(not_mask, axis=axis, keepdims=keepdims)
        return v / (c - ddof)

    def std(self, axis=None, dtype=None, out=None, ddof=0, keepdims=False, mean=None):
        import numpy as np
        return np.sqrt(self.var(axis=axis, ddof=ddof, keepdims=keepdims, mean=mean))

    def min(self, axis=None, out=None, fill_value=None, keepdims=False):
        import numpy as np
        # Fill masked with large value so they don't affect min
        fv = fill_value if fill_value is not None else maximum_fill_value(self.data)
        return np.min(self.filled(fv), axis=axis, keepdims=keepdims)

    def max(self, axis=None, out=None, fill_value=None, keepdims=False):
        import numpy as np
        # Fill masked with small value so they don't affect max
        fv = fill_value if fill_value is not None else minimum_fill_value(self.data)
        return np.max(self.filled(fv), axis=axis, keepdims=keepdims)

    def ptp(self, axis=None, out=None, fill_value=None, keepdims=False):
        return self.max(axis=axis, fill_value=fill_value) - self.min(axis=axis, fill_value=fill_value)

    def all(self, axis=None, out=None, keepdims=False):
        import numpy as np
        c = self.compressed()
        return np.all(c)

    def any(self, axis=None, out=None, keepdims=False):
        import numpy as np
        c = self.compressed()
        return np.any(c)

    def trace(self, offset=0, axis1=0, axis2=1, dtype=None, out=None):
        import numpy as np
        return np.trace(self.filled(0), offset=offset, axis1=axis1, axis2=axis2)

    def cumsum(self, axis=None, dtype=None, out=None):
        import numpy as np
        return MaskedArray(np.cumsum(self.filled(0), axis=axis), mask=self.mask)

    def cumprod(self, axis=None, dtype=None, out=None):
        import numpy as np
        return MaskedArray(np.cumprod(self.filled(1), axis=axis), mask=self.mask)

    def anom(self, axis=None, dtype=None):
        m = self.mean(axis=axis)
        return self - m

    # -- shape manipulation --

    def flatten(self, order='C'):
        return MaskedArray(self.data.flatten(), mask=self.mask.flatten(),
                           fill_value=self._fill_value)

    def ravel(self, order='C'):
        return self.flatten(order=order)

    def reshape(self, *args, **kwargs):
        if len(args) == 1 and isinstance(args[0], (tuple, list)):
            new_shape = args[0]
        else:
            new_shape = args
        return MaskedArray(self.data.reshape(new_shape),
                           mask=self.mask.reshape(new_shape),
                           fill_value=self._fill_value)

    def resize(self, new_shape):
        import numpy as np
        return MaskedArray(np.resize(self.data, new_shape),
                           mask=np.resize(self.mask, new_shape),
                           fill_value=self._fill_value)

    def transpose(self, *axes):
        import numpy as np
        if len(axes) == 0:
            axes = None
        elif len(axes) == 1 and isinstance(axes[0], (tuple, list)):
            axes = axes[0]
        return MaskedArray(np.transpose(self.data, axes),
                           mask=np.transpose(self.mask, axes),
                           fill_value=self._fill_value)

    def swapaxes(self, axis1, axis2):
        import numpy as np
        return MaskedArray(np.swapaxes(self.data, axis1, axis2),
                           mask=np.swapaxes(self.mask, axis1, axis2),
                           fill_value=self._fill_value)

    def squeeze(self, axis=None):
        import numpy as np
        return MaskedArray(np.squeeze(self.data, axis=axis),
                           mask=np.squeeze(self.mask, axis=axis),
                           fill_value=self._fill_value)

    def repeat(self, repeats, axis=None):
        import numpy as np
        return MaskedArray(np.repeat(self.data, repeats, axis=axis),
                           mask=np.repeat(self.mask, repeats, axis=axis),
                           fill_value=self._fill_value)

    def diagonal(self, offset=0, axis1=0, axis2=1):
        import numpy as np
        return MaskedArray(np.diagonal(self.data, offset, axis1, axis2),
                           mask=np.diagonal(self.mask, offset, axis1, axis2),
                           fill_value=self._fill_value)

    # -- indexing / element access --

    def __getitem__(self, key):
        import numpy as np
        d = self.data[key]
        m = self.mask[key]
        if isinstance(d, np.ndarray):
            return MaskedArray(d, mask=m, fill_value=self._fill_value)
        # scalar
        if m:
            return masked
        return d

    def __setitem__(self, key, value):
        if value is masked:
            self.mask[key] = True
        elif isinstance(value, MaskedArray):
            self.data[key] = value.data
            self.mask[key] = value.mask
        else:
            self.data[key] = value
            self.mask[key] = False

    def __len__(self):
        return len(self.data)

    def __iter__(self):
        for i in range(len(self)):
            yield self[i]

    def item(self, *args):
        return self.data.item(*args)

    # -- comparison --

    def __eq__(self, other):
        import numpy as np
        if isinstance(other, MaskedArray):
            return MaskedArray(self.data == other.data,
                               mask=np.logical_or(self.mask, other.mask))
        return MaskedArray(self.data == other, mask=self.mask)

    def __ne__(self, other):
        import numpy as np
        if isinstance(other, MaskedArray):
            return MaskedArray(self.data != other.data,
                               mask=np.logical_or(self.mask, other.mask))
        return MaskedArray(self.data != other, mask=self.mask)

    def __lt__(self, other):
        import numpy as np
        if isinstance(other, MaskedArray):
            return MaskedArray(self.data < other.data,
                               mask=np.logical_or(self.mask, other.mask))
        return MaskedArray(self.data < other, mask=self.mask)

    def __le__(self, other):
        import numpy as np
        if isinstance(other, MaskedArray):
            return MaskedArray(self.data <= other.data,
                               mask=np.logical_or(self.mask, other.mask))
        return MaskedArray(self.data <= other, mask=self.mask)

    def __gt__(self, other):
        import numpy as np
        if isinstance(other, MaskedArray):
            return MaskedArray(self.data > other.data,
                               mask=np.logical_or(self.mask, other.mask))
        return MaskedArray(self.data > other, mask=self.mask)

    def __ge__(self, other):
        import numpy as np
        if isinstance(other, MaskedArray):
            return MaskedArray(self.data >= other.data,
                               mask=np.logical_or(self.mask, other.mask))
        return MaskedArray(self.data >= other, mask=self.mask)

    # -- arithmetic --

    def _binop(self, other, op):
        import numpy as np
        if isinstance(other, MaskedArray):
            d = op(self.data, other.data)
            m = np.logical_or(self.mask, other.mask)
        else:
            d = op(self.data, other)
            m = self.mask
        return MaskedArray(d, mask=m, fill_value=self._fill_value)

    def __add__(self, other): return self._binop(other, lambda a, b: a + b)
    def __radd__(self, other): return self._binop(other, lambda a, b: b + a)
    def __sub__(self, other): return self._binop(other, lambda a, b: a - b)
    def __rsub__(self, other): return self._binop(other, lambda a, b: b - a)
    def __mul__(self, other): return self._binop(other, lambda a, b: a * b)
    def __rmul__(self, other): return self._binop(other, lambda a, b: b * a)
    def __truediv__(self, other): return self._binop(other, lambda a, b: a / b)
    def __rtruediv__(self, other): return self._binop(other, lambda a, b: b / a)
    def __floordiv__(self, other): return self._binop(other, lambda a, b: a // b)
    def __rfloordiv__(self, other): return self._binop(other, lambda a, b: b // a)
    def __mod__(self, other): return self._binop(other, lambda a, b: a % b)
    def __rmod__(self, other): return self._binop(other, lambda a, b: b % a)
    def __pow__(self, other): return self._binop(other, lambda a, b: a ** b)
    def __rpow__(self, other): return self._binop(other, lambda a, b: b ** a)
    def __and__(self, other): return self._binop(other, lambda a, b: a & b)
    def __or__(self, other): return self._binop(other, lambda a, b: a | b)
    def __xor__(self, other): return self._binop(other, lambda a, b: a ^ b)
    def __lshift__(self, other): return self._binop(other, lambda a, b: a << b)
    def __rshift__(self, other): return self._binop(other, lambda a, b: a >> b)

    def __neg__(self):
        return MaskedArray(-self.data, mask=self.mask, fill_value=self._fill_value)

    def __pos__(self):
        return MaskedArray(+self.data, mask=self.mask, fill_value=self._fill_value)

    def __abs__(self):
        import numpy as np
        return MaskedArray(np.abs(self.data), mask=self.mask, fill_value=self._fill_value)

    def __invert__(self):
        import numpy as np
        return MaskedArray(~self.data, mask=self.mask, fill_value=self._fill_value)

    # -- conversion --

    def copy(self, order='C'):
        return MaskedArray(self.data.copy(), mask=self.mask.copy(),
                           fill_value=self._fill_value)

    def astype(self, dtype, order='K', casting='unsafe', subok=True, copy=True):
        return MaskedArray(self.data.astype(dtype), mask=self.mask,
                           fill_value=self._fill_value)

    def tolist(self):
        if self.ndim == 0:
            if _any_true(self.mask):
                return None
            return self.data.tolist()
        result = []
        for i in range(len(self)):
            item = self[i]
            if isinstance(item, MaskedArray):
                result.append(item.tolist())
            elif item is masked:
                result.append(None)
            else:
                result.append(item)
        return result

    def tobytes(self, fill_value=None, order='C'):
        return self.filled(fill_value).tobytes()

    def torecords(self):
        return self.toflex()

    def toflex(self):
        raise NotImplementedError("toflex is not supported in this stub")

    def tofile(self, fid, sep='', format='%s'):
        self.filled().tofile(fid, sep=sep, format=format)

    # -- mask manipulation --

    def harden_mask(self):
        self._hardmask = True
        return self

    def soften_mask(self):
        self._hardmask = False
        return self

    def shrink_mask(self):
        if not _any_true(self.mask):
            self.mask = nomask
        return self

    def unshare_mask(self):
        self.mask = self.mask.copy()
        return self

    def ids(self):
        return (id(self.data), id(self.mask))

    def iscontiguous(self):
        return True

    # -- sorting --

    def sort(self, axis=-1, kind=None, order=None, endwith=True, fill_value=None):
        import numpy as np
        if fill_value is None:
            fill_value = minimum_fill_value(self.data) if endwith else maximum_fill_value(self.data)
        filled = self.filled(fill_value)
        idx = np.argsort(filled, axis=axis)
        self.data = np.take_along_axis(self.data, idx, axis=axis)
        self.mask = np.take_along_axis(self.mask, idx, axis=axis)

    def argsort(self, axis=None, kind=None, order=None, endwith=True, fill_value=None):
        import numpy as np
        if fill_value is None:
            fill_value = minimum_fill_value(self.data) if endwith else maximum_fill_value(self.data)
        return np.argsort(self.filled(fill_value), axis=axis)

    def argmin(self, axis=None, fill_value=None, out=None, keepdims=False):
        import numpy as np
        if fill_value is None:
            fill_value = minimum_fill_value(self.data)
        return np.argmin(self.filled(fill_value), axis=axis)

    def argmax(self, axis=None, fill_value=None, out=None, keepdims=False):
        import numpy as np
        if fill_value is None:
            fill_value = maximum_fill_value(self.data)
        return np.argmax(self.filled(fill_value), axis=axis)

    def argpartition(self, kth, axis=-1, kind='introselect', order=None):
        import numpy as np
        return np.argpartition(self.filled(), kth, axis=axis)

    def partition(self, kth, axis=-1, kind='introselect', order=None):
        import numpy as np
        self.data = np.partition(self.data, kth, axis=axis)

    def searchsorted(self, v, side='left', sorter=None):
        import numpy as np
        return np.searchsorted(self.filled(), v, side=side, sorter=sorter)

    # -- misc --

    def compress(self, condition, axis=None, out=None):
        import numpy as np
        return MaskedArray(np.compress(condition, self.data, axis=axis))

    def put(self, indices, values, mode='raise'):
        import numpy as np
        if isinstance(values, MaskedArray):
            np.put(self.data, indices, values.data)
            np.put(self.mask, indices, values.mask)
        else:
            np.put(self.data, indices, values)
            np.put(self.mask, indices, False)

    def take(self, indices, axis=None, out=None, mode='raise'):
        import numpy as np
        return MaskedArray(np.take(self.data, indices, axis=axis),
                           mask=np.take(self.mask, indices, axis=axis),
                           fill_value=self._fill_value)

    def dot(self, b, out=None, strict=False):
        import numpy as np
        b_data = b.data if isinstance(b, MaskedArray) else np.asarray(b)
        return np.dot(self.filled(0), b_data)

    def clip(self, min=None, max=None, out=None):
        import numpy as np
        return MaskedArray(np.clip(self.data, min, max), mask=self.mask,
                           fill_value=self._fill_value)

    def round(self, decimals=0, out=None):
        import numpy as np
        return MaskedArray(np.round(self.data, decimals), mask=self.mask,
                           fill_value=self._fill_value)

    def choose(self, choices, out=None, mode='raise'):
        import numpy as np
        return np.choose(self.data, choices)

    def conj(self):
        return self.conjugate()

    def conjugate(self):
        import numpy as np
        return MaskedArray(np.conjugate(self.data), mask=self.mask,
                           fill_value=self._fill_value)

    def byteswap(self, inplace=False):
        return MaskedArray(self.data.byteswap(inplace), mask=self.mask,
                           fill_value=self._fill_value)

    def getfield(self, dtype, offset=0):
        return self.data.getfield(dtype, offset)

    def setfield(self, val, dtype, offset=0):
        self.data.setfield(val, dtype, offset)

    def setflags(self, write=None, align=None, uic=None):
        self.data.setflags(write=write, align=align, uic=uic)

    def dump(self, file):
        raise NotImplementedError("dump is not supported in this stub")

    def dumps(self):
        raise NotImplementedError("dumps is not supported in this stub")

    def to_device(self, device, /, *, stream=None):
        return self

    def view(self, dtype=None, type=None):
        return self.copy()

    # -- repr --

    def __repr__(self):
        return "masked_array(data={}, mask={})".format(
            self.data.tolist(), self.mask.tolist())

    def __str__(self):
        return self.__repr__()

    def __array__(self, dtype=None):
        return self.filled()

    def __float__(self):
        return float(self.filled())

    def __int__(self):
        return int(self.filled())

    def __bool__(self):
        c = self.compressed()
        if len(c.tolist()) == 0:
            return False
        return bool(c)


# masked_array is an alias for the MaskedArray constructor in real numpy
masked_array = MaskedArray


# ---------------------------------------------------------------------------
# mvoid stub
# ---------------------------------------------------------------------------

class mvoid:
    """Stub for numpy.ma.mvoid (void scalar for structured masked arrays)."""
    def __init__(self, data, mask=None, dtype=None, fill_value=None):
        self.data = data
        self.mask = mask


# ---------------------------------------------------------------------------
# bool_ stub (numpy.ma.bool_ == numpy.bool_)
# ---------------------------------------------------------------------------

try:
    import numpy as _np
    bool_ = _np.bool_
except Exception:
    bool_ = bool


# ---------------------------------------------------------------------------
# mr_ (masked-array version of r_)
# ---------------------------------------------------------------------------

class mr_class:
    """Translate slice notation to masked array concatenation."""
    def __getitem__(self, key):
        import numpy as np
        if isinstance(key, slice):
            return MaskedArray(np.arange(
                key.start or 0,
                key.stop,
                key.step or 1))
        if isinstance(key, tuple):
            arrays = []
            for k in key:
                if isinstance(k, slice):
                    arrays.append(np.arange(k.start or 0, k.stop, k.step or 1))
                else:
                    arrays.append(np.asarray(k))
            return MaskedArray(np.concatenate(arrays))
        return MaskedArray(np.asarray(key))

mr_ = mr_class()


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _any_true(mask):
    """Check if any element in mask is True (works for bool, ndarray, nomask)."""
    if mask is nomask or mask is False:
        return False
    if isinstance(mask, bool):
        return mask
    try:
        flat = mask.flatten().tolist()
        return any(flat)
    except Exception:
        return bool(mask)


def _default_fill_value_for(data):
    """Return the default fill value for a given array/scalar."""
    import numpy as np
    try:
        dt = np.asarray(data).dtype
    except Exception:
        return 1e20
    kind = str(dt.kind) if hasattr(dt, 'kind') else str(dt)[0]
    if kind in ('i', 'u'):
        return 999999
    if kind == 'f':
        return 1e20
    if kind == 'c':
        return complex(1e20, 0)
    if kind in ('U', 'S', 'O'):
        return 'N/A'
    if kind == 'b':
        return True
    return 1e20


def default_fill_value(obj):
    """Return the default fill value for the argument."""
    import numpy as np
    if isinstance(obj, MaskedArray):
        return _default_fill_value_for(obj.data)
    return _default_fill_value_for(obj)


def minimum_fill_value(obj):
    """Return minimum fill value for obj's dtype (used for max reductions)."""
    import numpy as np
    try:
        dt = np.asarray(obj).dtype if not isinstance(obj, type) else np.dtype(obj)
    except Exception:
        return -1e20
    kind = str(dt.kind) if hasattr(dt, 'kind') else 'f'
    if kind == 'i':
        ii = np.iinfo(dt)
        return int(ii.min)
    if kind == 'u':
        return 0
    if kind == 'f':
        fi = np.finfo(dt)
        return float(-fi.max)
    if kind == 'b':
        return False
    return -1e20


def maximum_fill_value(obj):
    """Return maximum fill value for obj's dtype (used for min reductions)."""
    import numpy as np
    try:
        dt = np.asarray(obj).dtype if not isinstance(obj, type) else np.dtype(obj)
    except Exception:
        return 1e20
    kind = str(dt.kind) if hasattr(dt, 'kind') else 'f'
    if kind == 'i':
        ii = np.iinfo(dt)
        return int(ii.max)
    if kind == 'u':
        ui = np.iinfo(dt)
        return int(ui.max)
    if kind == 'f':
        fi = np.finfo(dt)
        return float(fi.max)
    if kind == 'b':
        return True
    return 1e20


def common_fill_value(a, b):
    """If a and b have the same fill value, return it; otherwise None."""
    fa = a._fill_value if isinstance(a, MaskedArray) else default_fill_value(a)
    fb = b._fill_value if isinstance(b, MaskedArray) else default_fill_value(b)
    if fa == fb:
        return fa
    return None


def set_fill_value(a, fill_value):
    """Set the fill value of a MaskedArray."""
    if isinstance(a, MaskedArray):
        a._fill_value = fill_value


# ---------------------------------------------------------------------------
# Creation functions
# ---------------------------------------------------------------------------

def array(data, mask=None, dtype=None, fill_value=None, keep_mask=True,
          hard_mask=False, shrink=True, copy=False, subok=True, ndmin=0, order=None):
    """Create a masked array."""
    return MaskedArray(data, mask=mask, dtype=dtype, fill_value=fill_value,
                       hard_mask=hard_mask, shrink=shrink, copy=copy, ndmin=ndmin)


def zeros(shape, dtype=None, order='C', **kw):
    """Return a masked array of zeros."""
    import numpy as np
    return MaskedArray(np.zeros(shape, dtype=dtype or "float64"))


def ones(shape, dtype=None, order='C', **kw):
    """Return a masked array of ones."""
    import numpy as np
    return MaskedArray(np.ones(shape, dtype=dtype or "float64"))


def empty(shape, dtype=None, order='C', **kw):
    """Return an empty masked array."""
    import numpy as np
    return MaskedArray(np.zeros(shape, dtype=dtype or "float64"))


def zeros_like(a, dtype=None, order='K', subok=True, shape=None, **kw):
    """Return a masked array of zeros with same shape as a."""
    import numpy as np
    ref = a.data if isinstance(a, MaskedArray) else np.asarray(a)
    s = shape if shape is not None else ref.shape
    d = dtype if dtype is not None else ref.dtype
    return MaskedArray(np.zeros(s, dtype=d))


def ones_like(a, dtype=None, order='K', subok=True, shape=None, **kw):
    """Return a masked array of ones with same shape as a."""
    import numpy as np
    ref = a.data if isinstance(a, MaskedArray) else np.asarray(a)
    s = shape if shape is not None else ref.shape
    d = dtype if dtype is not None else ref.dtype
    return MaskedArray(np.ones(s, dtype=d))


def empty_like(prototype, dtype=None, order='K', subok=True, shape=None, **kw):
    """Return an empty masked array with same shape as prototype."""
    return zeros_like(prototype, dtype=dtype, shape=shape)


def masked_all(shape, dtype=None):
    """Return a masked array of the given shape with all elements masked."""
    import numpy as np
    dt = dtype or "float64"
    return MaskedArray(np.zeros(shape, dtype=dt), mask=True)


def masked_all_like(arr):
    """Return a masked array with all elements masked, like arr."""
    import numpy as np
    ref = arr.data if isinstance(arr, MaskedArray) else np.asarray(arr)
    return MaskedArray(np.zeros(ref.shape, dtype=ref.dtype), mask=True)


def identity(n, dtype=None, **kw):
    """Return the identity masked array of size n."""
    import numpy as np
    return MaskedArray(np.identity(n, dtype=dtype))


def arange(*args, **kwargs):
    """Return a masked array of evenly spaced values."""
    import numpy as np
    fill_value = kwargs.pop('fill_value', None)
    kwargs.pop('hardmask', None)
    return MaskedArray(np.arange(*args, **kwargs), fill_value=fill_value)


def indices(dimensions, dtype=int, sparse=False, **kw):
    """Return an array representing indices of a grid."""
    import numpy as np
    return MaskedArray(np.indices(dimensions, dtype=dtype))


def frombuffer(buffer, dtype=None, count=-1, offset=0, **kw):
    """Create masked array from buffer."""
    import numpy as np
    return MaskedArray(np.frombuffer(buffer, dtype=dtype, count=count, offset=offset))


def fromfunction(function, shape, dtype=float, **kwargs):
    """Construct masked array by executing function over each coordinate."""
    import numpy as np
    return MaskedArray(np.fromfunction(function, shape, dtype=dtype, **kwargs))


def fromflex(fxarray):
    """Build a masked array from a flexible-type array (structured)."""
    raise NotImplementedError("fromflex not supported in this stub")


def asarray(a, dtype=None, order=None):
    """Convert input to masked array."""
    if isinstance(a, MaskedArray):
        if dtype is not None:
            return a.astype(dtype)
        return a
    return MaskedArray(a, dtype=dtype)


def asanyarray(a, dtype=None):
    """Convert input to masked array, passing through subclasses."""
    return asarray(a, dtype=dtype)


# ---------------------------------------------------------------------------
# Masking functions
# ---------------------------------------------------------------------------

def masked_where(condition, x, copy=True):
    """Mask where condition is True."""
    import numpy as np
    return MaskedArray(np.asarray(x), mask=np.asarray(condition))


def masked_equal(x, value):
    """Mask where equal to value."""
    import numpy as np
    x = np.asarray(x)
    return MaskedArray(x, mask=(x == value))


def masked_greater(x, value):
    """Mask where greater than value."""
    import numpy as np
    x = np.asarray(x)
    return MaskedArray(x, mask=(x > value))


def masked_less(x, value):
    """Mask where less than value."""
    import numpy as np
    x = np.asarray(x)
    return MaskedArray(x, mask=(x < value))


def masked_greater_equal(x, value):
    """Mask where greater than or equal to value."""
    import numpy as np
    x = np.asarray(x)
    return MaskedArray(x, mask=(x >= value))


def masked_less_equal(x, value):
    """Mask where less than or equal to value."""
    import numpy as np
    x = np.asarray(x)
    return MaskedArray(x, mask=(x <= value))


def masked_not_equal(x, value):
    """Mask where not equal to value."""
    import numpy as np
    x = np.asarray(x)
    return MaskedArray(x, mask=(x != value))


def masked_inside(x, v1, v2):
    """Mask where between v1 and v2 (inclusive)."""
    import numpy as np
    x = np.asarray(x)
    lo, hi = (v1, v2) if v1 <= v2 else (v2, v1)
    return MaskedArray(x, mask=np.logical_and(x >= lo, x <= hi))


def masked_outside(x, v1, v2):
    """Mask where outside v1 and v2."""
    import numpy as np
    x = np.asarray(x)
    lo, hi = (v1, v2) if v1 <= v2 else (v2, v1)
    return MaskedArray(x, mask=np.logical_or(x < lo, x > hi))


def masked_invalid(x, copy=True):
    """Mask NaN and Inf values."""
    import numpy as np
    x = np.asarray(x)
    mask = np.logical_or(np.isnan(x), np.isinf(x))
    return MaskedArray(x, mask=mask)


def masked_values(x, value, rtol=1e-5, atol=1e-8, copy=True, shrink=True):
    """Mask where approximately equal to value (for floats)."""
    import numpy as np
    x = np.asarray(x)
    mask = np.abs(x - value) <= (atol + rtol * np.abs(value))
    return MaskedArray(x, mask=mask)


def masked_object(x, value, copy=True, shrink=True):
    """Mask where equal to value (for object arrays)."""
    import numpy as np
    x = np.asarray(x)
    return MaskedArray(x, mask=(x == value))


# ---------------------------------------------------------------------------
# Mask query / manipulation
# ---------------------------------------------------------------------------

def is_masked(x):
    """Test whether input has masked values."""
    if isinstance(x, MaskedArray):
        return _any_true(x.mask)
    return False


def is_mask(m):
    """Test whether input is a valid mask."""
    import numpy as np
    if m is nomask:
        return True
    try:
        arr = np.asarray(m)
        return str(arr.dtype) == 'bool'
    except Exception:
        return False


def isMaskedArray(x):
    """Test whether input is a MaskedArray."""
    return isinstance(x, MaskedArray)

isMA = isMaskedArray
isarray = isMaskedArray


def getdata(a, subok=True):
    """Return data of a masked array as ndarray."""
    if isinstance(a, MaskedArray):
        return a.data
    import numpy as np
    return np.asarray(a)


def getmask(a):
    """Return the mask of a masked array, or nomask."""
    if isinstance(a, MaskedArray):
        if _any_true(a.mask):
            return a.mask
        return nomask
    return nomask


def getmaskarray(arr):
    """Return the mask array of a masked array, or full False mask."""
    import numpy as np
    if isinstance(arr, MaskedArray):
        return arr.mask
    return np.zeros(np.asarray(arr).shape, dtype="bool")


def make_mask(m, copy=False, shrink=True, dtype=None):
    """Create a boolean mask from an array."""
    import numpy as np
    if m is nomask or m is False:
        if shrink:
            return nomask
        return np.array(False)
    result = np.asarray(m, dtype="bool")
    if shrink and not _any_true(result):
        return nomask
    if copy:
        return result.copy()
    return result


def make_mask_descr(ndtype):
    """Construct a dtype description for masked structured arrays."""
    import numpy as np
    return np.dtype("bool")


def make_mask_none(newshape, dtype=None):
    """Return a boolean mask of all False with the given shape."""
    import numpy as np
    return np.zeros(newshape, dtype="bool")


def mask_or(m1, m2, copy=False, shrink=True):
    """Combine two masks with logical or."""
    import numpy as np
    if m1 is nomask and m2 is nomask:
        return nomask
    if m1 is nomask:
        return make_mask(m2, copy=copy, shrink=shrink)
    if m2 is nomask:
        return make_mask(m1, copy=copy, shrink=shrink)
    return np.logical_or(m1, m2)


def flatten_mask(mask):
    """Flatten a structured mask into a simple boolean mask."""
    import numpy as np
    if mask is nomask:
        return nomask
    return np.asarray(mask, dtype="bool").flatten()


def flatten_structured_array(a):
    """Flatten a structured array."""
    import numpy as np
    return np.asarray(a).flatten()


def mask_rowcols(a, axis=None):
    """Mask whole rows and/or columns of a 2D array that contain masked values."""
    import numpy as np
    if not isinstance(a, MaskedArray):
        return a
    if a.ndim != 2:
        raise NotImplementedError("mask_rowcols only works on 2D arrays")
    m = a.mask
    if not _any_true(m):
        return a
    row_mask = np.any(m, axis=1)
    col_mask = np.any(m, axis=0)
    new_mask = m.copy()
    if axis is None or axis == 0:
        for i in range(len(row_mask.tolist())):
            if row_mask.tolist()[i]:
                new_mask[i] = True
    if axis is None or axis == 1:
        for j in range(len(col_mask.tolist())):
            if col_mask.tolist()[j]:
                for i in range(new_mask.shape[0]):
                    new_mask[i][j] = True
    return MaskedArray(a.data, mask=new_mask, fill_value=a._fill_value)


def mask_rows(a, axis=None):
    """Mask rows of a 2D array that contain masked values."""
    return mask_rowcols(a, axis=0)


def mask_cols(a, axis=None):
    """Mask cols of a 2D array that contain masked values."""
    return mask_rowcols(a, axis=1)


def harden_mask(a):
    """Force the mask of a to be hard."""
    if isinstance(a, MaskedArray):
        a._hardmask = True
    return a


def soften_mask(a):
    """Force the mask of a to be soft."""
    if isinstance(a, MaskedArray):
        a._hardmask = False
    return a


# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------

def filled(a, fill_value=None):
    """Return input with masked values replaced by fill_value."""
    if isinstance(a, MaskedArray):
        return a.filled(fill_value)
    import numpy as np
    return np.asarray(a)


def compressed(x):
    """Return all non-masked data as a 1-D array."""
    if isinstance(x, MaskedArray):
        return x.compressed()
    import numpy as np
    return np.asarray(x).flatten()


def fix_invalid(a, mask=None, copy=True, fill_value=None):
    """Return with invalid data (NaN/Inf) masked and replaced."""
    import numpy as np
    a = np.asarray(a)
    invalid_mask = np.logical_or(np.isnan(a), np.isinf(a))
    if mask is not None:
        combined = np.logical_or(invalid_mask, np.asarray(mask))
    else:
        combined = invalid_mask
    return MaskedArray(a, mask=combined, fill_value=fill_value)


def count(a, axis=None, keepdims=False):
    """Count non-masked elements."""
    if isinstance(a, MaskedArray):
        return a.count(axis=axis, keepdims=keepdims)
    import numpy as np
    return np.asarray(a).size


def count_masked(arr, axis=None):
    """Count masked elements."""
    if isinstance(arr, MaskedArray):
        import numpy as np
        return int(np.sum(arr.mask.astype("int64"), axis=axis))
    return 0


def ids(a):
    """Return (id(data), id(mask))."""
    if isinstance(a, MaskedArray):
        return a.ids()
    return (id(a), id(nomask))


def ndim(obj):
    """Return number of dimensions."""
    if isinstance(obj, MaskedArray):
        return obj.ndim
    import numpy as np
    return np.ndim(obj)


def shape(obj):
    """Return shape of object."""
    if isinstance(obj, MaskedArray):
        return obj.shape
    import numpy as np
    return np.shape(obj)


def size(obj, axis=None):
    """Return number of elements along given axis."""
    if isinstance(obj, MaskedArray):
        if axis is None:
            return obj.size
        return obj.shape[axis]
    import numpy as np
    return np.size(obj, axis=axis)


def copy(a, *args, **params):
    """Return a copy of a masked array."""
    if isinstance(a, MaskedArray):
        return a.copy()
    import numpy as np
    return np.copy(a)


def ndenumerate(a, compressed=True):
    """Multidimensional index iterator for masked arrays."""
    import numpy as np
    if isinstance(a, MaskedArray):
        for idx in np.ndindex(a.shape):
            if compressed and a.mask[idx]:
                continue
            yield idx, a.data[idx]
    else:
        arr = np.asarray(a)
        for idx in np.ndindex(arr.shape):
            yield idx, arr[idx]


# ---------------------------------------------------------------------------
# Ufunc wrappers (unary)
# ---------------------------------------------------------------------------

def _apply_unary(func, x, fill=0.0):
    """Apply a unary numpy function, preserving mask."""
    import numpy as np
    if isinstance(x, MaskedArray):
        result_data = func(x.filled(fill))
        return MaskedArray(result_data, mask=x.mask, fill_value=x._fill_value)
    return func(np.asarray(x))


def _apply_binary(func, a, b, fill=0.0):
    """Apply a binary numpy function, merging masks."""
    import numpy as np
    a_ma = isinstance(a, MaskedArray)
    b_ma = isinstance(b, MaskedArray)
    a_data = a.filled(fill) if a_ma else np.asarray(a)
    b_data = b.filled(fill) if b_ma else np.asarray(b)
    result_data = func(a_data, b_data)
    if a_ma and b_ma:
        m = np.logical_or(a.mask, b.mask)
    elif a_ma:
        m = a.mask
    elif b_ma:
        m = b.mask
    else:
        return result_data
    return MaskedArray(result_data, mask=m)


# -- unary math --

def log(x):
    import numpy as np
    return _apply_unary(np.log, x, fill=1.0)

def log2(x):
    import numpy as np
    return _apply_unary(np.log2, x, fill=1.0)

def log10(x):
    import numpy as np
    return _apply_unary(np.log10, x, fill=1.0)

def exp(x):
    import numpy as np
    return _apply_unary(np.exp, x)

def sqrt(x):
    import numpy as np
    return _apply_unary(np.sqrt, x)

def absolute(x):
    import numpy as np
    return _apply_unary(np.abs, x)

abs = absolute

def fabs(x):
    import numpy as np
    return _apply_unary(np.fabs, x)

def negative(x):
    import numpy as np
    return _apply_unary(np.negative, x)

def conjugate(x):
    import numpy as np
    return _apply_unary(np.conjugate, x)

def sin(x):
    import numpy as np
    return _apply_unary(np.sin, x)

def cos(x):
    import numpy as np
    return _apply_unary(np.cos, x)

def tan(x):
    import numpy as np
    return _apply_unary(np.tan, x)

def arcsin(x):
    import numpy as np
    return _apply_unary(np.arcsin, x)

def arccos(x):
    import numpy as np
    return _apply_unary(np.arccos, x)

def arctan(x):
    import numpy as np
    return _apply_unary(np.arctan, x)

def sinh(x):
    import numpy as np
    return _apply_unary(np.sinh, x)

def cosh(x):
    import numpy as np
    return _apply_unary(np.cosh, x)

def tanh(x):
    import numpy as np
    return _apply_unary(np.tanh, x)

def arcsinh(x):
    import numpy as np
    return _apply_unary(np.arcsinh, x)

def arccosh(x):
    import numpy as np
    return _apply_unary(np.arccosh, x, fill=1.0)

def arctanh(x):
    import numpy as np
    return _apply_unary(np.arctanh, x)

def floor(x):
    import numpy as np
    return _apply_unary(np.floor, x)

def ceil(x):
    import numpy as np
    return _apply_unary(np.ceil, x)

def around(x, decimals=0, out=None):
    import numpy as np
    return _apply_unary(lambda v: np.round(v, decimals), x)

def angle(x):
    import numpy as np
    return _apply_unary(np.angle, x)

def logical_not(x):
    import numpy as np
    return _apply_unary(np.logical_not, x)


# -- _MaskedBinaryFunc: callable with .reduce/.accumulate support --

class _MaskedBinaryFunc:
    """Wraps a binary function with reduce/accumulate methods for ma ufunc compatibility."""

    def __init__(self, np_func_name, fill=0.0):
        self._np_func_name = np_func_name
        self._fill = fill

    def _get_np_func(self):
        import numpy as np
        return getattr(np, self._np_func_name)

    def __call__(self, a, b, *args, **kwargs):
        return _apply_binary(self._get_np_func(), a, b, fill=self._fill)

    def reduce(self, a, axis=0, **kwargs):
        """Reduce using the underlying numpy ufunc, respecting masks."""
        import numpy as np
        np_func = self._get_np_func()
        a_ma = isinstance(a, MaskedArray)
        if a_ma:
            a_data = a.filled(self._fill)
            mask = a.mask
        else:
            a_data = np.asarray(a)
            mask = None
        # Use the numpy ufunc's reduce
        result = np_func.reduce(a_data, axis=axis, **kwargs)
        return result

    def accumulate(self, a, axis=0, **kwargs):
        """Accumulate using the underlying numpy ufunc, respecting masks."""
        import numpy as np
        np_func = self._get_np_func()
        a_ma = isinstance(a, MaskedArray)
        if a_ma:
            a_data = a.filled(self._fill)
        else:
            a_data = np.asarray(a)
        result = np_func.accumulate(a_data, axis=axis, **kwargs)
        return result

    def outer(self, a, b, **kwargs):
        """Outer using the underlying numpy ufunc."""
        import numpy as np
        np_func = self._get_np_func()
        a_data = a.filled(self._fill) if isinstance(a, MaskedArray) else np.asarray(a)
        b_data = b.filled(self._fill) if isinstance(b, MaskedArray) else np.asarray(b)
        return np_func.outer(a_data, b_data, **kwargs)


# -- binary math --

add = _MaskedBinaryFunc('add', fill=0.0)

def subtract(a, b):
    import numpy as np
    return _apply_binary(np.subtract, a, b)

def multiply(a, b):
    import numpy as np
    return _apply_binary(np.multiply, a, b)

def divide(a, b):
    import numpy as np
    return _apply_binary(np.divide, a, b, fill=1.0)

true_divide = divide

def floor_divide(a, b):
    import numpy as np
    return _apply_binary(np.floor_divide, a, b, fill=1.0)

def remainder(a, b):
    import numpy as np
    return _apply_binary(np.remainder, a, b, fill=1.0)

mod = remainder

def fmod(a, b):
    import numpy as np
    return _apply_binary(np.fmod, a, b, fill=1.0)

def power(a, b, third=None):
    import numpy as np
    return _apply_binary(np.power, a, b, fill=1.0)

maximum = _MaskedBinaryFunc('maximum', fill=0.0)

minimum = _MaskedBinaryFunc('minimum', fill=0.0)

def hypot(a, b):
    import numpy as np
    return _apply_binary(np.hypot, a, b)

def arctan2(a, b):
    import numpy as np
    return _apply_binary(np.arctan2, a, b)

# comparison
def equal(a, b):
    import numpy as np
    return _apply_binary(np.equal, a, b)

def not_equal(a, b):
    import numpy as np
    return _apply_binary(np.not_equal, a, b)

def greater(a, b):
    import numpy as np
    return _apply_binary(np.greater, a, b)

def greater_equal(a, b):
    import numpy as np
    return _apply_binary(np.greater_equal, a, b)

def less(a, b):
    import numpy as np
    return _apply_binary(np.less, a, b)

def less_equal(a, b):
    import numpy as np
    return _apply_binary(np.less_equal, a, b)

# logical
def logical_and(a, b):
    import numpy as np
    return _apply_binary(np.logical_and, a, b)

def logical_or(a, b):
    import numpy as np
    return _apply_binary(np.logical_or, a, b)

def logical_xor(a, b):
    import numpy as np
    return _apply_binary(np.logical_xor, a, b)

# bitwise
def bitwise_and(a, b):
    import numpy as np
    return _apply_binary(np.bitwise_and, a, b)

def bitwise_or(a, b):
    import numpy as np
    return _apply_binary(np.bitwise_or, a, b)

def bitwise_xor(a, b):
    import numpy as np
    return _apply_binary(np.bitwise_xor, a, b)

def left_shift(a, n):
    import numpy as np
    return _apply_binary(np.left_shift, a, n)

def right_shift(a, n):
    import numpy as np
    return _apply_binary(np.right_shift, a, n)


# ---------------------------------------------------------------------------
# Aggregation / reduction functions (module-level)
# ---------------------------------------------------------------------------

def sum(a, axis=None, dtype=None, out=None, keepdims=False):
    if isinstance(a, MaskedArray):
        return a.sum(axis=axis, keepdims=keepdims)
    import numpy as np
    return np.sum(a, axis=axis, keepdims=keepdims)

def prod(a, axis=None, dtype=None, out=None, keepdims=False):
    if isinstance(a, MaskedArray):
        return a.prod(axis=axis, keepdims=keepdims)
    import numpy as np
    return np.prod(a, axis=axis, keepdims=keepdims)

product = prod

def mean(a, axis=None, dtype=None, out=None, keepdims=False):
    if isinstance(a, MaskedArray):
        return a.mean(axis=axis, keepdims=keepdims)
    import numpy as np
    return np.mean(a, axis=axis, keepdims=keepdims)

def var(a, axis=None, dtype=None, out=None, ddof=0, keepdims=False, mean=None):
    if isinstance(a, MaskedArray):
        return a.var(axis=axis, ddof=ddof, keepdims=keepdims, mean=mean)
    import numpy as np
    return np.var(a, axis=axis, ddof=ddof, keepdims=keepdims)

def std(a, axis=None, dtype=None, out=None, ddof=0, keepdims=False, mean=None):
    if isinstance(a, MaskedArray):
        return a.std(axis=axis, ddof=ddof, keepdims=keepdims, mean=mean)
    import numpy as np
    return np.std(a, axis=axis, ddof=ddof, keepdims=keepdims)

def min(obj, axis=None, out=None, fill_value=None, keepdims=False):
    if isinstance(obj, MaskedArray):
        return obj.min(axis=axis, fill_value=fill_value, keepdims=keepdims)
    import numpy as np
    return np.min(obj, axis=axis, keepdims=keepdims)

# np.ma uses "amin" as the dispatcher name
amax = max
amin = min

def max(obj, axis=None, out=None, fill_value=None, keepdims=False):
    if isinstance(obj, MaskedArray):
        return obj.max(axis=axis, fill_value=fill_value, keepdims=keepdims)
    import numpy as np
    return np.max(obj, axis=axis, keepdims=keepdims)

amax = max
amin = min

def ptp(obj, axis=None, out=None, fill_value=None, keepdims=False):
    if isinstance(obj, MaskedArray):
        return obj.ptp(axis=axis, fill_value=fill_value)
    import numpy as np
    return np.ptp(obj, axis=axis, keepdims=keepdims)

def all(a, axis=None, out=None, keepdims=False):
    if isinstance(a, MaskedArray):
        return a.all(axis=axis)
    import numpy as np
    return np.all(a, axis=axis)

def any(a, axis=None, out=None, keepdims=False):
    if isinstance(a, MaskedArray):
        return a.any(axis=axis)
    import numpy as np
    return np.any(a, axis=axis)

# alltrue / sometrue are deprecated aliases in numpy but still present in ma
alltrue = all
sometrue = any

def allequal(a, b, fill_value=True):
    """Test whether two arrays are element-wise equal (ignoring masked)."""
    import numpy as np
    if isinstance(a, MaskedArray):
        a = a.compressed()
    if isinstance(b, MaskedArray):
        b = b.compressed()
    a = np.asarray(a)
    b = np.asarray(b)
    if a.size != b.size:
        return False
    return bool(np.all(a == b))

def allclose(a, b, masked_equal=True, rtol=1e-5, atol=1e-8):
    import numpy as np
    if isinstance(a, MaskedArray):
        a = a.compressed()
    if isinstance(b, MaskedArray):
        b = b.compressed()
    return bool(np.allclose(a, b, rtol=rtol, atol=atol))

def average(a, axis=None, weights=None, returned=False, keepdims=False):
    import numpy as np
    if isinstance(a, MaskedArray):
        data = a.filled(0.0)
        not_mask = np.logical_not(a.mask).astype("float64")
        if weights is not None:
            w = np.asarray(weights) * not_mask
        else:
            w = not_mask
        s = np.sum(data * w, axis=axis, keepdims=keepdims)
        sw = np.sum(w, axis=axis, keepdims=keepdims)
        result = s / sw
        if returned:
            return result, sw
        return result
    result = np.average(a, axis=axis, weights=weights)
    if returned:
        if weights is None:
            return result, np.sum(np.ones_like(np.asarray(a)), axis=axis)
        return result, np.sum(np.asarray(weights), axis=axis)
    return result

def median(a, axis=None, out=None, overwrite_input=False, keepdims=False):
    import numpy as np
    if isinstance(a, MaskedArray):
        c = a.compressed()
        if len(c.tolist()) == 0:
            return float('nan')
        return float(np.median(c))
    return np.median(a, axis=axis, keepdims=keepdims)

def anom(a, axis=None, dtype=None):
    """Return anomalies (deviations from the mean)."""
    if isinstance(a, MaskedArray):
        return a.anom(axis=axis)
    import numpy as np
    arr = np.asarray(a)
    return arr - np.mean(arr, axis=axis, keepdims=True)

anomalies = anom

def cumsum(a, axis=None, dtype=None, out=None):
    if isinstance(a, MaskedArray):
        return a.cumsum(axis=axis)
    import numpy as np
    return np.cumsum(a, axis=axis)

def cumprod(a, axis=None, dtype=None, out=None):
    if isinstance(a, MaskedArray):
        return a.cumprod(axis=axis)
    import numpy as np
    return np.cumprod(a, axis=axis)


# ---------------------------------------------------------------------------
# Array manipulation
# ---------------------------------------------------------------------------

def concatenate(arrays, axis=0):
    import numpy as np
    data_list = []
    mask_list = []
    has_ma = False
    for a in arrays:
        if isinstance(a, MaskedArray):
            data_list.append(a.data)
            mask_list.append(a.mask)
            has_ma = True
        else:
            arr = np.asarray(a)
            data_list.append(arr)
            mask_list.append(np.zeros(arr.shape, dtype="bool"))
    d = np.concatenate(data_list, axis=axis)
    if has_ma:
        m = np.concatenate(mask_list, axis=axis)
        return MaskedArray(d, mask=m)
    return d

def reshape(a, new_shape, order='C'):
    if isinstance(a, MaskedArray):
        return a.reshape(new_shape)
    import numpy as np
    return np.reshape(a, new_shape)

def resize(x, new_shape):
    import numpy as np
    if isinstance(x, MaskedArray):
        return MaskedArray(np.resize(x.data, new_shape),
                           mask=np.resize(x.mask, new_shape),
                           fill_value=x._fill_value)
    return np.resize(x, new_shape)

def ravel(a, order='C'):
    if isinstance(a, MaskedArray):
        return a.ravel(order=order)
    import numpy as np
    return np.ravel(a, order=order)

def transpose(a, axes=None):
    if isinstance(a, MaskedArray):
        return a.transpose(*axes) if axes else a.transpose()
    import numpy as np
    return np.transpose(a, axes)

def swapaxes(a, *args, **params):
    if isinstance(a, MaskedArray) and len(args) >= 2:
        return a.swapaxes(args[0], args[1])
    import numpy as np
    return np.swapaxes(a, *args, **params)

def expand_dims(a, axis):
    import numpy as np
    if isinstance(a, MaskedArray):
        return MaskedArray(np.expand_dims(a.data, axis),
                           mask=np.expand_dims(a.mask, axis),
                           fill_value=a._fill_value)
    return np.expand_dims(a, axis)

def squeeze(a, axis=None, **kw):
    if isinstance(a, MaskedArray):
        return a.squeeze(axis=axis)
    import numpy as np
    return np.squeeze(a, axis=axis)

def stack(arrays, axis=0, out=None, **kw):
    import numpy as np
    data_list = []
    mask_list = []
    has_ma = False
    for a in arrays:
        if isinstance(a, MaskedArray):
            data_list.append(a.data)
            mask_list.append(a.mask)
            has_ma = True
        else:
            arr = np.asarray(a)
            data_list.append(arr)
            mask_list.append(np.zeros(arr.shape, dtype="bool"))
    d = np.stack(data_list, axis=axis)
    if has_ma:
        m = np.stack(mask_list, axis=axis)
        return MaskedArray(d, mask=m)
    return d

def vstack(tup, **kw):
    arrays = [atleast_2d(a) for a in tup]
    return concatenate(arrays, axis=0)

def hstack(tup, **kw):
    import numpy as np
    # hstack is concat along axis=1 for 2d+, axis=0 for 1d
    first = tup[0]
    ref = first.data if isinstance(first, MaskedArray) else np.asarray(first)
    if ref.ndim == 1:
        return concatenate(tup, axis=0)
    return concatenate(tup, axis=1)

def dstack(tup):
    import numpy as np
    # stack along third axis
    arrays = []
    for a in tup:
        if isinstance(a, MaskedArray):
            d = a.data
            m = a.mask
        else:
            d = np.asarray(a)
            m = np.zeros(d.shape, dtype="bool")
        # ensure at least 3d
        while d.ndim < 3:
            d = np.expand_dims(d, axis=-1) if d.ndim < 2 else np.expand_dims(d, axis=2)
            m = np.expand_dims(m, axis=-1) if m.ndim < 2 else np.expand_dims(m, axis=2)
        arrays.append((d, m))
    data_list = [a[0] for a in arrays]
    mask_list = [a[1] for a in arrays]
    return MaskedArray(np.concatenate(data_list, axis=2),
                       mask=np.concatenate(mask_list, axis=2))

row_stack = vstack

def column_stack(tup):
    import numpy as np
    arrays = []
    for a in tup:
        if isinstance(a, MaskedArray):
            d, m = a.data, a.mask
            if d.ndim == 1:
                d = d.reshape(-1, 1)
                m = m.reshape(-1, 1)
            arrays.append(MaskedArray(d, mask=m))
        else:
            arr = np.asarray(a)
            if arr.ndim == 1:
                arr = arr.reshape(-1, 1)
            arrays.append(arr)
    return concatenate(arrays, axis=1)

def hsplit(ary, indices_or_sections):
    import numpy as np
    if isinstance(ary, MaskedArray):
        data_parts = np.hsplit(ary.data, indices_or_sections)
        mask_parts = np.hsplit(ary.mask, indices_or_sections)
        return [MaskedArray(d, mask=m, fill_value=ary._fill_value)
                for d, m in zip(data_parts, mask_parts)]
    return np.hsplit(ary, indices_or_sections)

def append(a, b, axis=None):
    if axis is None:
        a = ravel(a)
        b = ravel(b)
        return concatenate([a, b])
    return concatenate([a, b], axis=axis)

def repeat(a, *args, **params):
    if isinstance(a, MaskedArray):
        return a.repeat(*args, **params)
    import numpy as np
    return np.repeat(a, *args, **params)

def diagonal(a, *args, **params):
    if isinstance(a, MaskedArray):
        return a.diagonal(*args, **params)
    import numpy as np
    return np.diagonal(a, *args, **params)

def nonzero(a):
    if isinstance(a, MaskedArray):
        return a.nonzero()
    import numpy as np
    return np.nonzero(a)


# ---------------------------------------------------------------------------
# Sorting & searching
# ---------------------------------------------------------------------------

def sort(a, axis=-1, kind=None, order=None, endwith=True, fill_value=None, stable=None):
    import numpy as np
    if isinstance(a, MaskedArray):
        fv = fill_value
        if fv is None:
            fv = minimum_fill_value(a.data) if endwith else maximum_fill_value(a.data)
        idx = np.argsort(a.filled(fv), axis=axis)
        if axis is None:
            return MaskedArray(np.take(a.data.flatten(), idx),
                               mask=np.take(a.mask.flatten(), idx))
        d = np.take_along_axis(a.data, idx, axis=axis)
        m = np.take_along_axis(a.mask, idx, axis=axis)
        return MaskedArray(d, mask=m, fill_value=a._fill_value)
    return np.sort(a, axis=axis)

def argsort(a, axis=None, kind=None, order=None, endwith=True, fill_value=None, stable=None):
    import numpy as np
    if isinstance(a, MaskedArray):
        fv = fill_value
        if fv is None:
            fv = minimum_fill_value(a.data) if endwith else maximum_fill_value(a.data)
        return np.argsort(a.filled(fv), axis=axis)
    return np.argsort(a, axis=axis)

def argmin(a, axis=None, fill_value=None, out=None, keepdims=False):
    if isinstance(a, MaskedArray):
        return a.argmin(axis=axis, fill_value=fill_value)
    import numpy as np
    return np.argmin(a, axis=axis)

def argmax(a, axis=None, fill_value=None, out=None, keepdims=False):
    if isinstance(a, MaskedArray):
        return a.argmax(axis=axis, fill_value=fill_value)
    import numpy as np
    return np.argmax(a, axis=axis)

def where(condition, x=None, y=None):
    import numpy as np
    if x is None and y is None:
        if isinstance(condition, MaskedArray):
            return np.where(condition.filled(False))
        return np.where(condition)
    x_data = x.data if isinstance(x, MaskedArray) else np.asarray(x)
    y_data = y.data if isinstance(y, MaskedArray) else np.asarray(y)
    cond = condition.data if isinstance(condition, MaskedArray) else np.asarray(condition)
    result = np.where(cond, x_data, y_data)
    # merge masks
    has_ma = isinstance(x, MaskedArray) or isinstance(y, MaskedArray)
    if has_ma:
        x_m = x.mask if isinstance(x, MaskedArray) else np.zeros(np.asarray(x).shape, dtype="bool")
        y_m = y.mask if isinstance(y, MaskedArray) else np.zeros(np.asarray(y).shape, dtype="bool")
        m = np.where(cond, x_m, y_m)
        return MaskedArray(result, mask=m)
    return result

def choose(indices, choices, out=None, mode='raise'):
    import numpy as np
    return np.choose(indices, choices)


# ---------------------------------------------------------------------------
# Element-level operations
# ---------------------------------------------------------------------------

def clip(a, a_min=None, a_max=None, out=None, **kw):
    import numpy as np
    if isinstance(a, MaskedArray):
        return a.clip(min=a_min, max=a_max)
    return np.clip(a, a_min, a_max)

def round(a, decimals=0, out=None):
    if isinstance(a, MaskedArray):
        return a.round(decimals)
    import numpy as np
    return np.round(a, decimals)

round_ = round

def compress(condition, a, axis=None, out=None):
    import numpy as np
    if isinstance(a, MaskedArray):
        return MaskedArray(np.compress(condition, a.data, axis=axis))
    return np.compress(condition, a, axis=axis)

def take(a, indices, axis=None, out=None, mode='raise'):
    if isinstance(a, MaskedArray):
        return a.take(indices, axis=axis)
    import numpy as np
    return np.take(a, indices, axis=axis)

def put(a, indices, values, mode='raise'):
    if isinstance(a, MaskedArray):
        a.put(indices, values, mode=mode)
    else:
        import numpy as np
        np.put(a, indices, values, mode=mode)

def putmask(a, mask, values):
    import numpy as np
    if isinstance(a, MaskedArray):
        np.putmask(a.data, mask, values)
    else:
        np.putmask(a, mask, values)

def diag(v, k=0):
    import numpy as np
    if isinstance(v, MaskedArray):
        return MaskedArray(np.diag(v.data, k), mask=np.diag(v.mask, k),
                           fill_value=v._fill_value)
    return np.diag(v, k)

def diagflat(v, k=0):
    import numpy as np
    if isinstance(v, MaskedArray):
        return MaskedArray(np.diagflat(v.data, k), mask=np.diagflat(v.mask, k),
                           fill_value=v._fill_value)
    return np.diagflat(v, k)

def dot(a, b, strict=False, out=None):
    import numpy as np
    a_data = a.filled(0) if isinstance(a, MaskedArray) else np.asarray(a)
    b_data = b.filled(0) if isinstance(b, MaskedArray) else np.asarray(b)
    return np.dot(a_data, b_data)

def inner(a, b):
    import numpy as np
    a_data = a.filled(0) if isinstance(a, MaskedArray) else np.asarray(a)
    b_data = b.filled(0) if isinstance(b, MaskedArray) else np.asarray(b)
    return np.inner(a_data, b_data)

innerproduct = inner

def outer(a, b):
    import numpy as np
    a_data = a.compressed() if isinstance(a, MaskedArray) else np.asarray(a).flatten()
    b_data = b.compressed() if isinstance(b, MaskedArray) else np.asarray(b).flatten()
    return np.outer(a_data, b_data)

outerproduct = outer

def trace(a, offset=0, axis1=0, axis2=1, dtype=None, out=None):
    if isinstance(a, MaskedArray):
        return a.trace(offset=offset, axis1=axis1, axis2=axis2)
    import numpy as np
    return np.trace(a, offset=offset, axis1=axis1, axis2=axis2)

def diff(a, n=1, axis=-1, prepend=None, append=None):
    import numpy as np
    if isinstance(a, MaskedArray):
        return MaskedArray(np.diff(a.data, n=n, axis=axis))
    return np.diff(a, n=n, axis=axis)

def convolve(a, v, mode='full', propagate_mask=True):
    import numpy as np
    a_data = a.filled(0) if isinstance(a, MaskedArray) else np.asarray(a)
    v_data = v.filled(0) if isinstance(v, MaskedArray) else np.asarray(v)
    return np.convolve(a_data, v_data, mode=mode)

def correlate(a, v, mode='valid', propagate_mask=True):
    import numpy as np
    a_data = a.filled(0) if isinstance(a, MaskedArray) else np.asarray(a)
    v_data = v.filled(0) if isinstance(v, MaskedArray) else np.asarray(v)
    return np.correlate(a_data, v_data, mode=mode)


# ---------------------------------------------------------------------------
# Set operations
# ---------------------------------------------------------------------------

def unique(ar, return_index=False, return_inverse=False):
    """Return the unique values of a masked array."""
    import numpy as np
    if isinstance(ar, MaskedArray):
        # Get data and mask
        data = np.asarray(ar._data) if hasattr(ar, '_data') else np.asarray(ar.data)
        m = ar._mask if hasattr(ar, '_mask') else getattr(ar, 'mask', nomask)
        if m is not nomask and m is not False:
            if isinstance(m, (list, tuple)):
                m = np.asarray(m).astype('bool')
            elif isinstance(m, np.ndarray):
                pass
            else:
                m = np.asarray([m]).astype('bool')
            # Get unmasked data
            unmasked_data = []
            unmasked_indices = []
            for i in range(len(data)):
                mi = m[i] if i < len(m) else False
                if isinstance(mi, np.ndarray):
                    mi = bool(mi)
                if not mi:
                    unmasked_data.append(float(data[i]) if hasattr(data, '__getitem__') else float(data))
                    unmasked_indices.append(i)
            has_masked = len(unmasked_data) < len(data)
            if len(unmasked_data) == 0:
                # All masked
                result_data = np.array([float(data[0]) if hasattr(data, '__getitem__') else float(data)])
                result_mask = np.array([True])
                result = MaskedArray(result_data, mask=result_mask)
                if return_index or return_inverse:
                    ret = [result]
                    if return_index:
                        ret.append(np.array([0]))
                    if return_inverse:
                        ret.append(np.zeros(len(data), dtype='int64'))
                    return tuple(ret)
                return result
            # Get unique from unmasked data
            unmasked_arr = np.array(unmasked_data)
            res = np.unique(unmasked_arr, return_index=True, return_inverse=True)
            uvals = res[0]
            uidx_in_unmasked = res[1]
            uinv_in_unmasked = res[2]
            # Map indices back to original array
            orig_indices = np.array(unmasked_indices)
            uidx = np.array([int(orig_indices[int(i)]) for i in uidx_in_unmasked])
            # Build full inverse mapping
            uinv = np.zeros(len(data), dtype='int64')
            for i, idx in enumerate(unmasked_indices):
                uinv[idx] = int(uinv_in_unmasked[i])
            if has_masked:
                # Add masked entry at end
                result_data = np.concatenate([uvals, np.array([-1.0])])
                result_mask_data = [False] * len(uvals) + [True]
                result = MaskedArray(result_data, mask=result_mask_data)
                masked_idx = len(uvals)  # index of the masked entry
                # Update indices
                first_masked_orig = -1
                for i in range(len(data)):
                    mi = m[i] if i < len(m) else False
                    if isinstance(mi, np.ndarray):
                        mi = bool(mi)
                    if mi:
                        if first_masked_orig < 0:
                            first_masked_orig = i
                        uinv[i] = masked_idx
                uidx = np.concatenate([uidx, np.array([first_masked_orig])])
            else:
                result = MaskedArray(uvals, mask=False)
            if return_index or return_inverse:
                ret = [result]
                if return_index:
                    ret.append(uidx)
                if return_inverse:
                    ret.append(uinv)
                return tuple(ret)
            return result
        else:
            # No mask
            res = np.unique(data, return_index=return_index, return_inverse=return_inverse)
            if isinstance(res, tuple):
                return (MaskedArray(res[0], mask=False),) + res[1:]
            return MaskedArray(res, mask=False)
    else:
        data = np.asarray(ar)
        res = np.unique(data, return_index=return_index, return_inverse=return_inverse)
        if isinstance(res, tuple):
            return (MaskedArray(res[0], mask=False),) + res[1:]
        return MaskedArray(res, mask=False)

def ediff1d(ary, to_end=None, to_begin=None):
    """Compute the differences between consecutive elements of a masked array."""
    import numpy as np
    if isinstance(ary, MaskedArray):
        data = np.asarray(ary._data) if hasattr(ary, '_data') else np.asarray(ary.data)
        m = ary._mask if hasattr(ary, '_mask') else getattr(ary, 'mask', nomask)
        # Compute differences
        diff_data = np.ediff1d(data)
        # Build mask: diff[i] is masked if ary[i] or ary[i+1] is masked
        if m is not nomask and m is not False:
            if isinstance(m, (list, tuple)):
                m = np.asarray(m).astype('bool')
            diff_mask = []
            for i in range(len(diff_data)):
                m_i = bool(m[i]) if i < len(m) else False
                m_ip1 = bool(m[i + 1]) if (i + 1) < len(m) else False
                diff_mask.append(m_i or m_ip1)
        else:
            diff_mask = [False] * len(diff_data)
        # Handle to_begin and to_end
        result_data = list(diff_data.tolist())
        result_mask = list(diff_mask)
        if to_begin is not None:
            if isinstance(to_begin, MaskedConstant) or to_begin is masked:
                result_data = [0.0] + result_data
                result_mask = [True] + result_mask
            elif isinstance(to_begin, (list, tuple)):
                result_data = list(to_begin) + result_data
                result_mask = [False] * len(to_begin) + result_mask
            elif isinstance(to_begin, MaskedArray):
                tb_data = to_begin._data if hasattr(to_begin, '_data') else to_begin.data
                tb_mask = to_begin._mask if hasattr(to_begin, '_mask') else getattr(to_begin, 'mask', nomask)
                if isinstance(tb_data, (list, tuple)):
                    result_data = list(tb_data) + result_data
                else:
                    result_data = list(np.asarray(tb_data).tolist()) + result_data
                if tb_mask is nomask or tb_mask is False:
                    result_mask = [False] * len(tb_data) + result_mask
                else:
                    result_mask = list(tb_mask) + result_mask
            else:
                result_data = [float(to_begin)] + result_data
                result_mask = [False] + result_mask
        if to_end is not None:
            if isinstance(to_end, MaskedConstant) or to_end is masked:
                result_data = result_data + [0.0]
                result_mask = result_mask + [True]
            elif isinstance(to_end, (list, tuple)):
                result_data = result_data + list(to_end)
                result_mask = result_mask + [False] * len(to_end)
            elif isinstance(to_end, MaskedArray):
                te_data = to_end._data if hasattr(to_end, '_data') else to_end.data
                te_mask = to_end._mask if hasattr(to_end, '_mask') else getattr(to_end, 'mask', nomask)
                if isinstance(te_data, (list, tuple)):
                    result_data = result_data + list(te_data)
                else:
                    result_data = result_data + list(np.asarray(te_data).tolist())
                if te_mask is nomask or te_mask is False:
                    result_mask = result_mask + [False] * len(te_data)
                else:
                    result_mask = result_mask + list(te_mask)
            else:
                result_data = result_data + [float(to_end)]
                result_mask = result_mask + [False]
        return MaskedArray(np.array(result_data), mask=result_mask)
    else:
        result = np.ediff1d(ary, to_end=to_end, to_begin=to_begin)
        return MaskedArray(result, mask=False)

def intersect1d(ar1, ar2, assume_unique=False):
    """Return the intersection of two masked arrays."""
    import numpy as np
    if isinstance(ar1, MaskedArray): ar1 = ar1.compressed()
    if isinstance(ar2, MaskedArray): ar2 = ar2.compressed()
    result = np.intersect1d(ar1, ar2)
    return MaskedArray(result, mask=False)

def union1d(ar1, ar2):
    """Return the union of two masked arrays."""
    import numpy as np
    if isinstance(ar1, MaskedArray): ar1 = ar1.compressed()
    if isinstance(ar2, MaskedArray): ar2 = ar2.compressed()
    result = np.union1d(ar1, ar2)
    return MaskedArray(result, mask=False)

def setdiff1d(ar1, ar2, assume_unique=False):
    """Return the set difference of two masked arrays."""
    import numpy as np
    if isinstance(ar1, MaskedArray): ar1 = ar1.compressed()
    if isinstance(ar2, MaskedArray): ar2 = ar2.compressed()
    result = np.setdiff1d(ar1, ar2)
    return MaskedArray(result, mask=False)

def setxor1d(ar1, ar2, assume_unique=False):
    """Return the set exclusive-or of two masked arrays."""
    import numpy as np
    if isinstance(ar1, MaskedArray): ar1 = ar1.compressed()
    if isinstance(ar2, MaskedArray): ar2 = ar2.compressed()
    result = np.setxor1d(ar1, ar2)
    return MaskedArray(result, mask=False)

def in1d(ar1, ar2, assume_unique=False, invert=False):
    """Test whether each element of ar1 is in ar2, masked-aware."""
    import numpy as np
    a1_data = ar1.compressed() if isinstance(ar1, MaskedArray) else np.asarray(ar1)
    a2_data = ar2.compressed() if isinstance(ar2, MaskedArray) else np.asarray(ar2)
    result = np.in1d(a1_data, a2_data, assume_unique=assume_unique, invert=invert)
    return result

def isin(element, test_elements, assume_unique=False, invert=False):
    """Test whether elements are in test_elements, masked-aware."""
    import numpy as np
    if isinstance(element, MaskedArray):
        data = np.asarray(element._data) if hasattr(element, '_data') else np.asarray(element.data)
        m = element._mask if hasattr(element, '_mask') else getattr(element, 'mask', nomask)
        te = test_elements.compressed() if isinstance(test_elements, MaskedArray) else np.asarray(test_elements)
        result = np.isin(data, te, assume_unique=assume_unique, invert=invert)
        if m is not nomask and m is not False:
            return MaskedArray(result, mask=m)
        return MaskedArray(result, mask=False)
    elem = np.asarray(element)
    te = test_elements.compressed() if isinstance(test_elements, MaskedArray) else np.asarray(test_elements)
    return np.isin(elem, te, assume_unique=assume_unique, invert=invert)


# ---------------------------------------------------------------------------
# Statistics (corrcoef, cov, polyfit, vander)
# ---------------------------------------------------------------------------

def corrcoef(x, y=None, rowvar=True, allow_masked=True):
    import numpy as np
    x_data = x.compressed() if isinstance(x, MaskedArray) else np.asarray(x)
    if y is not None:
        y_data = y.compressed() if isinstance(y, MaskedArray) else np.asarray(y)
        return np.corrcoef(x_data, y_data, rowvar=rowvar)
    return np.corrcoef(x_data, rowvar=rowvar)

def cov(x, y=None, rowvar=True, bias=False, allow_masked=True, ddof=None):
    import numpy as np
    x_data = x.compressed() if isinstance(x, MaskedArray) else np.asarray(x)
    kw = {}
    if ddof is not None:
        kw['ddof'] = ddof
    else:
        kw['bias'] = bias
    if y is not None:
        y_data = y.compressed() if isinstance(y, MaskedArray) else np.asarray(y)
        return np.cov(x_data, y_data, rowvar=rowvar, **kw)
    return np.cov(x_data, rowvar=rowvar, **kw)

def polyfit(x, y, deg, rcond=None, full=False, w=None, cov=False):
    import numpy as np
    x_data = x.compressed() if isinstance(x, MaskedArray) else np.asarray(x)
    y_data = y.compressed() if isinstance(y, MaskedArray) else np.asarray(y)
    return np.polyfit(x_data, y_data, deg, rcond=rcond, full=full, w=w, cov=cov)

def vander(x, n=None):
    import numpy as np
    x_data = x.data if isinstance(x, MaskedArray) else np.asarray(x)
    return np.vander(x_data, N=n)


# ---------------------------------------------------------------------------
# Misc functions from extras
# ---------------------------------------------------------------------------

def apply_along_axis(func1d, axis, arr, *args, **kwargs):
    import numpy as np
    data = arr.data if isinstance(arr, MaskedArray) else np.asarray(arr)
    return np.apply_along_axis(func1d, axis, data, *args, **kwargs)

def apply_over_axes(func, a, axes):
    import numpy as np
    data = a.data if isinstance(a, MaskedArray) else np.asarray(a)
    return np.apply_over_axes(func, data, axes)

def atleast_1d(*arys):
    import numpy as np
    results = []
    for a in arys:
        if isinstance(a, MaskedArray):
            d = np.atleast_1d(a.data)
            m = np.atleast_1d(a.mask)
            results.append(MaskedArray(d, mask=m, fill_value=a._fill_value))
        else:
            results.append(np.atleast_1d(a))
    if len(results) == 1:
        return results[0]
    return results

def atleast_2d(*arys):
    import numpy as np
    results = []
    for a in arys:
        if isinstance(a, MaskedArray):
            d = np.atleast_2d(a.data)
            m = np.atleast_2d(a.mask)
            results.append(MaskedArray(d, mask=m, fill_value=a._fill_value))
        else:
            results.append(np.atleast_2d(a))
    if len(results) == 1:
        return results[0]
    return results

def atleast_3d(*arys):
    import numpy as np
    results = []
    for a in arys:
        if isinstance(a, MaskedArray):
            d = np.atleast_3d(a.data)
            m = np.atleast_3d(a.mask)
            results.append(MaskedArray(d, mask=m, fill_value=a._fill_value))
        else:
            results.append(np.atleast_3d(a))
    if len(results) == 1:
        return results[0]
    return results

def clump_masked(a):
    """Return list of slices for masked runs."""
    if not isinstance(a, MaskedArray):
        return []
    mask = a.mask.flatten().tolist()
    slices = []
    start = None
    for i, m in enumerate(mask):
        if m and start is None:
            start = i
        elif not m and start is not None:
            slices.append(slice(start, i))
            start = None
    if start is not None:
        slices.append(slice(start, len(mask)))
    return slices

def clump_unmasked(a):
    """Return list of slices for unmasked runs."""
    if not isinstance(a, MaskedArray):
        return [slice(0, a.size if hasattr(a, 'size') else len(a))]
    mask = a.mask.flatten().tolist()
    slices = []
    start = None
    for i, m in enumerate(mask):
        if not m and start is None:
            start = i
        elif m and start is not None:
            slices.append(slice(start, i))
            start = None
    if start is not None:
        slices.append(slice(start, len(mask)))
    return slices

def flatnotmasked_edges(a):
    """Find first and last unmasked value indices."""
    if not isinstance(a, MaskedArray):
        import numpy as np
        s = np.asarray(a).size
        return [0, s - 1] if s > 0 else None
    mask = a.mask.flatten().tolist()
    first = None
    last = None
    for i, m in enumerate(mask):
        if not m:
            if first is None:
                first = i
            last = i
    if first is None:
        return None
    return [first, last]

def flatnotmasked_contiguous(a):
    """Find contiguous unmasked data in a flat array."""
    return clump_unmasked(a)

def notmasked_edges(a, axis=None):
    """Find first and last unmasked values along axis."""
    return flatnotmasked_edges(a)

def notmasked_contiguous(a, axis=None):
    """Find contiguous unmasked data."""
    return flatnotmasked_contiguous(a)

def compress_rows(a):
    """Suppress rows with masked values in a 2D array."""
    import numpy as np
    if not isinstance(a, MaskedArray) or a.ndim != 2:
        return a
    row_has_mask = np.any(a.mask, axis=1)
    rows_to_keep = np.logical_not(row_has_mask)
    return MaskedArray(np.compress(rows_to_keep, a.data, axis=0))

def compress_cols(a):
    """Suppress columns with masked values in a 2D array."""
    import numpy as np
    if not isinstance(a, MaskedArray) or a.ndim != 2:
        return a
    col_has_mask = np.any(a.mask, axis=0)
    cols_to_keep = np.logical_not(col_has_mask)
    return MaskedArray(np.compress(cols_to_keep, a.data, axis=1))

def compress_rowcols(x, axis=None):
    """Suppress rows and/or columns with masked values."""
    if axis is None:
        x = compress_rows(x)
        return compress_cols(x)
    if axis == 0:
        return compress_rows(x)
    return compress_cols(x)

def compress_nd(x, axis=None):
    """Suppress slices along axis with masked values."""
    if axis is None:
        return compress_rowcols(x)
    return compress_rowcols(x, axis=axis)


def _covhelper(x, y=None, rowvar=True, allow_masked=True):
    """Helper function for covariance/correlation of masked arrays."""
    import numpy as np
    x = array(x, copy=False)
    if y is not None:
        y = array(y, copy=False)
    if not rowvar and x.ndim > 1:
        x = x.T
        if y is not None:
            y = y.T
    if y is not None:
        # Combine x and y
        xdata = x.data if isinstance(x, MaskedArray) else np.asarray(x)
        ydata = y.data if isinstance(y, MaskedArray) else np.asarray(y)
        if xdata.ndim == 1:
            xdata = xdata.reshape(1, -1)
        if ydata.ndim == 1:
            ydata = ydata.reshape(1, -1)
        combined_data = np.concatenate([xdata, ydata], axis=0)
        xmask = x.mask if isinstance(x, MaskedArray) else np.zeros(xdata.shape, dtype='bool')
        ymask = y.mask if isinstance(y, MaskedArray) else np.zeros(ydata.shape, dtype='bool')
        if xmask.ndim == 0:
            xmask = np.full(xdata.shape, bool(xmask), dtype='bool')
        if ymask.ndim == 0:
            ymask = np.full(ydata.shape, bool(ymask), dtype='bool')
        combined_mask = np.concatenate([xmask, ymask], axis=0)
        x = MaskedArray(combined_data, mask=combined_mask)
    else:
        if not isinstance(x, MaskedArray):
            x = MaskedArray(np.asarray(x))
        if x.ndim == 1:
            xdata = x.data.reshape(1, -1)
            xmask = x.mask.reshape(1, -1) if x.mask.ndim > 0 else np.full((1, xdata.shape[1]), bool(x.mask), dtype='bool')
            x = MaskedArray(xdata, mask=xmask)
    return x, x.shape[0]


# ---------------------------------------------------------------------------
# Submodule stubs
# ---------------------------------------------------------------------------

class _TestUtils:
    """Stub for numpy.ma.testutils — delegates to numpy.testing."""

    @staticmethod
    def assert_(val, msg=""):
        if not val:
            raise AssertionError(msg or "assertion failed")

    @staticmethod
    def assert_equal_records(a, b):
        import numpy as np
        np.testing.assert_array_equal(np.asarray(a), np.asarray(b))

    @staticmethod
    def assert_mask_equal(m1, m2):
        import numpy as np
        if m1 is nomask and m2 is nomask:
            return
        np.testing.assert_array_equal(np.asarray(m1), np.asarray(m2))

    @staticmethod
    def assert_not_equal(a, b):
        import numpy as np
        if np.all(np.asarray(a) == np.asarray(b)):
            raise AssertionError("{} == {}".format(a, b))

    @staticmethod
    def fail_if_equal(actual, desired, err_msg=""):
        """Raise AssertionError if two arrays are equal."""
        import numpy as np
        if np.all(np.asarray(actual) == np.asarray(desired)):
            raise AssertionError(err_msg or "{} == {}".format(actual, desired))

    @staticmethod
    def assert_almost_equal(actual, desired, decimal=7, err_msg="", verbose=True):
        import numpy as np
        np.testing.assert_almost_equal(actual, desired, decimal=decimal,
                                       err_msg=err_msg, verbose=verbose)

    @staticmethod
    def assert_array_almost_equal(actual, desired, decimal=6, err_msg="", verbose=True):
        import numpy as np
        np.testing.assert_array_almost_equal(actual, desired, decimal=decimal,
                                             err_msg=err_msg, verbose=verbose)

    def __getattr__(self, name):
        import numpy as np
        if hasattr(np.testing, name):
            return getattr(np.testing, name)
        raise AttributeError("module 'numpy.ma.testutils' has no attribute '{}'".format(name))

_sys.modules['numpy.ma.testutils'] = _TestUtils()


class _CoreModule:
    """Stub for numpy.ma.core — delegates attribute access back to numpy.ma."""
    def __getattr__(self, name):
        import numpy.ma as ma
        if hasattr(ma, name):
            return getattr(ma, name)
        raise AttributeError("module 'numpy.ma.core' has no attribute '{}'".format(name))

_sys.modules['numpy.ma.core'] = _CoreModule()


class _ExtrasModule:
    """Stub for numpy.ma.extras — delegates attribute access back to numpy.ma."""
    def __getattr__(self, name):
        import numpy.ma as ma
        if hasattr(ma, name):
            return getattr(ma, name)
        raise AttributeError("module 'numpy.ma.extras' has no attribute '{}'".format(name))

_sys.modules['numpy.ma.extras'] = _ExtrasModule()
