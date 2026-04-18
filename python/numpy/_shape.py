"""Shape and dimension manipulation: reshape, transpose, flip, broadcast, etc."""
import math as _math
import _numpy_native as _native
from _numpy_native import ndarray
from ._helpers import (
    AxisError, _ObjectArray, _copy_into,
    _builtin_range, _builtin_min, _builtin_max,
)
from ._core_types import dtype, _ScalarType, _normalize_dtype
from ._creation import array, asarray, zeros, ones, empty, arange, concatenate, linspace

__all__ = [
    'reshape', '_transpose_with_axes', 'transpose', 'flatten', 'ravel',
    'squeeze', 'expand_dims',
    'atleast_1d', 'atleast_2d', 'atleast_3d',
    'moveaxis', 'rollaxis', '_swapaxes_structured', 'swapaxes',
    'flip', 'flipud', 'fliplr', 'rot90',
    'ndim', 'size',
    'broadcast', '_BroadcastIter', 'broadcast_to', 'broadcast_arrays',
    'broadcast_shapes',
]


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
    # Delegate copy= and order= to the instance method (which handles memory tagging)
    kw = {}
    if order is not None:
        kw['order'] = order
    if copy is not None:
        kw['copy'] = copy
    return a.reshape(newshape, **kw)


def _transpose_with_axes(a, axes):
    """Transpose ndarray with an arbitrary axis permutation (pure Python).

    Parameters
    ----------
    a : ndarray
    axes : list/tuple of int - the desired permutation of axes.

    Returns an ndarray with axes reordered according to *axes*.
    """
    if type(a).__name__ == 'StructuredArray' and hasattr(a, 'shape'):
        ndim_a = len(a.shape)
        if axes is None:
            axes = list(range(ndim_a - 1, -1, -1))
        axes = list(axes)
        if axes == list(range(ndim_a)):
            return a
        # Use _swapaxes_structured iteratively for general permutation
        result = a
        # Decompose permutation into sequence of swaps (selection sort)
        perm = axes[:]
        for i in range(ndim_a):
            if perm[i] != i:
                j = perm.index(i)
                result = _swapaxes_structured(result, i, j)
                perm[i], perm[j] = perm[j], perm[i]
        return result
    if not isinstance(a, ndarray):
        a = asarray(a)
    shape = a.shape
    ndim_a = len(shape)
    if axes is None:
        return a.T
    axes = list(axes)
    if len(axes) != ndim_a:
        raise ValueError("axes don't match array")
    norm_axes = []
    for ax in axes:
        ax_n = int(ax)
        if ax_n < 0:
            ax_n += ndim_a
        if ax_n < 0 or ax_n >= ndim_a:
            raise ValueError("axes don't match array")
        norm_axes.append(ax_n)
    if len(set(norm_axes)) != ndim_a:
        raise ValueError("axes don't match array")
    # Fast-paths
    if norm_axes == list(range(ndim_a)):
        return a.copy() if hasattr(a, 'copy') else array(a.tolist())
    if ndim_a == 2 and norm_axes == [1, 0] and type(a).__name__ != '_ObjectArray':
        return a.T
    return a.transpose(tuple(norm_axes))


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
    from ._helpers import _ObjectArray
    if isinstance(a, _ObjectArray):
        new_shape = list(a.shape)
        if axis < 0:
            axis = len(new_shape) + 1 + axis
        new_shape.insert(axis, 1)
        return a.reshape(tuple(new_shape))
    return a


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
    from numpy.ma import MaskedArray
    results = []
    for a in arys:
        if isinstance(a, MaskedArray):
            # Preserve MaskedArray type
            if a.ndim == 0:
                import numpy as _np
                d = a.data.reshape([1, 1]) if isinstance(a.data, ndarray) else _np.array(a.data).reshape([1, 1])
                m = _np.ones([1, 1], dtype='bool') if isinstance(a.mask, bool) and a.mask else _np.zeros([1, 1], dtype='bool')
                a = MaskedArray(d, mask=m, fill_value=a._fill_value)
            elif a.ndim == 1:
                import numpy as _np
                d = a.data.reshape([1, a.data.size]) if isinstance(a.data, ndarray) else _np.array(a.data).reshape([1, len(a)])
                m = a.mask.reshape([1, a.mask.size]) if isinstance(a.mask, ndarray) else _np.broadcast_to(_np.asarray(a.mask, dtype='bool'), [1, len(a)]).copy()
                a = MaskedArray(d, mask=m, fill_value=a._fill_value)
        else:
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


def _swapaxes_structured(a, axis1, axis2):
    """Swap two axes of an N-D StructuredArray."""
    import itertools
    from ._creation import _create_structured_array
    shape = list(a.shape)
    ndim = len(shape)
    # New shape after swap
    new_shape = shape[:]
    new_shape[axis1], new_shape[axis2] = new_shape[axis2], new_shape[axis1]
    field_names = a.dtype.names if hasattr(a.dtype, 'names') else None
    flat_records = []
    for out_idx in itertools.product(*[range(s) for s in new_shape]):
        # Map output index to source index
        src_idx = list(out_idx)
        src_idx[axis1], src_idx[axis2] = out_idx[axis2], out_idx[axis1]
        val = a[tuple(src_idx)]
        if field_names and type(val).__name__ == 'void':
            flat_records.append(tuple(val[n] for n in field_names))
        elif isinstance(val, tuple):
            flat_records.append(val)
        else:
            v = val.tolist() if hasattr(val, 'tolist') else val
            flat_records.append(v if isinstance(v, tuple) else (v,))
    sa_flat = _create_structured_array(flat_records, a.dtype)
    return sa_flat.reshape(new_shape)


def swapaxes(a, axis1, axis2):
    """Interchange two axes of an array."""
    if type(a).__name__ == 'StructuredArray' and hasattr(a, 'shape'):
        ndim_a = len(a.shape)
        if axis1 < 0:
            axis1 += ndim_a
        if axis2 < 0:
            axis2 += ndim_a
        if axis1 == axis2:
            return a
        return _swapaxes_structured(a, axis1, axis2)
    a = asarray(a) if not isinstance(a, ndarray) else a
    ndim_a = a.ndim
    if axis1 < 0:
        axis1 += ndim_a
    if axis2 < 0:
        axis2 += ndim_a
    if axis1 == axis2:
        return a
    return a.swapaxes(axis1, axis2)


def flip(a, axis=None):
    """Reverse the order of elements along the given axis."""
    from ._helpers import AxisError as _AxisError
    a = asarray(a)
    if axis is None:
        # Flip along all axes
        axes = tuple(_builtin_range(a.ndim))
    elif isinstance(axis, (tuple, list)):
        axes = tuple(axis)
    else:
        axes = (int(axis),)
    # Validate axes
    for ax in axes:
        if ax < -a.ndim or ax >= a.ndim:
            raise _AxisError(
                "axis {} is out of bounds for array of dimension {}".format(ax, a.ndim))
    # Normalize negative axes
    axes = tuple(ax % a.ndim if ax < 0 else ax for ax in axes)
    # Check for duplicate normalized axes in AxisError context
    for ax in axes:
        if ax >= a.ndim:
            raise _AxisError(
                "axis {} is out of bounds for array of dimension {}".format(ax, a.ndim))
    # Apply flips
    result = a
    for ax in axes:
        result = _native.flip(result, ax)
    return result


def flipud(a):
    """Flip array upside down (reverse along axis 0)."""
    if not isinstance(a, ndarray):
        a = asarray(a)
    return _native.flipud(a)


def fliplr(a):
    """Flip array left-right (reverse along axis 1)."""
    if not isinstance(a, ndarray):
        a = asarray(a)
    if a.ndim < 2:
        raise ValueError("Input must be >= 2-d.")
    return _native.fliplr(a)


def rot90(a, k=1, axes=(0, 1)):
    """Rotate an array by 90 degrees in the plane specified by axes."""
    axes = tuple(axes)
    if len(axes) != 2:
        raise ValueError("len(axes) must be 2.")
    a = asarray(a)
    if (axes[0] >= a.ndim or axes[0] < -a.ndim
            or axes[1] >= a.ndim or axes[1] < -a.ndim):
        raise ValueError(
            "Axes={} out of range for array of ndim={}.".format(axes, a.ndim))
    # Normalize negative axes
    ax0 = axes[0] % a.ndim
    ax1 = axes[1] % a.ndim
    if ax0 == ax1:
        raise ValueError("Axes must be different.")
    k = k % 4
    if k == 0:
        return a[:]
    if k == 2:
        return flip(flip(a, ax0), ax1)
    # Build transposed axes list: swap ax0 and ax1
    axes_list = list(_builtin_range(a.ndim))
    axes_list[ax0], axes_list[ax1] = axes_list[ax1], axes_list[ax0]
    if k == 1:
        return transpose(flip(a, ax1), axes_list)
    else:  # k == 3
        return flip(transpose(a, axes_list), ax1)


def ndim(a):
    if isinstance(a, ndarray):
        return a.ndim
    try:
        return asarray(a).ndim
    except Exception:
        return 0

def size(a, axis=None):
    if not isinstance(a, ndarray):
        a = asarray(a)
    if axis is None:
        return a.size
    if isinstance(axis, tuple):
        if len(axis) == 0:
            return 1
        result = 1
        for ax in axis:
            result *= a.shape[ax]
        return result
    return a.shape[axis]


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
        self._broadcasted = [broadcast_to(a, self.shape) for a in arrays]
        self.iters = tuple(_BroadcastIter(bc, a) for bc, a in zip(self._broadcasted, arrays))
        self.reset()

    def __iter__(self):
        return self

    def __next__(self):
        if self.index >= self.size:
            raise StopIteration
        result = tuple(float(next(it)) for it in self._flat_iters)
        self.index += 1
        return result

    def reset(self):
        self.index = 0
        self._flat_iters = [bc.flat.__iter__() for bc in self._broadcasted]


def broadcast_shapes(*shapes):
    """Compute the broadcast result shape from multiple shapes."""
    if not shapes:
        return ()
    # Normalize: integers are treated as 1-d shapes
    shapes = [s if hasattr(s, '__len__') else (s,) for s in shapes]
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


def broadcast_to(arr, shape):
    """Broadcast an array to a new shape using reshape + tile."""
    from ._iteration import tile
    arr = asarray(arr)
    arr_shape = arr.shape
    # Normalize shape: int -> (int,)
    if isinstance(shape, (int,)):
        shape = (shape,)
    else:
        shape = tuple(shape)
    # Validate: no negative dimensions
    for s in shape:
        if s < 0:
            raise ValueError(f"all elements of broadcast shape must be non-negative")
    if arr_shape == shape:
        return arr
    ndim = len(shape)
    arr_ndim = len(arr_shape)
    # Cannot reduce dimensionality
    if arr_ndim > ndim:
        raise ValueError(
            f"input operand has more dimensions than allowed by the axis remapping"
        )
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
            raise ValueError(
                f"operands could not be broadcast together with remapped shapes "
                f"[original->remapped]: {arr.shape}->... and requested shape {shape}"
            )
    # Guard against unreasonably large output (raise MemoryError like NumPy)
    _MAX_ELEMENTS = 2 ** 50  # ~1 quadrillion elements
    _total = 1
    for s in shape:
        _total *= s
        if _total > _MAX_ELEMENTS:
            raise MemoryError(
                "array is too large; required memory exceeds available resources"
            )
    return tile(arr, reps)


def broadcast_arrays(*args):
    """Broadcast any number of arrays against each other."""
    arrays = [asarray(a) for a in args]
    if len(arrays) == 0:
        return []
    shape = broadcast_shapes(*(a.shape for a in arrays))
    return [broadcast_to(a, shape) for a in arrays]
