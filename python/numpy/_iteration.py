"""Iteration helpers: apply_along_axis, vectorize, repeat, tile, roll, resize, trim_zeros."""
import _numpy_native as _native
from _numpy_native import ndarray
from ._helpers import (
    AxisError, _ObjectArray,
    _builtin_range, _builtin_min, _builtin_max,
)
from ._creation import array, asarray, zeros, ones, empty, arange, concatenate

__all__ = [
    'apply_along_axis', 'apply_over_axes', 'vectorize',
    'repeat', 'tile', '_native_resize', '_resize_structured', 'resize',
    'roll',
    'trim_zeros', '_trim_zeros_is_nonzero', '_trim_zeros_slice_nonzero',
]


def apply_along_axis(func1d, axis, arr, *args, **kwargs):
    """Apply a function to 1-D slices of an array along the given axis."""
    from ._join import stack
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
            if isinstance(elem, (tuple, list)) and len(elem) == 2:
                # Complex element (re, im) - store as [re, im] pair
                slice_vals.append([float(elem[0]), float(elem[1])])
            elif isinstance(elem, complex):
                slice_vals.append([elem.real, elem.imag])
            else:
                slice_vals.append(float(elem))
        # Check if we have complex values (stored as [re, im] pairs)
        if slice_vals and isinstance(slice_vals[0], list):
            slice_arr = array(slice_vals)  # (n, 2) ndarray
        else:
            slice_arr = array(slice_vals)
        result = func1d(slice_arr, *args, **kwargs)
        results.append(result)
    # Reshape results back to out_shape
    # If func1d returns scalar, result shape is out_shape
    first_res = results[0]
    if isinstance(first_res, ndarray):
        # func returns array
        res_shape = first_res.shape
        if res_shape == (shape[axis],):
            # Result has same length as input slice - insert axis back
            # Final shape: out_shape[:axis] + (shape[axis],) + out_shape[axis:]
            # We need to reconstruct the full array
            final_shape = list(out_shape[:axis]) + [shape[axis]] + list(out_shape[axis:])
            # Build flat list
            flat_vals = []
            for r in results:
                flat_vals.extend(r.flatten().tolist())
            result_arr = array(flat_vals).reshape(final_shape)
            # Need to move axis back: currently results are indexed by outer dims first
            # then result values. We need to transpose to put axis in correct position.
            return result_arr
        else:
            # Result has different shape - try basic reshape
            try:
                result_arr = asarray(results)
                return result_arr.reshape(out_shape)
            except ValueError:
                # If reshape fails, return as stack
                return stack(results).reshape(out_shape + res_shape)
    else:
        try:
            return array([float(r) for r in results]).reshape(out_shape)
        except (TypeError, ValueError):
            return array(results).reshape(out_shape)


class vectorize:
    """Generalized function class.

    Takes a nested sequence of objects or numpy arrays as inputs and returns
    a single numpy array or a tuple of numpy arrays by applying the function
    element-by-element.
    """
    def __init__(self, pyfunc=None, otypes=None, doc=None, excluded=None, cache=False, signature=None):
        if pyfunc is None:
            # Called as decorator factory: vectorize(otypes=..., signature=...)
            # Return a callable that wraps the function
            self._deferred_args = dict(otypes=otypes, doc=doc, excluded=excluded,
                                       cache=cache, signature=signature)
            self.pyfunc = None
            return
        if not callable(pyfunc):
            raise TypeError("pyfunc must be a callable, not {!r}".format(type(pyfunc).__name__))
        self.pyfunc = pyfunc
        self.otypes = otypes
        self.signature = signature
        self.excluded = excluded if excluded is not None else set()
        if doc is not None:
            self.__doc__ = doc
        elif hasattr(pyfunc, '__doc__') and pyfunc.__doc__:
            self.__doc__ = pyfunc.__doc__
        if hasattr(pyfunc, '__name__'):
            self.__name__ = pyfunc.__name__

    @staticmethod
    def _otype_to_dtype(otypes):
        """Convert otypes specification to dtype string."""
        if isinstance(otypes, (list, tuple)):
            _ot = otypes[0]
        elif isinstance(otypes, str):
            _ot = otypes[0]
        else:
            _ot = otypes
        return str(asarray([], dtype=_ot).dtype)

    def _call_with_signature(self, args, kwargs):
        """Handle vectorize call when a gufunc signature is specified."""
        import itertools
        from .lib._function_base_impl import _parse_gufunc_signature
        from ._shape import broadcast_shapes, broadcast_to
        in_specs, out_specs = _parse_gufunc_signature(self.signature)
        n_in = len(in_specs)

        if len(args) != n_in:
            raise TypeError(
                "wrong number of positional arguments: expected {}, got {}".format(
                    n_in, len(args)))

        arr_args = [asarray(a) for a in args]

        # Collect input dimension sizes and loop shapes
        dim_sizes = {}
        loop_shapes = []
        for i, (arr, spec) in enumerate(zip(arr_args, in_specs)):
            n_core = len(spec)
            if arr.ndim < n_core:
                raise ValueError(
                    "input {} does not have enough dimensions".format(i))
            loop_shapes.append(arr.shape[:arr.ndim - n_core])
            core_shape = arr.shape[arr.ndim - n_core:]
            for dim_name, size in zip(spec, core_shape):
                if dim_name in dim_sizes:
                    if dim_sizes[dim_name] != size:
                        raise ValueError(
                            "inconsistent size for core dimension {!r}: "
                            "have {} and {}".format(dim_name, dim_sizes[dim_name], size))
                else:
                    dim_sizes[dim_name] = size

        # Broadcast loop shapes
        bc_loop_shape = tuple(broadcast_shapes(*loop_shapes)) if loop_shapes else ()

        # Broadcast each input to bc_loop_shape + core_shape
        bc_arr_args = []
        for arr, spec in zip(arr_args, in_specs):
            n_core = len(spec)
            core_shape = arr.shape[arr.ndim - n_core:] if n_core > 0 else ()
            target_shape = bc_loop_shape + core_shape
            if arr.shape != target_shape:
                arr = broadcast_to(arr, target_shape)
            bc_arr_args.append(arr)

        n_loop = 1
        for s in bc_loop_shape:
            n_loop *= s

        otypes = self.otypes

        if n_loop == 0:
            # No iterations - need otypes or known dims to build output
            if otypes is None:
                raise ValueError("otypes must be specified for empty inputs")
            dt = self._otype_to_dtype(otypes)
            outputs = []
            for ospec in out_specs:
                # Check all output dims are known from inputs
                for dim_name in ospec:
                    if dim_name not in dim_sizes:
                        raise ValueError(
                            "new output dimensions not allowed with empty inputs "
                            "when output size cannot be determined")
                out_core_shape = tuple(dim_sizes[d] for d in ospec)
                out_shape = bc_loop_shape + out_core_shape
                outputs.append(zeros(out_shape, dtype=dt))
            return outputs[0] if len(outputs) == 1 else tuple(outputs)

        # Build loop indices
        if bc_loop_shape:
            loop_indices = list(itertools.product(*[range(s) for s in bc_loop_shape]))
        else:
            loop_indices = [()]

        n_out = len(out_specs)
        all_results = []  # list of tuples of arrays (one per output)

        for idx in loop_indices:
            call_args = [arr[idx] if idx else arr[()] for arr in bc_arr_args]
            res = self.pyfunc(*call_args, **kwargs)

            # Normalise to tuple of outputs
            if not isinstance(res, tuple):
                res = (res,)
            if len(res) != n_out:
                raise ValueError(
                    "wrong number of outputs: expected {}, got {}".format(
                        n_out, len(res)))
            res = tuple(asarray(r) for r in res)

            # Validate / register output core dimension sizes
            for r, ospec in zip(res, out_specs):
                for dim_name, size in zip(ospec, r.shape):
                    if dim_name in dim_sizes:
                        if dim_sizes[dim_name] != size:
                            raise ValueError(
                                "inconsistent size for core dimension {!r}: "
                                "have {} and {}".format(
                                    dim_name, dim_sizes[dim_name], size))
                    else:
                        dim_sizes[dim_name] = size

            all_results.append(res)

        # Assemble outputs
        outputs = []
        for k, ospec in enumerate(out_specs):
            out_core_shape = tuple(dim_sizes.get(d, 0) for d in ospec)
            out_shape = bc_loop_shape + out_core_shape

            if not bc_loop_shape:
                # Single call: result is already the right shape, no stacking needed
                result = all_results[0][k]
                if otypes is not None:
                    dt = self._otype_to_dtype(otypes)
                    if str(result.dtype) != dt:
                        result = result.astype(dt)
                if out_shape and result.shape != out_shape:
                    result = result.reshape(out_shape)
            else:
                res_k = [r[k] for r in all_results]
                if otypes is not None:
                    dt = self._otype_to_dtype(otypes)
                    result = array(res_k, dtype=dt).reshape(out_shape)
                else:
                    result = array(res_k).reshape(out_shape)
            outputs.append(result)

        return outputs[0] if len(outputs) == 1 else tuple(outputs)

    def __call__(self, *args, **kwargs):
        from ._shape import broadcast_arrays
        # If used as decorator factory (pyfunc=None), wrap the passed function
        if self.pyfunc is None:
            if len(args) == 1 and callable(args[0]):
                func = args[0]
                kwargs2 = self._deferred_args
                result = vectorize(func, **kwargs2)
                return result
            raise TypeError("vectorize requires a callable as first argument")

        # Dispatch to signature-aware implementation if needed
        if self.signature is not None:
            return self._call_with_signature(args, kwargs)

        # Separate excluded args (passed as-is) from vectorized args
        excluded = self.excluded
        vec_indices = [i for i in range(len(args)) if i not in excluded]
        vec_args_orig = [args[i] for i in vec_indices]

        # Convert vectorized args to arrays
        arr_args = [asarray(a) for a in vec_args_orig]
        if len(arr_args) == 0:
            return array([])
        # Broadcast only vectorized args to common shape
        broadcasted = broadcast_arrays(*arr_args)
        shape = broadcasted[0].shape
        n = broadcasted[0].size
        otypes = self.otypes
        if n == 0:
            # Empty input: return empty array with correct shape
            if otypes is None:
                raise ValueError(
                    "otypes must be specified for empty inputs")
            dt = str(asarray([], dtype=otypes[0] if isinstance(otypes, (list, tuple, str)) else otypes).dtype)
            return zeros(shape, dtype=dt)
        results = []
        flat_args = [b.flatten() for b in broadcasted]
        for i in range(n):
            # Reconstruct full argument list: vectorized elements + excluded originals
            call_args = list(args)
            for vi, b in enumerate(flat_args):
                call_args[vec_indices[vi]] = b[i]
            results.append(self.pyfunc(*call_args, **kwargs))
        # Check if result is tuple (multi-output)
        if isinstance(results[0], tuple):
            nout = len(results[0])
            out = []
            for k in range(nout):
                vals = [r[k] for r in results]
                out.append(array(vals).reshape(shape))
            return tuple(out)
        if otypes is not None:
            # Coerce to specified otype
            _ot = otypes[0] if isinstance(otypes, (list, tuple)) else otypes[0] if isinstance(otypes, str) else otypes
            if _ot is object:
                from ._helpers import _ObjectArray
                obj_arr = _ObjectArray(shape)
                for i, v in enumerate(results):
                    obj_arr._data[i] = v
                return obj_arr
            result = array(results).astype(str(asarray([], dtype=_ot).dtype))
        else:
            result = array(results)
            # Preserve input dtype to avoid silent promotion (e.g. int32 -> int64)
            # Use the first array argument's dtype as the target
            if len(arr_args) > 0:
                _in_dt = str(arr_args[0].dtype)
                _res_dt = str(result.dtype)
                # Only coerce if result dtype is a promotion of the input dtype
                # (e.g. int16/int32 promoted to int64, float16/float32 to float64)
                _dt_map = {
                    'int16': 'int64', 'int32': 'int64',
                    'uint16': 'int64', 'uint32': 'int64',
                    'float16': 'float64', 'float32': 'float64',
                }
                if _dt_map.get(_in_dt) == _res_dt:
                    result = result.astype(_in_dt)
        if shape != result.shape:
            result = result.reshape(shape)
        return result


def repeat(a, repeats, axis=None):
    if not isinstance(a, ndarray):
        a = asarray(a)
    # If repeats is an array or list, implement manually along axis
    if isinstance(repeats, (ndarray, list, tuple)):
        reps = [int(x) for x in (repeats.flatten().tolist() if isinstance(repeats, ndarray) else repeats)]
        if axis is None:
            flat = a.flatten().tolist()
            if len(reps) == 1:
                reps = reps * len(flat)
            out = []
            for val, r in zip(flat, reps):
                out.extend([val] * r)
            return asarray(out).astype(str(a.dtype))
        else:
            import numpy as _np
            slices = []
            n = a.shape[axis]
            if len(reps) == 1:
                reps = reps * n
            for i in range(n):
                idx = [slice(None)] * a.ndim
                idx[axis] = i
                sl = a[tuple(idx)]
                for _ in range(reps[i]):
                    slices.append(sl)
            return _np.stack(slices, axis=axis)
    return _native.repeat(a, repeats, axis)


def tile(a, reps):
    from numpy.ma import MaskedArray
    from numpy._helpers import _ObjectArray
    if isinstance(a, MaskedArray):
        # Tile both data and mask
        data_tiled = _native.tile(a.data if isinstance(a.data, ndarray) else asarray(a.data), reps)
        mask_tiled = _native.tile(asarray(a.mask, dtype="bool") if not isinstance(a.mask, ndarray) else a.mask, reps)
        return MaskedArray(data_tiled, mask=mask_tiled, fill_value=a._fill_value)
    if isinstance(a, _ObjectArray):
        # Tile object array by repeating data
        import itertools as _it
        reps_list = [reps] if isinstance(reps, int) else list(reps)
        # Pad reps to match ndim
        while len(reps_list) < a.ndim:
            reps_list = [1] + reps_list
        # Compute new shape
        shape = list(a.shape)
        while len(shape) < len(reps_list):
            shape = [1] + shape
        new_shape = tuple(s * r for s, r in zip(shape, reps_list))
        # Build tiled data
        out_data = []
        for multi_idx in _it.product(*[range(ns) for ns in new_shape]):
            src_idx = tuple(i % s for i, s in zip(multi_idx, shape))
            flat = 0
            mult = 1
            for d in range(len(shape) - 1, -1, -1):
                flat += src_idx[d] * mult
                mult *= shape[d]
            out_data.append(a._data[flat])
        return _ObjectArray(out_data, a._dtype, shape=new_shape)
    if not isinstance(a, ndarray):
        a = asarray(a)
    return _native.tile(a, reps)


def _native_resize(col, total):
    """Tile a 1D ndarray to length total."""
    n = len(col)
    flat = col.flatten().tolist()
    result_vals = [flat[i % n] for i in _builtin_range(total)]
    return asarray(result_vals).astype(str(col.dtype))


def _resize_structured(a, new_shape):
    """Resize a StructuredArray by tiling its fields to fill new_shape."""
    import json as _json
    import _numpy_native as _native_mod
    from numpy import StructuredArray
    from _numpy_native import ndarray

    if isinstance(new_shape, int):
        new_shape = (new_shape,)
    else:
        new_shape = tuple(new_shape)

    total = 1
    for s in new_shape:
        total *= s

    dt = a.dtype
    native = object.__getattribute__(a, '_native_arr')

    if total == 0:
        return zeros(new_shape, dtype=dt)

    # Build dtype_json for the new native array
    dtype_json = _json.dumps([[nm, str(dt.fields[nm][0])] for nm in dt.names])

    # Tile each field independently
    new_fields = []
    for name in dt.names:
        col = native[name]  # PyNdArray, 1D
        if len(col) == 0:
            tiled = zeros(total, dtype=str(col.dtype))
            if not isinstance(tiled, ndarray):
                tiled = asarray([0] * total).astype(str(col.dtype))
        else:
            tiled = _native_resize(col, total)
        new_fields.append((name, tiled))

    native_fields = [(name, col) for name, col in new_fields]
    new_native = _native_mod.StructuredArray(native_fields, [total], dtype_json)
    flat = StructuredArray(new_native)

    if len(new_shape) == 1:
        return flat
    return flat.reshape(new_shape)


def resize(a, new_shape):
    from numpy import StructuredArray
    if isinstance(a, StructuredArray):
        return _resize_structured(a, new_shape)
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


def _trim_zeros_is_nonzero(v):
    """Check if a value is nonzero; handles complex-as-tuple from RustPython."""
    if isinstance(v, tuple) and len(v) == 2:
        return v[0] != 0 or v[1] != 0
    try:
        return bool(v != 0)
    except (TypeError, ValueError):
        return bool(v)


def _trim_zeros_slice_nonzero(sl):
    """Return True if sl (scalar or array) has any nonzero element."""
    if isinstance(sl, ndarray):
        if sl.ndim == 0:
            v = sl[()]
            if v is None:
                return False
            return _trim_zeros_is_nonzero(v)
        for v in sl.flatten().tolist():
            if _trim_zeros_is_nonzero(v):
                return True
        return False
    # Plain Python scalar (int, float, complex, tuple, etc.)
    return _trim_zeros_is_nonzero(sl)


def trim_zeros(filt, trim='fb', axis=None):
    """Trim leading and/or trailing zeros from a 1-D array or along axes."""
    # Validate trim parameter
    for ch in trim:
        if ch not in ('f', 'F', 'b', 'B'):
            raise ValueError(
                "unexpected character(s) in `trim`: '{}'".format(trim)
            )
    if not trim:
        raise ValueError(
            "unexpected character(s) in `trim`: '{}'".format(trim)
        )

    is_list = isinstance(filt, list)
    arr = asarray(filt)

    if arr.ndim == 0:
        return arr

    trim_lower = trim.lower()

    # Determine axes to process
    if axis is None:
        axes = tuple(_builtin_range(arr.ndim))
    elif isinstance(axis, int):
        axes = (arr.ndim + axis if axis < 0 else axis,)
    else:
        ax_list = list(axis)
        if not ax_list:
            # Empty axis tuple → no trimming
            if is_list and arr.ndim == 1:
                return arr.tolist()
            return arr
        axes = tuple(arr.ndim + ax if ax < 0 else ax for ax in ax_list)

    slices = [slice(None)] * arr.ndim

    for ax in axes:
        n = arr.shape[ax]
        start = 0
        end = n

        if 'f' in trim_lower:
            found = False
            for i in _builtin_range(n):
                idx = tuple(i if dim == ax else slice(None)
                            for dim in _builtin_range(arr.ndim))
                sl = arr[idx]
                # Check if any element is nonzero (sl may be scalar or array)
                nonz = _trim_zeros_slice_nonzero(sl)
                if nonz:
                    start = i
                    found = True
                    break
            if not found:
                start = n

        if 'b' in trim_lower:
            found = False
            for i in _builtin_range(n - 1, -1, -1):
                idx = tuple(i if dim == ax else slice(None)
                            for dim in _builtin_range(arr.ndim))
                sl = arr[idx]
                nonz = _trim_zeros_slice_nonzero(sl)
                if nonz:
                    end = i + 1
                    found = True
                    break
            if not found:
                end = start  # make this axis empty

        if start >= end:
            slices[ax] = slice(0, 0)
        else:
            slices[ax] = slice(start, end)

    from ._helpers import _ObjectArray as _ObjArr
    if isinstance(arr, _ObjArr):
        # _ObjectArray doesn't support tuple indexing with slices
        # Extract elements individually for 1D case
        if arr.ndim == 1:
            sl = slices[0]
            start_i = sl.start if sl.start is not None else 0
            stop_i = sl.stop if sl.stop is not None else arr.shape[0]
            elems = [arr[i] for i in _builtin_range(start_i, stop_i)]
            result = asarray(elems)
            if is_list:
                return elems
            return result
    result = arr[tuple(slices)]

    if is_list and arr.ndim == 1:
        return result.tolist()
    return result
