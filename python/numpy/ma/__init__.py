"""numpy.ma - masked array support."""


class MaskedConstant:
    """The masked constant."""
    def __repr__(self):
        return '--'
    def __str__(self):
        return '--'
    def __bool__(self):
        return False
    def __eq__(self, other):
        return isinstance(other, MaskedConstant)


masked = MaskedConstant()


class MaskedArray:
    """Simplified masked array."""
    def __init__(self, data, mask=None, dtype=None, fill_value=None):
        import numpy as np
        self.data = np.asarray(data)
        if mask is None:
            self.mask = np.zeros(self.data.shape).astype("bool")
        elif isinstance(mask, bool) and not mask:
            self.mask = np.zeros(self.data.shape).astype("bool")
        elif isinstance(mask, bool) and mask:
            self.mask = np.ones(self.data.shape).astype("bool")
        else:
            self.mask = np.asarray(mask).astype("bool")
        self._fill_value = fill_value if fill_value is not None else 0.0

    @property
    def shape(self):
        return self.data.shape

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
    def fill_value(self):
        return self._fill_value

    def filled(self, fill_value=None):
        import numpy as np
        fv = fill_value if fill_value is not None else self._fill_value
        result = self.data.copy()
        mask_list = self.mask.tolist()
        data_list = result.tolist()
        if isinstance(data_list, list) and len(data_list) > 0 and isinstance(data_list[0], list):
            # 2D
            for i in range(len(data_list)):
                for j in range(len(data_list[i])):
                    if mask_list[i][j]:
                        data_list[i][j] = float(fv)
        else:
            # 1D
            if isinstance(data_list, list):
                for i in range(len(data_list)):
                    if mask_list[i]:
                        data_list[i] = float(fv)
            else:
                if mask_list:
                    data_list = float(fv)
        return np.array(data_list)

    def compressed(self):
        """Return all non-masked values."""
        import numpy as np
        data_list = self.data.flatten().tolist()
        mask_list = self.mask.flatten().tolist()
        return np.array([d for d, m in zip(data_list, mask_list) if not m])

    def count(self, axis=None):
        """Count non-masked elements."""
        mask_list = self.mask.flatten().tolist()
        return sum(1 for m in mask_list if not m)

    def sum(self, axis=None):
        return self.compressed().sum() if axis is None else self.filled(0.0).sum(axis=axis)

    def mean(self, axis=None):
        c = self.compressed()
        if len(c.tolist()) == 0:
            return float('nan')
        return c.mean() if axis is None else self.filled(0.0).mean(axis=axis)

    def tolist(self):
        return self.data.tolist()

    def __repr__(self):
        return "masked_array(data={}, mask={})".format(self.data.tolist(), self.mask.tolist())

    def __getitem__(self, key):
        d = self.data[key]
        m = self.mask[key]
        return MaskedArray(d, mask=m, fill_value=self._fill_value)

    def __len__(self):
        return len(self.data)

    # Arithmetic operators
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
    def __pow__(self, other): return self._binop(other, lambda a, b: a ** b)
    def __neg__(self):
        import numpy as np
        return MaskedArray(-self.data, mask=self.mask, fill_value=self._fill_value)
    def __abs__(self):
        import numpy as np
        return MaskedArray(np.abs(self.data), mask=self.mask, fill_value=self._fill_value)

    def sum(self, axis=None, keepdims=False):
        import numpy as np
        filled = self.filled(0.0)
        result = np.sum(filled, axis=axis, keepdims=keepdims)
        return result

    def mean(self, axis=None, keepdims=False):
        import numpy as np
        filled = self.filled(0.0)
        not_mask = np.logical_not(self.mask).astype("float64")
        s = np.sum(filled * not_mask, axis=axis, keepdims=keepdims)
        c = np.sum(not_mask, axis=axis, keepdims=keepdims)
        return s / c

    def copy(self):
        import numpy as np
        return MaskedArray(self.data.copy(), mask=self.mask.copy(), fill_value=self._fill_value)

    def astype(self, dtype):
        return MaskedArray(self.data.astype(dtype), mask=self.mask, fill_value=self._fill_value)

    def flatten(self):
        import numpy as np
        return MaskedArray(self.data.flatten(), mask=self.mask.flatten(), fill_value=self._fill_value)

    def tolist(self):
        return self.data.tolist()


def masked_array(data, mask=None, dtype=None, fill_value=None):
    """Create a masked array."""
    return MaskedArray(data, mask=mask, dtype=dtype, fill_value=fill_value)


def array(data, mask=None, dtype=None, fill_value=None):
    """Create a masked array (alias)."""
    return MaskedArray(data, mask=mask, dtype=dtype, fill_value=fill_value)


def is_masked(x):
    """Test whether input has masked values."""
    if isinstance(x, MaskedArray):
        mask_list = x.mask.flatten().tolist()
        return any(m for m in mask_list)
    return False


def masked_equal(x, value):
    """Mask where equal to value."""
    import numpy as np
    x = np.asarray(x)
    mask = (x == value)
    return MaskedArray(x, mask=mask)


def masked_greater(x, value):
    """Mask where greater than value."""
    import numpy as np
    x = np.asarray(x)
    mask = (x > value)
    return MaskedArray(x, mask=mask)


def masked_less(x, value):
    """Mask where less than value."""
    import numpy as np
    x = np.asarray(x)
    mask = (x < value)
    return MaskedArray(x, mask=mask)


def masked_where(condition, x):
    """Mask where condition is True."""
    import numpy as np
    return MaskedArray(np.asarray(x), mask=np.asarray(condition))


def masked_invalid(x):
    """Mask NaN and Inf values."""
    import numpy as np
    x = np.asarray(x)
    mask = np.logical_or(np.isnan(x), np.isinf(x))
    return MaskedArray(x, mask=mask)


def masked_less_equal(x, value):
    """Mask where less than or equal to value."""
    import numpy as np
    x = np.asarray(x)
    mask = (x <= value)
    return MaskedArray(x, mask=mask)


def masked_greater_equal(x, value):
    """Mask where greater than or equal to value."""
    import numpy as np
    x = np.asarray(x)
    mask = (x >= value)
    return MaskedArray(x, mask=mask)


def masked_not_equal(x, value):
    """Mask where not equal to value."""
    import numpy as np
    x = np.asarray(x)
    mask = (x != value)
    return MaskedArray(x, mask=mask)


def masked_inside(x, v1, v2):
    """Mask where between v1 and v2 (inclusive)."""
    import numpy as np
    x = np.asarray(x)
    mask = np.logical_and(x >= min(v1, v2), x <= max(v1, v2))
    return MaskedArray(x, mask=mask)


def zeros(shape, dtype=None):
    """Return a masked array of zeros."""
    import numpy as np
    return MaskedArray(np.zeros(shape, dtype=dtype or "float64"))

def ones(shape, dtype=None):
    """Return a masked array of ones."""
    import numpy as np
    return MaskedArray(np.ones(shape, dtype=dtype or "float64"))

def empty(shape, dtype=None):
    """Return a masked array of uninitialized data (zeros in practice)."""
    import numpy as np
    return MaskedArray(np.zeros(shape, dtype=dtype or "float64"))

def masked_outside(x, v1, v2):
    """Mask where outside v1 and v2."""
    import numpy as np
    x = np.asarray(x)
    mask = np.logical_or(x < min(v1, v2), x > max(v1, v2))
    return MaskedArray(x, mask=mask)


def getmaskarray(arr):
    """Return the mask of a masked array, or full False mask."""
    import numpy as np
    if isinstance(arr, MaskedArray):
        return arr.mask
    return np.zeros(np.asarray(arr).shape).astype("bool")


def getdata(arr):
    """Return data of a masked array as ndarray."""
    if isinstance(arr, MaskedArray):
        return arr.data
    import numpy as np
    return np.asarray(arr)


def fix_invalid(a, mask=None, copy=True, fill_value=None):
    """Return with invalid data (NaN/Inf) masked and replaced."""
    import numpy as np
    a = np.asarray(a)
    invalid_mask = np.logical_or(np.isnan(a), np.isinf(a))
    if mask is not None:
        combined = np.logical_or(invalid_mask, np.asarray(mask))
    else:
        combined = invalid_mask
    result = MaskedArray(a, mask=combined, fill_value=fill_value)
    return result
