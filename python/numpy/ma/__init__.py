"""numpy.ma stub - masked array support."""
import numpy as _np


# Sentinel value for masked elements
masked = None  # used as np.ma.masked
nomask = False  # used as np.ma.nomask


class MaskedArray:
    """Minimal masked array implementation for RustPython compatibility."""

    def __init__(self, data, mask=None):
        import numpy as _np
        if hasattr(data, 'tolist'):
            self._data = data
        else:
            self._data = _np.array(data, dtype=float)
        n = len(self._data) if hasattr(self._data, '__len__') else 1
        if mask is None:
            self.mask = [False] * n
        elif hasattr(mask, 'tolist'):
            raw = mask.tolist()
            self.mask = [bool(v) for v in raw] if isinstance(raw, list) else [bool(raw)]
        elif hasattr(mask, '__iter__'):
            self.mask = [bool(v) for v in mask]
        else:
            self.mask = [bool(mask)] * n

    def __len__(self):
        return len(self._data)

    @property
    def shape(self):
        return self._data.shape if hasattr(self._data, 'shape') else (len(self._data),)

    def __getitem__(self, idx):
        if self.mask[idx] if isinstance(idx, int) and 0 <= idx < len(self.mask) else False:
            return masked
        return self._data[idx]

    def __setitem__(self, idx, value):
        if value is masked:
            if isinstance(idx, int):
                self.mask[idx] = True
            else:
                for i in range(len(self.mask)):
                    self.mask[i] = True
        else:
            self._data[idx] = value
            if isinstance(idx, int) and 0 <= idx < len(self.mask):
                self.mask[idx] = False

    def __iter__(self):
        return iter(self._data.tolist() if hasattr(self._data, 'tolist') else self._data)

    def tolist(self):
        return self._data.tolist() if hasattr(self._data, 'tolist') else list(self._data)

    @property
    def data(self):
        return self._data

    def __truediv__(self, other):
        import numpy as _np
        result_data = self._data / float(other)
        return MaskedArray(result_data, list(self.mask))

    def filled(self, fill_value=0):
        """Return data with masked values replaced by fill_value."""
        import numpy as _np
        vals = self.tolist()
        result = [fill_value if self.mask[i] else vals[i] for i in range(len(vals))]
        return _np.array(result, dtype=float)

    def compressed(self):
        """Return data with masked values removed."""
        import numpy as _np
        vals = self.tolist()
        result = [vals[i] for i in range(len(vals)) if not self.mask[i]]
        return _np.array(result, dtype=float)

    def __repr__(self):
        return f"MaskedArray({self._data}, mask={self.mask})"


def masked_where(condition, a, copy=True):
    """Return `a` as a MaskedArray with entries masked where `condition` is True."""
    import numpy as _np
    if hasattr(condition, 'tolist'):
        mask = condition.tolist()
    elif hasattr(condition, '__iter__'):
        mask = list(condition)
    else:
        mask = [bool(condition)]
    if hasattr(a, 'copy') and copy:
        data = a.copy()
    else:
        data = a
    return MaskedArray(data, mask)


def masked_invalid(a, copy=True):
    """Mask NaN and Inf values."""
    import numpy as _np, math
    if hasattr(a, 'tolist'):
        vals = a.tolist()
    else:
        vals = list(a)
    mask = [math.isnan(v) or math.isinf(v) for v in vals]
    return MaskedArray(_np.array(vals, dtype=float), mask)


def log(a):
    """Element-wise natural log of a MaskedArray."""
    import math
    import numpy as _np
    if isinstance(a, MaskedArray):
        vals = a.tolist()
        result = []
        mask = list(a.mask)
        for i, v in enumerate(vals):
            if mask[i] or v <= 0:
                result.append(0.0)
                mask[i] = True
            else:
                result.append(math.log(v))
        return MaskedArray(_np.array(result, dtype=float), mask)
    # fallback: apply numpy log
    return _np.log(_np.asarray(a, dtype=float))


def filled(a, fill_value=0):
    """Return data with masked values replaced by fill_value."""
    if isinstance(a, MaskedArray):
        vals = a.tolist()
        result = [fill_value if a.mask[i] else vals[i] for i in range(len(vals))]
        return _np.array(result, dtype=float)
    return a


def array(data, mask=None, dtype=None):
    """Create a MaskedArray (np.ma.array equivalent)."""
    if isinstance(data, MaskedArray):
        if mask is None:
            mask = data.mask
        data = data._data
    if dtype is not None:
        if hasattr(data, 'tolist'):
            data = _np.array(data.tolist(), dtype=dtype)
        else:
            data = _np.array(data, dtype=dtype)
    return MaskedArray(data, mask)


def getmaskarray(a):
    """Return the mask of a masked array as a boolean array."""
    if isinstance(a, MaskedArray):
        return _np.array(a.mask, dtype=bool)
    # Non-masked arrays have all-False mask
    if hasattr(a, 'shape'):
        return _np.zeros(a.shape, dtype=bool)
    n = len(a) if hasattr(a, '__len__') else 1
    return _np.zeros(n, dtype=bool)


def getmask(a):
    """Return the mask of a masked array, or nomask."""
    if isinstance(a, MaskedArray):
        return a.mask
    return nomask


def compress_rowcols(a, axis=0):
    """Not fully implemented."""
    return a
