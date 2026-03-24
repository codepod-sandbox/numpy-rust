"""numpy.ma stub - masked array support."""
import numpy as _np


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
            self.mask = mask.tolist()
        elif hasattr(mask, '__iter__'):
            self.mask = list(mask)
        else:
            self.mask = [bool(mask)] * n

    def __len__(self):
        return len(self._data)

    def __getitem__(self, idx):
        return self._data[idx]

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
