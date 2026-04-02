"""numpy.lib - library of routines."""
import sys as _sys
from numpy.lib import stride_tricks, mixins, format

# Register all _*_impl submodules under their public names
from numpy.lib import (
    _index_tricks_impl as index_tricks,
    _function_base_impl as function_base,
    _arraypad_impl as arraypad,
    _arraysetops_impl as arraysetops,
    _histograms_impl as histograms,
    _nanfunctions_impl as nanfunctions,
    _polynomial_impl as polynomial,
    _shape_base_impl as shape_base,
    _stride_tricks_impl as _stride_tricks_impl_mod,
    _twodim_base_impl as twodim_base,
    _type_check_impl as type_check,
    _ufunclike_impl as ufunclike,
)

# Make them importable as numpy.lib.X
_sys.modules['numpy.lib.index_tricks'] = index_tricks
_sys.modules['numpy.lib.function_base'] = function_base
_sys.modules['numpy.lib.arraypad'] = arraypad
_sys.modules['numpy.lib.arraysetops'] = arraysetops
_sys.modules['numpy.lib.histograms'] = histograms
_sys.modules['numpy.lib.nanfunctions'] = nanfunctions
_sys.modules['numpy.lib.polynomial'] = polynomial
_sys.modules['numpy.lib.shape_base'] = shape_base
_sys.modules['numpy.lib.twodim_base'] = twodim_base
_sys.modules['numpy.lib.type_check'] = type_check
_sys.modules['numpy.lib.ufunclike'] = ufunclike


class NumpyVersion:
    """Parse and compare NumPy version strings."""
    def __init__(self, vstring):
        self.vstring = vstring
        parts = vstring.split('.')
        self.major = int(parts[0]) if len(parts) > 0 else 0
        self.minor = int(parts[1]) if len(parts) > 1 else 0
        self.bugfix = int(parts[2].split('rc')[0].split('a')[0].split('b')[0]) if len(parts) > 2 else 0

    def __repr__(self):
        return f"NumpyVersion('{self.vstring}')"

    def __str__(self):
        return self.vstring

    def _compare(self, other):
        if isinstance(other, str):
            other = NumpyVersion(other)
        for a, b in [(self.major, other.major), (self.minor, other.minor), (self.bugfix, other.bugfix)]:
            if a < b:
                return -1
            if a > b:
                return 1
        return 0

    def __eq__(self, other): return self._compare(other) == 0
    def __ne__(self, other): return self._compare(other) != 0
    def __lt__(self, other): return self._compare(other) < 0
    def __le__(self, other): return self._compare(other) <= 0
    def __gt__(self, other): return self._compare(other) > 0
    def __ge__(self, other): return self._compare(other) >= 0


class _ScimathModule:
    """numpy.lib.scimath — complex-safe math functions.

    Unlike regular numpy, these return complex results for negative inputs
    (e.g. sqrt(-1) = 1j, log(-1) = pi*j).
    """
    @staticmethod
    def sqrt(x):
        import numpy as np
        import cmath
        x = np.asarray(x)
        if x.ndim == 0:
            v = float(x)
            if v < 0:
                return complex(cmath.sqrt(v))
            return np.sqrt(x)
        flat = x.flatten().tolist()
        any_neg = any(v < 0 for v in flat)
        if any_neg:
            return np.array([complex(cmath.sqrt(v)) for v in flat]).reshape(x.shape)
        return np.sqrt(x)

    @staticmethod
    def log(x):
        import numpy as np
        import cmath
        x = np.asarray(x)
        if x.ndim == 0:
            v = float(x)
            if v <= 0:
                return complex(cmath.log(v))
            return np.log(x)
        result = []
        for v in x.flatten().tolist():
            if v <= 0:
                result.append(complex(cmath.log(v)))
            else:
                result.append(complex(cmath.log(v)))
        return np.array(result)

    @staticmethod
    def log2(x):
        import numpy as np
        import cmath, math
        x = np.asarray(x)
        if x.ndim == 0:
            v = float(x)
            if v <= 0:
                return complex(cmath.log(v) / math.log(2))
            return np.log2(x)
        return _ScimathModule.log(x) / math.log(2)

    @staticmethod
    def log10(x):
        import numpy as np
        import cmath, math
        x = np.asarray(x)
        if x.ndim == 0:
            v = float(x)
            if v <= 0:
                return complex(cmath.log10(v))
            return np.log10(x)
        result = []
        for v in np.asarray(x).flatten().tolist():
            result.append(complex(cmath.log10(v)))
        return np.array(result)

    @staticmethod
    def power(x, p):
        import numpy as np
        return np.power(np.asarray(x, dtype='complex128'), p)

    @staticmethod
    def arccos(x):
        import numpy as np
        import cmath
        x = np.asarray(x)
        if x.ndim == 0:
            return complex(cmath.acos(float(x)))
        return np.array([complex(cmath.acos(v)) for v in x.flatten().tolist()])

    @staticmethod
    def arcsin(x):
        import numpy as np
        import cmath
        x = np.asarray(x)
        if x.ndim == 0:
            return complex(cmath.asin(float(x)))
        return np.array([complex(cmath.asin(v)) for v in x.flatten().tolist()])

    def __getattr__(self, name):
        import numpy as np
        if hasattr(np, name):
            return getattr(np, name)
        raise AttributeError(f"module 'numpy.lib.scimath' has no attribute '{name}'")

scimath = _ScimathModule()

import sys as _sys
_sys.modules['numpy.lib.scimath'] = scimath
