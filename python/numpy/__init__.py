"""NumPy-compatible Python package wrapping the Rust native module."""
import sys as _sys
import math as _stdlib_math
from functools import reduce as _reduce

__version__ = "1.26.0"

# Import from native Rust module
import _numpy_native as _native
from _numpy_native import ndarray
from _numpy_native import dot
from _numpy_native import concatenate as _native_concatenate
from ._helpers import *
from ._core_types import *
from ._datetime import *
from ._creation import *
from ._math import *
from ._reductions import *
from ._manipulation import *
from ._bitwise import *
from ._ufunc import *
from ._poly import *
from ._indexing import *
from ._window import *
from ._io import *
from ._stubs import *
from ._linalg_ext import *
from ._fft_ext import *
from ._random_ext import *


# Import submodules so they're accessible as numpy.linalg etc.
from _numpy_native import linalg, fft, random

# Register Rust submodules in sys.modules so `from numpy.random import ...` works
_sys.modules["numpy.linalg"] = linalg
_sys.modules["numpy.fft"] = fft
_sys.modules["numpy.random"] = random


# --- Constants --------------------------------------------------------------
nan = float("nan")
inf = float("inf")
pi = _stdlib_math.pi
e = _stdlib_math.e
newaxis = None
PINF = float("inf")
NINF = float("-inf")
PZERO = 0.0
NZERO = -0.0
Inf = inf
Infinity = inf
NaN = nan
NAN = nan
euler_gamma = 0.5772156649015329
ALLOW_THREADS = 1
little_endian = True

# numpy 1.x compat: np.bool (deprecated, can't shadow builtin 'bool' in module
# scope since isinstance checks recurse). We set it via __getattr__ below.

# --- Aliases ----------------------------------------------------------------

absolute = abs
conjugate = conj

radians = deg2rad
degrees = rad2deg

round_ = around
round = around

mod = remainder
divmod = divmod_

special = type('special', (), {
    'gamma': staticmethod(gamma),
    'erf': staticmethod(erf),
    'erfc': staticmethod(erfc),
    'lgamma': staticmethod(lgamma),
    'j0': staticmethod(j0),
    'j1': staticmethod(j1),
    'y0': staticmethod(y0),
    'y1': staticmethod(y1),
})()

# Link top-level functions into linalg module
linalg.trace = trace
linalg.cross = cross        # delegate to top-level cross()
linalg.diagonal = diagonal  # delegate to top-level diagonal()
linalg.outer = outer        # delegate to top-level outer()

# --- Import submodules so np.ma and np.polynomial are accessible ------------
import numpy.ma as ma
import numpy.polynomial as polynomial

# --- Module-level __getattr__ for deprecated aliases like np.bool -----------
def __getattr__(name):
    _bi = __import__("builtins")
    _deprecated_aliases = {
        'bool': _bi.bool,
        'int': _bi.int,
        'float': _bi.float,
        'complex': _bi.complex,
        'str': _bi.str,
        'object': _bi.object,
    }
    if name in _deprecated_aliases:
        return _deprecated_aliases[name]
    raise AttributeError(f"module 'numpy' has no attribute '{name}'")
