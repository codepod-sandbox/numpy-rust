"""numpy.polynomial.polynomial - polynomial functions.

Provides module-level access to polynomial functions for imports like:
    from numpy.polynomial.polynomial import polyval
"""
from numpy.polynomial import polynomial as _poly

polyval = _poly.polyval
polyfit = _poly.polyfit
polyadd = _poly.polyadd
polysub = _poly.polysub
polymul = _poly.polymul
polyder = _poly.polyder
polyint = _poly.polyint
