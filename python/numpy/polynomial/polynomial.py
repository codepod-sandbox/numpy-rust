"""numpy.polynomial.polynomial - polynomial functions.

Provides module-level access to polynomial functions for imports like:
    from numpy.polynomial.polynomial import polyval
    import numpy.polynomial.polynomial as poly
"""
from numpy.polynomial import polynomial as _poly

# Class
Polynomial = _poly.Polynomial

# Constants
polydomain = _poly.polydomain
polyzero = _poly.polyzero
polyone = _poly.polyone
polyx = _poly.polyx

# Functions
polytrim = _poly.polytrim
polyline = _poly.polyline
polyadd = _poly.polyadd
polysub = _poly.polysub
polymulx = _poly.polymulx
polymul = _poly.polymul
polydiv = _poly.polydiv
polypow = _poly.polypow
polyval = _poly.polyval
polyval2d = _poly.polyval2d
polyval3d = _poly.polyval3d
polygrid2d = _poly.polygrid2d
polygrid3d = _poly.polygrid3d
polyder = _poly.polyder
polyint = _poly.polyint
polyroots = _poly.polyroots
polyvander = _poly.polyvander
polyvander2d = _poly.polyvander2d
polyvander3d = _poly.polyvander3d
polycompanion = _poly.polycompanion
polyfromroots = _poly.polyfromroots
polyvalfromroots = _poly.polyvalfromroots
polyfit = _poly.polyfit
