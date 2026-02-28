"""numpy.polynomial.legendre - Legendre polynomial functions.

Provides module-level access to Legendre polynomial functions for imports like:
    from numpy.polynomial.legendre import legval
"""
from numpy.polynomial import legendre as _leg

legval = _leg.legval
legfit = _leg.legfit
legadd = _leg.legadd
legsub = _leg.legsub
legmul = _leg.legmul
legder = _leg.legder
legint = _leg.legint
