"""numpy.polynomial.laguerre - Laguerre polynomial functions.

Provides module-level access to Laguerre polynomial functions for imports like:
    from numpy.polynomial.laguerre import lagval
"""
from numpy.polynomial import laguerre as _lag

lagval = _lag.lagval
lagfit = _lag.lagfit
lagadd = _lag.lagadd
lagsub = _lag.lagsub
lagmul = _lag.lagmul
lagder = _lag.lagder
lagint = _lag.lagint
