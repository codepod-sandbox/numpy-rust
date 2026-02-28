"""numpy.polynomial.hermite - Physicist's Hermite polynomial functions.

Provides module-level access to Hermite polynomial functions for imports like:
    from numpy.polynomial.hermite import hermval
"""
from numpy.polynomial import hermite as _herm

hermval = _herm.hermval
hermfit = _herm.hermfit
hermadd = _herm.hermadd
hermsub = _herm.hermsub
hermmul = _herm.hermmul
hermder = _herm.hermder
hermint = _herm.hermint
