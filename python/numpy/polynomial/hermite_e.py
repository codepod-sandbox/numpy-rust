"""numpy.polynomial.hermite_e - Probabilist's Hermite polynomial functions.

Provides module-level access to probabilist's Hermite polynomial functions for imports like:
    from numpy.polynomial.hermite_e import hermeval
"""
from numpy.polynomial import hermite_e as _herme

hermeval = _herme.hermeval
hermefit = _herme.hermefit
hermeadd = _herme.hermeadd
hermesub = _herme.hermesub
hermeder = _herme.hermeder
hermeint = _herme.hermeint
