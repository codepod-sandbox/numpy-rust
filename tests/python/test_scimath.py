"""Tests for numpy.lib.scimath complex-safe math functions."""
import numpy as np

passed = 0
failed = 0

def check(name, got, expected, tol=1e-8):
    global passed, failed
    if abs(got - expected) <= tol:
        passed += 1
    else:
        print(f"FAIL {name}: got {got!r}, expected {expected!r}")
        failed += 1

def get_imag(v):
    """Get imaginary part from complex value (may be tuple (re, im) or complex)."""
    if isinstance(v, tuple):
        return v[1]
    if isinstance(v, complex):
        return v.imag
    if hasattr(v, 'imag'):
        return v.imag
    return 0.0

def is_complex(v):
    """Check if value is complex (tuple (re,im) or Python complex)."""
    if isinstance(v, tuple) and len(v) == 2:
        return True
    if isinstance(v, complex):
        return True
    if hasattr(v, 'imag') and not isinstance(v, (int, float)):
        return True
    return False

sm = np.lib.scimath

# sqrt of positive: real result
r = sm.sqrt(np.array([4.0, 9.0]))
check("sqrt(4)", float(r.flatten().tolist()[0]), 2.0)

# sqrt of negative: complex result
r = sm.sqrt(np.array([-4.0]))
v = r.flatten().tolist()[0]
assert is_complex(v), f"sqrt(-4) should be complex, got {v!r}"
check("sqrt(-4).imag", abs(get_imag(v)), 2.0)
passed += 1

# log of negative: complex
r = sm.log(np.array([-1.0]))
v = r.flatten().tolist()[0]
import math as _m
assert is_complex(v), f"log(-1) should be complex, got {v!r}"
check("log(-1).imag", abs(get_imag(v)), _m.pi)

# arcsin out-of-domain: complex
r = sm.arcsin(np.array([2.0]))
v = r.flatten().tolist()[0]
assert is_complex(v), f"arcsin(2) should be complex, got {v!r}"
passed += 1

# arccos out-of-domain: complex
r = sm.arccos(np.array([2.0]))
v = r.flatten().tolist()[0]
assert is_complex(v), f"arccos(2) should be complex, got {v!r}"
passed += 1

# power with negative base
r = sm.power(np.array([-1.0]), np.array([0.5]))
v = r.flatten().tolist()[0]
assert is_complex(v), f"(-1)^0.5 should be complex, got {v!r}"
passed += 1

print(f"test_scimath: {passed} passed, {failed} failed")
if failed:
    raise SystemExit(1)
