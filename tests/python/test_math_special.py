"""Tests for libm-backed special math functions."""
import numpy as np
import math

passed = 0
failed = 0

def check(name, got, expected, tol=1e-8):
    global passed, failed
    if abs(got - expected) <= tol:
        passed += 1
    else:
        print(f"FAIL {name}: got {got}, expected {expected}")
        failed += 1

def check_arr(name, arr, expected_list, tol=1e-8):
    vals = arr.flatten().tolist()
    for i, (g, e) in enumerate(zip(vals, expected_list)):
        check(f"{name}[{i}]", g, e, tol)

# cbrt
check_arr("cbrt", np.cbrt([8.0, -27.0, 1.0]), [2.0, -3.0, 1.0])

# gamma
check("gamma(1)", float(np.gamma(np.array([1.0])).flatten().tolist()[0]), 1.0)
check("gamma(5)", float(np.gamma(np.array([5.0])).flatten().tolist()[0]), 24.0)

# lgamma
check("lgamma(1)", float(np.lgamma(np.array([1.0])).flatten().tolist()[0]), 0.0)
check("lgamma(5)", float(np.lgamma(np.array([5.0])).flatten().tolist()[0]), math.log(24.0))

# erf
check("erf(0)", float(np.erf(np.array([0.0])).flatten().tolist()[0]), 0.0)
check("erf(1)", float(np.erf(np.array([1.0])).flatten().tolist()[0]), 0.8427007929, tol=1e-6)

# erfc
check("erfc(0)", float(np.erfc(np.array([0.0])).flatten().tolist()[0]), 1.0)

# j0
check("j0(0)", float(np.j0(np.array([0.0])).flatten().tolist()[0]), 1.0)

# j1
check("j1(0)", float(np.j1(np.array([0.0])).flatten().tolist()[0]), 0.0)

# y0: y0(1) ≈ 0.0882569642
check("y0(1)", float(np.y0(np.array([1.0])).flatten().tolist()[0]), 0.0882569642, tol=1e-7)

# y1: y1(1) ≈ -0.7812128213
check("y1(1)", float(np.y1(np.array([1.0])).flatten().tolist()[0]), -0.7812128213, tol=1e-4)

# copysign
a = np.copysign(np.array([1.0, -2.0, 3.0]), np.array([-1.0, 1.0, -1.0]))
check_arr("copysign", a, [-1.0, 2.0, -3.0])

# hypot
check("hypot(3,4)", float(np.hypot(np.array([3.0]), np.array([4.0])).flatten().tolist()[0]), 5.0)

# fmod
check("fmod(7,3)", float(np.fmod(np.array([7.0]), np.array([3.0])).flatten().tolist()[0]), 1.0)

# ldexp
check("ldexp(1,3)", float(np.ldexp(np.array([1.0]), np.array([3])).flatten().tolist()[0]), 8.0)

# maximum (NaN-propagating)
r = np.maximum(np.array([float('nan'), 1.0, 3.0]), np.array([1.0, float('nan'), 2.0]))
vals = r.flatten().tolist()
assert vals[0] != vals[0], f"maximum NaN propagation failed: {vals[0]}"  # NaN != NaN
passed += 1

# fmax (NaN-ignoring)
r = np.fmax(np.array([float('nan'), 1.0, 3.0]), np.array([1.0, float('nan'), 2.0]))
vals = r.flatten().tolist()
check("fmax NaN-ignore[0]", vals[0], 1.0)
check("fmax NaN-ignore[1]", vals[1], 1.0)

# logaddexp
import math as _m
expected = _m.log(_m.exp(1.0) + _m.exp(2.0))
check("logaddexp(1,2)", float(np.logaddexp(np.array([1.0]), np.array([2.0])).flatten().tolist()[0]), expected)

# frexp
m, e = np.frexp(np.array([12.0]))
check("frexp mantissa", float(m.flatten().tolist()[0]), 0.75)
check("frexp exponent", float(e.flatten().tolist()[0]), 4.0)

# modf
frac, intg = np.modf(np.array([3.7, -2.5]))
check("modf frac[0]", float(frac.flatten().tolist()[0]), 0.7)
check("modf int[0]", float(intg.flatten().tolist()[0]), 3.0)

print(f"test_math_special: {passed} passed, {failed} failed")
if failed:
    raise SystemExit(1)
