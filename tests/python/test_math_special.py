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

print(f"test_math_special: {passed} passed, {failed} failed")
if failed:
    raise SystemExit(1)
