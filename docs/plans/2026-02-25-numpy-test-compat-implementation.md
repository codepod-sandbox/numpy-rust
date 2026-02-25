# NumPy Test Compatibility Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Run NumPy's `test_numeric.py` against numpy-rust and establish a pass-rate baseline.

**Architecture:** Pure Python `numpy.testing` module injected into the Rust numpy namespace via conftest.py. Vendored NumPy test file run with real pytest. Stubs for missing internal modules.

**Tech Stack:** Python (numpy.testing assertions), pytest 8.4.2, RustPython

---

### Task 1: Create directory structure and conftest.py

**Files:**
- Create: `tests/numpy_compat/conftest.py`
- Create: `tests/numpy_compat/_support/__init__.py`

**Step 1: Create the directories**

```bash
mkdir -p tests/numpy_compat/_support
```

**Step 2: Write conftest.py**

This file injects pure-Python submodules into the Rust-backed numpy namespace so that `from numpy.testing import assert_equal` works. It stubs out missing internal modules.

**Step 3: Create empty `_support/__init__.py`**

**Step 4: Verify conftest loads**

Run numpy-python with a test import to confirm the module injection works.

**Step 5: Commit**

---

### Task 2: Implement numpy.testing assertion functions

**Files:**
- Create: `tests/numpy_compat/_support/testing_utils.py`

Implement these assertion functions using our numpy operations:
- `assert_`, `assert_equal`, `assert_almost_equal`, `assert_approx_equal`
- `assert_array_equal`, `assert_array_almost_equal`, `assert_allclose`
- `assert_array_less`, `assert_array_compare`
- `assert_raises`, `assert_raises_regex` (delegate to pytest.raises)
- `assert_warns`, `assert_no_warnings`
- `assert_array_max_ulp`, `assert_string_equal`
- Platform flags: `HAS_REFCOUNT`, `IS_WASM`, `IS_PYPY`, `IS_PYSTON`
- Utilities: `suppress_warnings`, `break_cycles`, `runstring`, `temppath`

**Step 2: Verify the module loads**

**Step 3: Commit**

---

### Task 3: Vendor test_numeric.py from NumPy 2.2.3

**Files:**
- Create: `tests/numpy_compat/test_numeric.py` (vendored from NumPy)

**Step 1: Download from GitHub**

Fetch `numpy/_core/tests/test_numeric.py` from NumPy v2.2.3.

**Step 2: Verify (~4210 lines)**

**Step 3: Commit the vendored test file**

---

### Task 4: Run baseline and record results

**Step 1: Run pytest --collect-only to see what's discovered**

This will likely show import errors. Record them.

**Step 2: Fix import errors iteratively**

Based on errors, update conftest.py stubs until collection succeeds. Common issues:
- Missing `numpy._core` submodule attributes
- Missing numpy functions
- Import of `rational` from `_rational_tests`

**Step 3: Run the actual tests and record pass/fail/error/skip counts**

**Step 4: Commit the working state**

---

### Task 5: Improve pass rate with targeted fixes

Iterative. For each batch of failing tests:

1. Identify the most common failure reason
2. If missing numpy function: implement in numpy-rust-core + python bindings
3. If missing stub: add to conftest.py
4. If unsupportable: mark as xfail with reason
5. Re-run and measure improvement

Priority failure categories (expected):
- Missing dtypes (complex, uint, string types)
- Missing array methods (clip, repeat, take, put, compress)
- Missing top-level functions (full, empty, fromiter, indices)
- Missing ufuncs (sin, cos, exp, log)
- Float edge cases (nan, inf handling)
