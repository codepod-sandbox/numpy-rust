# NumPy Test Compatibility Design

## Goal

Run NumPy's own test suite against numpy-rust to measure and drive compatibility, starting with `test_numeric.py` and `test_multiarray.py` from NumPy 2.2.x.

## Context

- Real pytest 8.4.2 runs natively in the numpy-python binary (RustPython git main)
- 159 Rust core tests + 75 Python integration tests already pass
- The binary has `freeze-stdlib` + `host_env` features enabled

## Architecture

### numpy.testing (pure Python)

A Python module at `tests/numpy_compat/numpy/testing/` providing NumPy's standard assertion functions. These wrap our existing numpy operations with tolerance-based comparison logic:

- `assert_equal(actual, desired)` — exact equality
- `assert_array_equal(x, y)` — element-wise array equality
- `assert_allclose(actual, desired, rtol, atol)` — tolerance-based floating point comparison
- `assert_almost_equal(actual, desired, decimal)` — decimal-place comparison
- `assert_array_almost_equal(actual, desired, decimal)` — array version
- `assert_raises(exception, callable, *args)` — context manager for expected exceptions
- `assert_array_less(x, y)` — element-wise less-than
- `assert_warns` — stub (warnings module may not fully work)

### Vendored NumPy tests

Copy `test_numeric.py` and `test_multiarray.py` from NumPy 2.2.3 into `tests/numpy_compat/`. Run unmodified — skip/xfail tests that need unimplemented features rather than editing the test source.

### Stubs for C-extension test modules

Minimal Python modules that raise `pytest.skip` or provide no-op implementations:

- `numpy._core._multiarray_tests` — C test helpers (e.g. `array_indexing_abort`)
- `numpy._core._rational_tests` — rational dtype tests
- `hypothesis` — property-based testing (not needed initially)

### Module resolution

The numpy-python binary's `sys.path` will include `tests/numpy_compat/` so that `import numpy` resolves to our Rust-backed module while `from numpy.testing import ...` finds the pure Python testing utilities.

## Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Test source | Vendored snapshot | Reproducible, no git submodule complexity |
| Initial scope | test_numeric.py, test_multiarray.py | Core array operations, highest coverage value |
| numpy.testing | Pure Python | No C dependencies, straightforward to implement |
| pytest | Real pytest 8.4.2 | Already working in RustPython, no stubs needed |
| Test runner | pytest via numpy-python binary | `numpy-python -m pytest tests/numpy_compat/` |
