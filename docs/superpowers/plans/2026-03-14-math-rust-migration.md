# Math Python-to-Rust Migration Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace all Python-level math loops and composites in `_math.py`, `_reductions.py`, and `_ScimathModule` with Rust implementations backed by the `libm` crate, making the NumPy implementation suitable for WASM.

**Architecture:** Add `libm = "0.2"` (pure-Rust, WASM-native C math port) to `numpy-rust-core`. New functions land in `ops/math.rs` using existing macro patterns (`float_only_unary!`) plus two new macros (`math_binary!`, `scimath_unary!`). Each new Rust function gets a `#[pyfunction]` binding in `lib.rs`; Python wrappers collapse to thin `asarray()` + `_native.func()` delegations.

**Tech Stack:** Rust, `libm = "0.2"`, `ndarray`, RustPython (`#[pyfunction]`), existing project macros (`float_only_unary!`, `ensure_float`, `broadcast_array_data`)

**Spec:** `docs/superpowers/specs/2026-03-14-math-rust-migration-design.md`

---

## File Map

**Modified:**
- `crates/numpy-rust-core/Cargo.toml` — add `libm = "0.2"`
- `crates/numpy-rust-core/src/ops/math.rs` — all Group 1–5 Rust implementations
- `crates/numpy-rust-core/src/ops/numerical.rs` — `trapz`, `cumulative_trapezoid`
- `crates/numpy-rust-python/src/lib.rs` — ~35 new `#[pyfunction]` bindings
- `python/numpy/_math.py` — replace Python loops with thin wrappers
- `python/numpy/_reductions.py` — replace `trapz`, `cumulative_trapezoid` with thin wrappers
- `python/numpy/_stubs.py` — replace `_ScimathModule` methods with `_native.scimath_*` calls

**Created:**
- `tests/python/test_math_special.py` — tests for Group 1–4 libm functions
- `tests/python/test_scimath.py` — tests for Group 5 scimath complex-safe functions

---

## Key Patterns

### Existing macro (use for Group 1)
```rust
// In ops/math.rs — already exists, returns Result<NdArray>, errors on complex
float_only_unary!(name, f32_closure_or_fn, f64_closure_or_fn);
```

### New macro to add (Group 2 binary ops)
```rust
macro_rules! math_binary {
    ($name:ident, $f32_fn:expr, $f64_fn:expr) => {
        impl NdArray {
            pub fn $name(&self, other: &NdArray) -> Result<NdArray> {
                if self.dtype().is_complex() || other.dtype().is_complex() {
                    return Err(NumpyError::TypeError(
                        concat!(stringify!($name), " not supported for complex arrays").into(),
                    ));
                }
                let data_a = ensure_float(&self.data);
                let data_b = ensure_float(&other.data);
                let out_shape = broadcast_shape(self.shape(), other.shape())?;
                let data_a = broadcast_array_data(&data_a, &out_shape);
                let data_b = broadcast_array_data(&data_b, &out_shape);
                let result = match (data_a, data_b) {
                    (ArrayData::Float32(a), ArrayData::Float32(b)) => ArrayData::Float32(
                        ndarray::Zip::from(&a).and(&b)
                            .map_collect(|&x, &y| $f32_fn(x, y)).into_shared(),
                    ),
                    (ArrayData::Float64(a), ArrayData::Float64(b)) => ArrayData::Float64(
                        ndarray::Zip::from(&a).and(&b)
                            .map_collect(|&x, &y| $f64_fn(x, y)).into_shared(),
                    ),
                    _ => unreachable!(),
                };
                Ok(NdArray::from_data(result))
            }
        }
    };
}
```

### lib.rs patterns
```rust
// Infallible unary (returns NdArray, not Result):
#[pyfunction]
fn sinh(a: vm::PyRef<PyNdArray>, _vm: &VirtualMachine) -> PyNdArray {
    PyNdArray::from_core(a.inner().sinh())
}

// Fallible unary (returns Result<NdArray>):
#[pyfunction]
fn log1p(a: vm::PyRef<PyNdArray>, vm: &VirtualMachine) -> PyResult<PyNdArray> {
    a.inner()
        .log1p()
        .map(PyNdArray::from_core)
        .map_err(|e| vm.new_type_error(e.to_string()))
}

// Binary (uses PyObjectRef for broadcasting):
#[pyfunction]
fn copysign(x1: PyObjectRef, x2: PyObjectRef, vm: &VirtualMachine) -> PyResult<PyNdArray> {
    let a = obj_to_ndarray(&x1, vm)?;
    let b = obj_to_ndarray(&x2, vm)?;
    a.copysign(&b)
        .map(PyNdArray::from_core)
        .map_err(|e| vm.new_value_error(e.to_string()))
}
```

### Python wrapper pattern (thin delegation)
```python
def gamma(x):
    return _native.gamma(asarray(x))
```

### Test runner commands
```bash
cargo test --release -p numpy-rust-core                  # Rust unit tests
cargo build --release                                     # Build Python binary
./target/release/numpy-python tests/python/test_foo.py  # Python tests
./target/release/numpy-python tests/numpy_compat/run_compat.py --ci  # Compat suite
```

---

## Chunk 1: Setup + Group 1 Libm Unary Functions

Functions: `cbrt`, `gamma`, `lgamma`, `erf`, `erfc`, `j0`, `j1`, `y0`, `y1`

### Task 1: Add libm dependency

**Files:**
- Modify: `crates/numpy-rust-core/Cargo.toml`

- [ ] **Add libm to Cargo.toml**

```toml
# In [dependencies] section:
libm = "0.2"
```

- [ ] **Verify it compiles**

```bash
cargo check -p numpy-rust-core
```
Expected: no errors.

- [ ] **Commit**

```bash
git add crates/numpy-rust-core/Cargo.toml
git commit -m "feat(deps): add libm 0.2 for WASM-native math functions"
```

---

### Task 2: Add Group 1 libm unary functions to ops/math.rs

**Files:**
- Modify: `crates/numpy-rust-core/src/ops/math.rs`

The existing `float_only_unary!` macro already does what we need — it maps a float32 op and float64 op element-wise, errors on complex. Add `use libm;` at the top of the file (or use the full path `libm::cbrt`), then add macro invocations at the end of the unary section.

- [ ] **Write Rust unit tests first** (add to `#[cfg(test)]` block at end of math.rs)

```rust
#[cfg(test)]
mod tests {
    use super::*;
    use crate::array_data::ArrayData;

    fn arr(v: Vec<f64>) -> NdArray {
        NdArray::from_vec(v)
    }

    /// Extract f64 values from a Float64 NdArray for assertions.
    fn f64_vals(r: &NdArray) -> Vec<f64> {
        let ArrayData::Float64(a) = r.data() else { panic!("expected Float64, got {:?}", r.dtype()) };
        a.iter().copied().collect()
    }

    #[test]
    fn test_cbrt() {
        let a = arr(vec![8.0, -27.0, 0.0]);
        let r = a.cbrt().unwrap();
        let vals = f64_vals(&r);
        assert!((vals[0] - 2.0).abs() < 1e-10, "cbrt(8) = {}", vals[0]);
        assert!((vals[1] - (-3.0)).abs() < 1e-10, "cbrt(-27) = {}", vals[1]);
        assert_eq!(vals[2], 0.0);
    }

    #[test]
    fn test_gamma() {
        let a = arr(vec![1.0, 2.0, 5.0]);
        let r = a.gamma().unwrap();
        let vals = f64_vals(&r);
        assert!((vals[0] - 1.0).abs() < 1e-10); // gamma(1) = 1
        assert!((vals[1] - 1.0).abs() < 1e-10); // gamma(2) = 1
        assert!((vals[2] - 24.0).abs() < 1e-10); // gamma(5) = 24
    }

    #[test]
    fn test_erf() {
        let a = arr(vec![0.0, 1.0, -1.0]);
        let r = a.erf().unwrap();
        let vals = f64_vals(&r);
        assert_eq!(vals[0], 0.0);
        assert!((vals[1] - 0.8427007929).abs() < 1e-8);
        assert!((vals[2] + 0.8427007929).abs() < 1e-8);
    }

    #[test]
    fn test_j0() {
        let a = arr(vec![0.0, 1.0]);
        let r = a.j0().unwrap();
        let vals = f64_vals(&r);
        assert!((vals[0] - 1.0).abs() < 1e-10); // j0(0) = 1
        assert!((vals[1] - 0.7651976866).abs() < 1e-8);
    }
}

- [ ] **Run tests to verify they fail**

```bash
cargo test --release -p numpy-rust-core 2>&1 | grep -E "FAILED|error"
```
Expected: compile error — `cbrt`, `gamma`, etc. not found.

- [ ] **Implement: add `use libm;` and macro invocations in math.rs**

Add after the existing `float_only_unary!` invocations (around line 195):

```rust
// --- libm-backed unary functions ---
// These use float_only_unary! (no complex support).
// libm functions return NaN for out-of-domain inputs; no panics.
float_only_unary!(cbrt,   libm::cbrtf,   libm::cbrt);
float_only_unary!(gamma,  libm::tgammaf, libm::tgamma);
float_only_unary!(lgamma, libm::lgammaf, libm::lgamma);
float_only_unary!(erf,    libm::erff,    libm::erf);
float_only_unary!(erfc,   libm::erfcf,   libm::erfc);
float_only_unary!(j0,     libm::j0f,     libm::j0);
float_only_unary!(j1,     libm::j1f,     libm::j1);
float_only_unary!(y0,     libm::y0f,     libm::y0);
float_only_unary!(y1,     libm::y1f,     libm::y1);
```

- [ ] **Run Rust tests — verify they pass**

```bash
cargo test --release -p numpy-rust-core 2>&1 | grep -E "test_cbrt|test_gamma|test_erf|test_j0|FAILED|ok"
```
Expected: all new tests pass.

- [ ] **Commit**

```bash
git add crates/numpy-rust-core/src/ops/math.rs
git commit -m "feat(math): add libm-backed unary functions (cbrt, gamma, erf, Bessel)"
```

---

### Task 3: Expose Group 1 functions in lib.rs

**Files:**
- Modify: `crates/numpy-rust-python/src/lib.rs`

- [ ] **Add #[pyfunction] bindings** — add after the existing `signbit` and `logical_not` functions (around line 450):

```rust
    // --- libm-backed unary functions ---

    #[pyfunction]
    fn cbrt(a: vm::PyRef<PyNdArray>, vm: &VirtualMachine) -> PyResult<PyNdArray> {
        a.inner().cbrt()
            .map(PyNdArray::from_core)
            .map_err(|e| vm.new_value_error(e.to_string()))
    }

    #[pyfunction]
    fn gamma(a: vm::PyRef<PyNdArray>, vm: &VirtualMachine) -> PyResult<PyNdArray> {
        a.inner().gamma()
            .map(PyNdArray::from_core)
            .map_err(|e| vm.new_value_error(e.to_string()))
    }

    #[pyfunction]
    fn lgamma(a: vm::PyRef<PyNdArray>, vm: &VirtualMachine) -> PyResult<PyNdArray> {
        a.inner().lgamma()
            .map(PyNdArray::from_core)
            .map_err(|e| vm.new_value_error(e.to_string()))
    }

    #[pyfunction]
    fn erf(a: vm::PyRef<PyNdArray>, vm: &VirtualMachine) -> PyResult<PyNdArray> {
        a.inner().erf()
            .map(PyNdArray::from_core)
            .map_err(|e| vm.new_value_error(e.to_string()))
    }

    #[pyfunction]
    fn erfc(a: vm::PyRef<PyNdArray>, vm: &VirtualMachine) -> PyResult<PyNdArray> {
        a.inner().erfc()
            .map(PyNdArray::from_core)
            .map_err(|e| vm.new_value_error(e.to_string()))
    }

    #[pyfunction]
    fn j0(a: vm::PyRef<PyNdArray>, vm: &VirtualMachine) -> PyResult<PyNdArray> {
        a.inner().j0()
            .map(PyNdArray::from_core)
            .map_err(|e| vm.new_value_error(e.to_string()))
    }

    #[pyfunction]
    fn j1(a: vm::PyRef<PyNdArray>, vm: &VirtualMachine) -> PyResult<PyNdArray> {
        a.inner().j1()
            .map(PyNdArray::from_core)
            .map_err(|e| vm.new_value_error(e.to_string()))
    }

    #[pyfunction]
    fn y0(a: vm::PyRef<PyNdArray>, vm: &VirtualMachine) -> PyResult<PyNdArray> {
        a.inner().y0()
            .map(PyNdArray::from_core)
            .map_err(|e| vm.new_value_error(e.to_string()))
    }

    #[pyfunction]
    fn y1(a: vm::PyRef<PyNdArray>, vm: &VirtualMachine) -> PyResult<PyNdArray> {
        a.inner().y1()
            .map(PyNdArray::from_core)
            .map_err(|e| vm.new_value_error(e.to_string()))
    }
```

- [ ] **Build the binary**

```bash
cargo build --release 2>&1 | grep -E "error|warning\[" | head -20
```
Expected: clean build.

- [ ] **Smoke-check that new functions are registered on `_native`**

```bash
./target/release/numpy-python -c "import _numpy_native as n; print(n.cbrt, n.gamma, n.j0)"
```
Expected: prints function objects, no AttributeError.

- [ ] **Write Python tests** — create `tests/python/test_math_special.py`:

```python
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
check("erf(1)", float(np.erf(np.array([1.0])).flatten().tolist()[0]), 0.8427007929, tol=1e-7)

# erfc
check("erfc(0)", float(np.erfc(np.array([0.0])).flatten().tolist()[0]), 1.0)

# j0
check("j0(0)", float(np.j0(np.array([0.0])).flatten().tolist()[0]), 1.0)

# j1
check("j1(0)", float(np.j1(np.array([0.0])).flatten().tolist()[0]), 0.0)

# y0: y0(1) ≈ 0.0882569642
check("y0(1)", float(np.y0(np.array([1.0])).flatten().tolist()[0]), 0.0882569642, tol=1e-7)

# y1: y1(1) ≈ -0.7812128213
check("y1(1)", float(np.y1(np.array([1.0])).flatten().tolist()[0]), -0.7812128213, tol=1e-7)

print(f"test_math_special: {passed} passed, {failed} failed")
if failed:
    raise SystemExit(1)
```

- [ ] **Run the test**

```bash
./target/release/numpy-python tests/python/test_math_special.py
```
Expected: all pass (the old Python implementations still work at this point).

- [ ] **Replace Python implementations with thin wrappers** in `python/numpy/_math.py`:

Replace the following functions (find each by name, replace the whole body):

```python
def cbrt(x):
    """Return the element-wise cube root."""
    x = asarray(x)
    return _native.cbrt(x)

def gamma(x):
    """Gamma function."""
    return _native.gamma(asarray(x))

def lgamma(x):
    """Log of the absolute value of the gamma function."""
    return _native.lgamma(asarray(x))

def erf(x):
    """Error function."""
    return _native.erf(asarray(x))

def erfc(x):
    """Complementary error function: 1 - erf(x)."""
    return _native.erfc(asarray(x))

def j0(x):
    """Bessel function of the first kind, order 0."""
    return _native.j0(asarray(x))

def j1(x):
    """Bessel function of the first kind, order 1."""
    return _native.j1(asarray(x))

def y0(x):
    """Bessel function of the second kind, order 0."""
    return _native.y0(asarray(x))

def y1(x):
    """Bessel function of the second kind, order 1."""
    return _native.y1(asarray(x))
```

- [ ] **Run tests again to verify wrappers work**

```bash
./target/release/numpy-python tests/python/test_math_special.py
```
Expected: all pass.

- [ ] **Run compat suite to check for regressions**

```bash
./target/release/numpy-python tests/numpy_compat/run_compat.py --ci 2>&1 | tail -5
```
Expected: same pass/fail count as before (1207 passed, 3 xfails).

- [ ] **Commit**

```bash
git add crates/numpy-rust-python/src/lib.rs python/numpy/_math.py tests/python/test_math_special.py
git commit -m "feat(math): expose libm unary functions to Python, replace Python loops"
```

---

## Chunk 2: Group 2 — Binary Math Operations

Functions: `copysign`, `hypot`, `fmod`, `ldexp`, `nextafter`, `logaddexp`, `logaddexp2`, `maximum`, `minimum`, `fmax`, `fmin`

### Task 4: Add math_binary! macro and binary op implementations

**Files:**
- Modify: `crates/numpy-rust-core/src/ops/math.rs`

- [ ] **Write Rust unit tests first** (add to `#[cfg(test)]` block):

```rust
    #[test]
    fn test_copysign() {
        let a = arr(vec![1.0, -2.0, 3.0]);
        let b = arr(vec![-1.0, 1.0, -1.0]);
        let r = a.copysign(&b).unwrap();
        let vals = f64_vals(&r);
        assert_eq!(vals, vec![-1.0, 2.0, -3.0]);
    }

    #[test]
    fn test_hypot() {
        let a = arr(vec![3.0, 0.0]);
        let b = arr(vec![4.0, 5.0]);
        let r = a.hypot(&b).unwrap();
        let vals = f64_vals(&r);
        assert!((vals[0] - 5.0).abs() < 1e-10);
        assert!((vals[1] - 5.0).abs() < 1e-10);
    }

    #[test]
    fn test_maximum_nan_propagation() {
        let a = arr(vec![f64::NAN, 1.0, 3.0]);
        let b = arr(vec![1.0, f64::NAN, 2.0]);
        let r = a.maximum(&b).unwrap();
        let vals = f64_vals(&r);
        assert!(vals[0].is_nan()); // NAN propagates
        assert!(vals[1].is_nan()); // NAN propagates
        assert_eq!(vals[2], 3.0);
    }

    #[test]
    fn test_fmax_nan_ignoring() {
        let a = arr(vec![f64::NAN, 1.0, 3.0]);
        let b = arr(vec![1.0, f64::NAN, 2.0]);
        let r = a.fmax(&b).unwrap();
        let vals = f64_vals(&r);
        assert_eq!(vals[0], 1.0); // NaN ignored
        assert_eq!(vals[1], 1.0); // NaN ignored
        assert_eq!(vals[2], 3.0);
    }

    #[test]
    fn test_logaddexp() {
        // logaddexp(1, 2) = log(e^1 + e^2) ≈ 2.3132617
        let a = arr(vec![1.0]);
        let b = arr(vec![2.0]);
        let r = a.logaddexp(&b).unwrap();
        let vals = f64_vals(&r);
        let expected = (1.0_f64.exp() + 2.0_f64.exp()).ln();
        assert!((vals[0] - expected).abs() < 1e-10);
    }
```

- [ ] **Run to verify they fail**

```bash
cargo test --release -p numpy-rust-core 2>&1 | grep -E "error\[" | head -5
```
Expected: compile errors for missing methods.

- [ ] **Add `math_binary!` macro and Group 2 implementations to math.rs**

Add after the libm unary section:

```rust
// --- Binary float math macro ---
macro_rules! math_binary {
    ($name:ident, $f32_fn:expr, $f64_fn:expr) => {
        impl NdArray {
            pub fn $name(&self, other: &NdArray) -> Result<NdArray> {
                if self.dtype().is_complex() || other.dtype().is_complex() {
                    return Err(NumpyError::TypeError(
                        concat!(stringify!($name), " not supported for complex arrays").into(),
                    ));
                }
                let data_a = ensure_float(&self.data);
                let data_b = ensure_float(&other.data);
                // Unify dtypes: Float32 only if BOTH inputs are Float32; otherwise Float64.
                let out_dtype = if data_a.dtype() == DType::Float32 && data_b.dtype() == DType::Float32 {
                    DType::Float32
                } else {
                    DType::Float64
                };
                let data_a = cast_array_data(&data_a, out_dtype);
                let data_b = cast_array_data(&data_b, out_dtype);
                let out_shape = broadcast_shape(self.shape(), other.shape())?;
                let data_a = broadcast_array_data(&data_a, &out_shape);
                let data_b = broadcast_array_data(&data_b, &out_shape);
                let result = match (data_a, data_b) {
                    (ArrayData::Float32(a), ArrayData::Float32(b)) => ArrayData::Float32(
                        ndarray::Zip::from(&a)
                            .and(&b)
                            .map_collect(|&x, &y| $f32_fn(x, y))
                            .into_shared(),
                    ),
                    (ArrayData::Float64(a), ArrayData::Float64(b)) => ArrayData::Float64(
                        ndarray::Zip::from(&a)
                            .and(&b)
                            .map_collect(|&x, &y| $f64_fn(x, y))
                            .into_shared(),
                    ),
                    _ => unreachable!(),
                };
                Ok(NdArray::from_data(result))
            }
        }
    };
}

math_binary!(copysign,  libm::copysignf, libm::copysign);
math_binary!(hypot,     libm::hypotf,    libm::hypot);
math_binary!(fmod,      libm::fmodf,     libm::fmod);
math_binary!(nextafter, libm::nextafterf, libm::nextafter);
math_binary!(
    logaddexp,
    |a: f32, b: f32| {
        let mx = a.max(b);
        mx + (1.0_f32 + (-(a - b).abs()).exp()).ln()
    },
    |a: f64, b: f64| {
        let mx = a.max(b);
        mx + (1.0_f64 + (-(a - b).abs()).exp()).ln()
    }
);
math_binary!(
    logaddexp2,
    |a: f32, b: f32| {
        let mx = a.max(b);
        mx + (1.0_f32 + (-(a - b).abs()).exp2()).log2()
    },
    |a: f64, b: f64| {
        let mx = a.max(b);
        mx + (1.0_f64 + (-(a - b).abs()).exp2()).log2()
    }
);
// fmax/fmin: NaN-ignoring (f32::max / f64::max returns non-NaN operand)
math_binary!(fmax, f32::max, f64::max);
math_binary!(fmin, f32::min, f64::min);
// maximum/minimum: NaN-propagating
math_binary!(
    maximum,
    |a: f32, b: f32| if a.is_nan() || b.is_nan() { f32::NAN } else { a.max(b) },
    |a: f64, b: f64| if a.is_nan() || b.is_nan() { f64::NAN } else { a.max(b) }
);
math_binary!(
    minimum,
    |a: f32, b: f32| if a.is_nan() || b.is_nan() { f32::NAN } else { a.min(b) },
    |a: f64, b: f64| if a.is_nan() || b.is_nan() { f64::NAN } else { a.min(b) }
);
```

- [ ] **Add ldexp separately** (mixed types: float × int → float):

```rust
impl NdArray {
    /// ldexp(x, n) = x * 2^n, element-wise. n is cast to Int32.
    pub fn ldexp(&self, exp_arr: &NdArray) -> Result<NdArray> {
        if self.dtype().is_complex() {
            return Err(NumpyError::TypeError(
                "ldexp not supported for complex arrays".into(),
            ));
        }
        let data_a = ensure_float(&self.data);
        let exp_i32 = cast_array_data(&exp_arr.data, DType::Int32);
        let out_shape = broadcast_shape(self.shape(), exp_arr.shape())?;
        let data_a = broadcast_array_data(&data_a, &out_shape);
        let exp_i32 = broadcast_array_data(&exp_i32, &out_shape);
        let result = match (data_a, exp_i32) {
            (ArrayData::Float32(a), ArrayData::Int32(e)) => ArrayData::Float32(
                ndarray::Zip::from(&a)
                    .and(&e)
                    .map_collect(|&x, &n| libm::ldexpf(x, n))
                    .into_shared(),
            ),
            (ArrayData::Float64(a), ArrayData::Int32(e)) => ArrayData::Float64(
                ndarray::Zip::from(&a)
                    .and(&e)
                    .map_collect(|&x, &n| libm::ldexp(x, n))
                    .into_shared(),
            ),
            _ => unreachable!(),
        };
        Ok(NdArray::from_data(result))
    }
}
```

- [ ] **Run Rust tests — verify they pass**

```bash
cargo test --release -p numpy-rust-core 2>&1 | grep -E "test_copysign|test_hypot|test_maximum|test_fmax|test_logaddexp|FAILED|ok"
```
Expected: all new tests pass.

- [ ] **Commit**

```bash
git add crates/numpy-rust-core/src/ops/math.rs
git commit -m "feat(math): add math_binary! macro and binary ops (copysign, hypot, fmod, ldexp, logaddexp, maximum, fmax)"
```

---

### Task 5: Expose Group 2 functions in lib.rs + simplify Python wrappers

**Files:**
- Modify: `crates/numpy-rust-python/src/lib.rs`
- Modify: `python/numpy/_math.py`

- [ ] **Add #[pyfunction] bindings to lib.rs**

Add after the Group 1 bindings:

```rust
    // --- Binary math functions ---

    #[pyfunction]
    fn copysign(x1: PyObjectRef, x2: PyObjectRef, vm: &VirtualMachine) -> PyResult<PyNdArray> {
        let a = obj_to_ndarray(&x1, vm)?;
        let b = obj_to_ndarray(&x2, vm)?;
        a.copysign(&b).map(PyNdArray::from_core).map_err(|e| vm.new_value_error(e.to_string()))
    }

    #[pyfunction]
    fn hypot(x1: PyObjectRef, x2: PyObjectRef, vm: &VirtualMachine) -> PyResult<PyNdArray> {
        let a = obj_to_ndarray(&x1, vm)?;
        let b = obj_to_ndarray(&x2, vm)?;
        a.hypot(&b).map(PyNdArray::from_core).map_err(|e| vm.new_value_error(e.to_string()))
    }

    #[pyfunction]
    fn fmod(x1: PyObjectRef, x2: PyObjectRef, vm: &VirtualMachine) -> PyResult<PyNdArray> {
        let a = obj_to_ndarray(&x1, vm)?;
        let b = obj_to_ndarray(&x2, vm)?;
        a.fmod(&b).map(PyNdArray::from_core).map_err(|e| vm.new_value_error(e.to_string()))
    }

    #[pyfunction]
    fn ldexp(x1: PyObjectRef, x2: PyObjectRef, vm: &VirtualMachine) -> PyResult<PyNdArray> {
        let a = obj_to_ndarray(&x1, vm)?;
        let b = obj_to_ndarray(&x2, vm)?;
        a.ldexp(&b).map(PyNdArray::from_core).map_err(|e| vm.new_value_error(e.to_string()))
    }

    #[pyfunction]
    fn nextafter(x1: PyObjectRef, x2: PyObjectRef, vm: &VirtualMachine) -> PyResult<PyNdArray> {
        let a = obj_to_ndarray(&x1, vm)?;
        let b = obj_to_ndarray(&x2, vm)?;
        a.nextafter(&b).map(PyNdArray::from_core).map_err(|e| vm.new_value_error(e.to_string()))
    }

    #[pyfunction]
    fn logaddexp(x1: PyObjectRef, x2: PyObjectRef, vm: &VirtualMachine) -> PyResult<PyNdArray> {
        let a = obj_to_ndarray(&x1, vm)?;
        let b = obj_to_ndarray(&x2, vm)?;
        a.logaddexp(&b).map(PyNdArray::from_core).map_err(|e| vm.new_value_error(e.to_string()))
    }

    #[pyfunction]
    fn logaddexp2(x1: PyObjectRef, x2: PyObjectRef, vm: &VirtualMachine) -> PyResult<PyNdArray> {
        let a = obj_to_ndarray(&x1, vm)?;
        let b = obj_to_ndarray(&x2, vm)?;
        a.logaddexp2(&b).map(PyNdArray::from_core).map_err(|e| vm.new_value_error(e.to_string()))
    }

    #[pyfunction]
    fn maximum(x1: PyObjectRef, x2: PyObjectRef, vm: &VirtualMachine) -> PyResult<PyNdArray> {
        let a = obj_to_ndarray(&x1, vm)?;
        let b = obj_to_ndarray(&x2, vm)?;
        a.maximum(&b).map(PyNdArray::from_core).map_err(|e| vm.new_value_error(e.to_string()))
    }

    #[pyfunction]
    fn minimum(x1: PyObjectRef, x2: PyObjectRef, vm: &VirtualMachine) -> PyResult<PyNdArray> {
        let a = obj_to_ndarray(&x1, vm)?;
        let b = obj_to_ndarray(&x2, vm)?;
        a.minimum(&b).map(PyNdArray::from_core).map_err(|e| vm.new_value_error(e.to_string()))
    }

    #[pyfunction]
    fn fmax(x1: PyObjectRef, x2: PyObjectRef, vm: &VirtualMachine) -> PyResult<PyNdArray> {
        let a = obj_to_ndarray(&x1, vm)?;
        let b = obj_to_ndarray(&x2, vm)?;
        a.fmax(&b).map(PyNdArray::from_core).map_err(|e| vm.new_value_error(e.to_string()))
    }

    #[pyfunction]
    fn fmin(x1: PyObjectRef, x2: PyObjectRef, vm: &VirtualMachine) -> PyResult<PyNdArray> {
        let a = obj_to_ndarray(&x1, vm)?;
        let b = obj_to_ndarray(&x2, vm)?;
        a.fmin(&b).map(PyNdArray::from_core).map_err(|e| vm.new_value_error(e.to_string()))
    }
```

- [ ] **Build**

```bash
cargo build --release 2>&1 | grep "error" | head -10
```
Expected: clean build.

- [ ] **Add tests to `test_math_special.py`**

Add before the final `print` / `raise SystemExit`:

```python
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
```

- [ ] **Replace Python implementations in `_math.py`**

```python
def copysign(x1, x2):
    """Change the sign of x1 to that of x2, element-wise."""
    return _native.copysign(asarray(x1), asarray(x2))

def hypot(x1, x2):
    """Element-wise sqrt(x1**2 + x2**2)."""
    return _native.hypot(asarray(x1), asarray(x2))

def fmod(x1, x2):
    """Return the element-wise remainder of division (C-style)."""
    return _native.fmod(asarray(x1), asarray(x2))

def ldexp(x1, x2):
    """Return x1 * 2**x2, element-wise."""
    return _native.ldexp(asarray(x1), asarray(x2))

def nextafter(x1, x2):
    """Return the next floating-point value after x1 towards x2, element-wise."""
    return _native.nextafter(asarray(x1), asarray(x2))

def logaddexp(x1, x2):
    """Logarithm of the sum of exponentiations of the inputs."""
    return _native.logaddexp(asarray(x1), asarray(x2))

def logaddexp2(x1, x2):
    """Logarithm base 2 of the sum of exponentiations of the inputs in base 2."""
    return _native.logaddexp2(asarray(x1), asarray(x2))

def maximum(x1, x2):
    return _native.maximum(asarray(x1), asarray(x2))

def minimum(x1, x2):
    return _native.minimum(asarray(x1), asarray(x2))

def fmax(x1, x2):
    """Element-wise maximum, ignoring NaNs."""
    return _native.fmax(asarray(x1), asarray(x2))

def fmin(x1, x2):
    """Element-wise minimum, ignoring NaNs."""
    return _native.fmin(asarray(x1), asarray(x2))
```

- [ ] **Run tests + compat**

```bash
./target/release/numpy-python tests/python/test_math_special.py
./target/release/numpy-python tests/numpy_compat/run_compat.py --ci 2>&1 | tail -5
```
Expected: all tests pass, compat unchanged.

- [ ] **Commit**

```bash
git add crates/numpy-rust-python/src/lib.rs python/numpy/_math.py tests/python/test_math_special.py
git commit -m "feat(math): expose binary math ops to Python (copysign, hypot, fmod, ldexp, maximum, fmax, logaddexp)"
```

---

## Chunk 3: Group 3+4 — Special-Return and Multi-Param Functions

Functions: `frexp`, `modf`, `nan_to_num`, `spacing`, `i0`

### Task 6: frexp and modf (tuple returns)

**Files:**
- Modify: `crates/numpy-rust-core/src/ops/math.rs`
- Modify: `crates/numpy-rust-python/src/lib.rs`
- Modify: `python/numpy/_math.py`

- [ ] **Write Rust tests**

```rust
    #[test]
    fn test_frexp() {
        let a = arr(vec![12.0, 0.5]);
        let (m, e) = a.frexp().unwrap();
        let mv = f64_vals(&m);
        // Exponent array is Int32; extract values via ArrayData pattern match
        let ArrayData::Int32(e_arr) = e.data() else { panic!("expected Int32") };
        let ev: Vec<i32> = e_arr.iter().copied().collect();
        assert!((mv[0] - 0.75).abs() < 1e-10); // 12 = 0.75 * 2^4
        assert_eq!(ev[0], 4_i32);
        assert!((mv[1] - 0.5).abs() < 1e-10); // 0.5 = 0.5 * 2^0
        assert_eq!(ev[1], 0_i32);
    }

    #[test]
    fn test_modf() {
        let a = arr(vec![3.7, -2.5, 0.0]);
        let (frac, intg) = a.modf().unwrap();
        let fv = f64_vals(&frac);
        let iv = f64_vals(&intg);
        assert!((fv[0] - 0.7).abs() < 1e-10);
        assert_eq!(iv[0], 3.0);
        assert!((fv[1] - (-0.5)).abs() < 1e-10);
        assert_eq!(iv[1], -2.0);
    }
```

- [ ] **Implement frexp in math.rs**

```rust
impl NdArray {
    /// Decomposes each element into mantissa and base-2 exponent.
    /// Returns (mantissa: Float64, exponent: Int32), both same shape as input.
    pub fn frexp(&self) -> Result<(NdArray, NdArray)> {
        if self.dtype().is_complex() {
            return Err(NumpyError::TypeError(
                "frexp not supported for complex arrays".into(),
            ));
        }
        let data = ensure_float(&self.data);
        let shape: Vec<usize> = self.shape().to_vec();
        match data {
            ArrayData::Float64(a) => {
                let mut mantissas = Vec::with_capacity(a.len());
                let mut exponents = Vec::with_capacity(a.len());
                for &x in a.iter() {
                    let (m, e) = libm::frexp(x);
                    mantissas.push(m);
                    exponents.push(e as i32);
                }
                let m = ndarray::ArrayD::from_shape_vec(shape.clone(), mantissas)
                    .unwrap()
                    .into_shared();
                let e = ndarray::ArrayD::from_shape_vec(shape, exponents)
                    .unwrap()
                    .into_shared();
                Ok((
                    NdArray::from_data(ArrayData::Float64(m)),
                    NdArray::from_data(ArrayData::Int32(e)),
                ))
            }
            ArrayData::Float32(a) => {
                let mut mantissas = Vec::with_capacity(a.len());
                let mut exponents = Vec::with_capacity(a.len());
                for &x in a.iter() {
                    let (m, e) = libm::frexpf(x);
                    mantissas.push(m);
                    exponents.push(e as i32);
                }
                let m = ndarray::ArrayD::from_shape_vec(shape.clone(), mantissas)
                    .unwrap()
                    .into_shared();
                let e = ndarray::ArrayD::from_shape_vec(shape, exponents)
                    .unwrap()
                    .into_shared();
                Ok((
                    NdArray::from_data(ArrayData::Float32(m)),
                    NdArray::from_data(ArrayData::Int32(e)),
                ))
            }
            _ => unreachable!(),
        }
    }

    /// Splits each element into fractional and integer parts.
    /// Returns (fractional, integer), both same dtype as input (float).
    pub fn modf(&self) -> Result<(NdArray, NdArray)> {
        if self.dtype().is_complex() {
            return Err(NumpyError::TypeError(
                "modf not supported for complex arrays".into(),
            ));
        }
        let data = ensure_float(&self.data);
        let shape: Vec<usize> = self.shape().to_vec();
        match data {
            ArrayData::Float64(a) => {
                let mut fracs = Vec::with_capacity(a.len());
                let mut ints = Vec::with_capacity(a.len());
                for &x in a.iter() {
                    let (frac, int_part) = libm::modf(x);
                    fracs.push(frac);
                    ints.push(int_part);
                }
                let f = ndarray::ArrayD::from_shape_vec(shape.clone(), fracs)
                    .unwrap()
                    .into_shared();
                let i = ndarray::ArrayD::from_shape_vec(shape, ints)
                    .unwrap()
                    .into_shared();
                Ok((
                    NdArray::from_data(ArrayData::Float64(f)),
                    NdArray::from_data(ArrayData::Float64(i)),
                ))
            }
            ArrayData::Float32(a) => {
                let mut fracs = Vec::with_capacity(a.len());
                let mut ints = Vec::with_capacity(a.len());
                for &x in a.iter() {
                    let (frac, int_part) = libm::modff(x);
                    fracs.push(frac);
                    ints.push(int_part);
                }
                let f = ndarray::ArrayD::from_shape_vec(shape.clone(), fracs)
                    .unwrap()
                    .into_shared();
                let i = ndarray::ArrayD::from_shape_vec(shape, ints)
                    .unwrap()
                    .into_shared();
                Ok((
                    NdArray::from_data(ArrayData::Float32(f)),
                    NdArray::from_data(ArrayData::Float32(i)),
                ))
            }
            _ => unreachable!(),
        }
    }
}
```

- [ ] **Run Rust tests**

```bash
cargo test --release -p numpy-rust-core 2>&1 | grep -E "test_frexp|test_modf|FAILED"
```

- [ ] **Add lib.rs bindings for frexp and modf**

```rust
    #[pyfunction]
    fn frexp(
        a: vm::PyRef<PyNdArray>,
        vm: &VirtualMachine,
    ) -> PyResult<(PyNdArray, PyNdArray)> {
        let (mantissa, exponent) = a
            .inner()
            .frexp()
            .map_err(|e| vm.new_value_error(e.to_string()))?;
        Ok((PyNdArray::from_core(mantissa), PyNdArray::from_core(exponent)))
    }

    #[pyfunction]
    fn modf(
        a: vm::PyRef<PyNdArray>,
        vm: &VirtualMachine,
    ) -> PyResult<(PyNdArray, PyNdArray)> {
        let (frac, int_part) = a
            .inner()
            .modf()
            .map_err(|e| vm.new_value_error(e.to_string()))?;
        Ok((PyNdArray::from_core(frac), PyNdArray::from_core(int_part)))
    }
```

- [ ] **Build**

```bash
cargo build --release 2>&1 | grep "error" | head -5
```

- [ ] **Replace Python implementations in `_math.py`**

```python
def frexp(x):
    """Decompose elements into mantissa and twos exponent."""
    if isinstance(x, (int, float)):
        return _math.frexp(float(x))
    return _native.frexp(asarray(x))

def modf(x):
    """Return the fractional and integral parts of an array, element-wise."""
    if isinstance(x, (int, float)):
        return _math.modf(float(x))
    return _native.modf(asarray(x))
```

- [ ] **Add to test_math_special.py and run**

```python
# frexp
m, e = np.frexp(np.array([12.0]))
check("frexp mantissa", float(m.flatten().tolist()[0]), 0.75)
check("frexp exponent", float(e.flatten().tolist()[0]), 4.0)

# modf
frac, intg = np.modf(np.array([3.7, -2.5]))
check("modf frac[0]", float(frac.flatten().tolist()[0]), 0.7)
check("modf int[0]", float(intg.flatten().tolist()[0]), 3.0)
```

```bash
./target/release/numpy-python tests/python/test_math_special.py
```

- [ ] **Commit**

```bash
git add crates/numpy-rust-core/src/ops/math.rs crates/numpy-rust-python/src/lib.rs python/numpy/_math.py tests/python/test_math_special.py
git commit -m "feat(math): add frexp and modf with tuple returns via libm"
```

---

### Task 7: nan_to_num, spacing, i0

**Files:**
- Modify: `crates/numpy-rust-core/src/ops/math.rs`
- Modify: `crates/numpy-rust-python/src/lib.rs`
- Modify: `python/numpy/_math.py`

- [ ] **Write Rust tests**

```rust
    #[test]
    fn test_nan_to_num() {
        let a = NdArray::from_data(ArrayData::Float64(
            ndarray::array![f64::NAN, f64::INFINITY, f64::NEG_INFINITY, 1.0]
                .into_dyn()
                .into_shared(),
        ));
        let r = a.nan_to_num(0.0, 1e308, -1e308);
        let vals = f64_vals(&r);
        assert_eq!(vals[0], 0.0);
        assert_eq!(vals[1], 1e308);
        assert_eq!(vals[2], -1e308);
        assert_eq!(vals[3], 1.0);
    }

    #[test]
    fn test_nan_to_num_integer_passthrough() {
        let a = NdArray::from_data(ArrayData::Int64(
            ndarray::array![1_i64, 2, 3].into_dyn().into_shared(),
        ));
        let r = a.nan_to_num(0.0, 1e308, -1e308);
        // Integer arrays must pass through unchanged
        assert!(matches!(r.data(), ArrayData::Int64(_)));
    }

    #[test]
    fn test_i0() {
        let a = arr(vec![0.0, 1.0]);
        let r = a.i0();
        let vals = f64_vals(&r);
        assert!((vals[0] - 1.0).abs() < 1e-10); // I0(0) = 1
        assert!((vals[1] - 1.2660658778).abs() < 1e-8); // I0(1) ≈ 1.2660658778
    }
```

- [ ] **Implement in math.rs**

```rust
impl NdArray {
    /// Replace NaN, +Inf, -Inf with finite values. Integer arrays pass through unchanged.
    /// nan: replacement for NaN (default 0.0)
    /// posinf: replacement for +Inf (default f64::MAX)
    /// neginf: replacement for -Inf (default -f64::MAX)
    pub fn nan_to_num(&self, nan: f64, posinf: f64, neginf: f64) -> NdArray {
        // Integer/bool arrays cannot have NaN or Inf — pass through unchanged
        match self.dtype() {
            crate::dtype::DType::Bool
            | crate::dtype::DType::Int32
            | crate::dtype::DType::Int64 => return self.deep_copy(),
            _ => {}
        }
        let data = ensure_float(&self.data);
        let result = match data {
            ArrayData::Float32(a) => ArrayData::Float32(
                a.mapv(|x| {
                    if x.is_nan() {
                        nan as f32
                    } else if x == f32::INFINITY {
                        posinf as f32
                    } else if x == f32::NEG_INFINITY {
                        neginf as f32
                    } else {
                        x
                    }
                })
                .into_shared(),
            ),
            ArrayData::Float64(a) => ArrayData::Float64(
                a.mapv(|x| {
                    if x.is_nan() {
                        nan
                    } else if x.is_infinite() && x > 0.0 {
                        posinf
                    } else if x.is_infinite() && x < 0.0 {
                        neginf
                    } else {
                        x
                    }
                })
                .into_shared(),
            ),
            _ => unreachable!(),
        };
        NdArray::from_data(result)
    }

    /// Distance between x and the nearest adjacent floating-point number.
    pub fn spacing(&self) -> Result<NdArray> {
        if self.dtype().is_complex() {
            return Err(NumpyError::TypeError(
                "spacing not supported for complex arrays".into(),
            ));
        }
        let data = ensure_float(&self.data);
        let result = match data {
            ArrayData::Float32(a) => ArrayData::Float32(
                a.mapv(|x| {
                    let ax = x.abs();
                    libm::nextafterf(ax, f32::INFINITY) - ax
                })
                .into_shared(),
            ),
            ArrayData::Float64(a) => ArrayData::Float64(
                a.mapv(|x| {
                    let ax = x.abs();
                    libm::nextafter(ax, f64::INFINITY) - ax
                })
                .into_shared(),
            ),
            _ => unreachable!(),
        };
        Ok(NdArray::from_data(result))
    }

    /// Modified Bessel function of the first kind, order 0.
    /// Uses series expansion: I0(x) = Σ ((x/2)^k / k!)^2
    pub fn i0(&self) -> NdArray {
        let data = ensure_float(&self.data);
        let result = match data {
            ArrayData::Float64(a) => ArrayData::Float64(
                a.mapv(|x| {
                    let mut val = 1.0_f64;
                    let mut term = 1.0_f64;
                    let h = x * 0.5;
                    for k in 1_u32..30 {
                        term *= (h * h) / (k * k) as f64;
                        val += term;
                        if term.abs() < 1e-15 * val.abs() {
                            break;
                        }
                    }
                    val
                })
                .into_shared(),
            ),
            ArrayData::Float32(a) => ArrayData::Float32(
                a.mapv(|x| {
                    let mut val = 1.0_f32;
                    let mut term = 1.0_f32;
                    let h = x * 0.5;
                    for k in 1_u32..25 {
                        term *= (h * h) / (k * k) as f32;
                        val += term;
                        if term.abs() < 1e-7_f32 * val.abs() {
                            break;
                        }
                    }
                    val
                })
                .into_shared(),
            ),
            _ => unreachable!(),
        };
        NdArray::from_data(result)
    }
}
```

- [ ] **Run Rust tests**

```bash
cargo test --release -p numpy-rust-core 2>&1 | grep -E "test_nan_to_num|test_i0|FAILED"
```

- [ ] **Add lib.rs bindings**

```rust
    #[pyfunction]
    fn nan_to_num(
        a: vm::PyRef<PyNdArray>,
        nan: f64,
        posinf: f64,
        neginf: f64,
        _vm: &VirtualMachine,
    ) -> PyNdArray {
        PyNdArray::from_core(a.inner().nan_to_num(nan, posinf, neginf))
    }

    #[pyfunction]
    fn spacing(a: vm::PyRef<PyNdArray>, vm: &VirtualMachine) -> PyResult<PyNdArray> {
        a.inner()
            .spacing()
            .map(PyNdArray::from_core)
            .map_err(|e| vm.new_value_error(e.to_string()))
    }

    #[pyfunction]
    fn i0(a: vm::PyRef<PyNdArray>, _vm: &VirtualMachine) -> PyNdArray {
        PyNdArray::from_core(a.inner().i0())
    }
```

- [ ] **Build**

```bash
cargo build --release 2>&1 | grep "error" | head -5
```

- [ ] **Replace Python implementations in `_math.py`**

```python
def nan_to_num(x, copy=True, nan=0.0, posinf=None, neginf=None):
    """Replace NaN with zero and infinity with large finite numbers."""
    x = asarray(x)
    posinf_val = posinf if posinf is not None else 1.7976931348623157e+308
    neginf_val = neginf if neginf is not None else -1.7976931348623157e+308
    return _native.nan_to_num(x, float(nan), float(posinf_val), float(neginf_val))

def spacing(x):
    """Return the distance between x and the nearest adjacent number."""
    if isinstance(x, (int, float)):
        ax = _math.fabs(float(x))
        return _math.nextafter(ax, _math.inf) - ax
    return _native.spacing(asarray(x))

def i0(x):
    """Modified Bessel function of the first kind, order 0."""
    return _native.i0(asarray(x))
```

- [ ] **Add tests and run**

```python
# nan_to_num
r = np.nan_to_num(np.array([float('nan'), float('inf'), float('-inf'), 1.0]))
vals = r.flatten().tolist()
check("nan_to_num nan", vals[0], 0.0)
check("nan_to_num posinf", 1 if vals[1] > 1e307 else 0, 1)
check("nan_to_num neginf", 1 if vals[2] < -1e307 else 0, 1)
check("nan_to_num normal", vals[3], 1.0)

# i0
check("i0(0)", float(np.i0(np.array([0.0])).flatten().tolist()[0]), 1.0)
check("i0(1)", float(np.i0(np.array([1.0])).flatten().tolist()[0]), 1.2660658778, tol=1e-7)
```

```bash
./target/release/numpy-python tests/python/test_math_special.py
./target/release/numpy-python tests/numpy_compat/run_compat.py --ci 2>&1 | tail -5
```

- [ ] **Commit**

```bash
git add crates/numpy-rust-core/src/ops/math.rs crates/numpy-rust-python/src/lib.rs python/numpy/_math.py tests/python/test_math_special.py
git commit -m "feat(math): add nan_to_num, spacing, i0 in Rust"
```

---

## Chunk 4: Group 5 — Scimath Complex-Safe Operations

Functions: `scimath_sqrt`, `scimath_log`, `scimath_log2`, `scimath_log10`, `scimath_arcsin`, `scimath_arccos`, `scimath_arctanh`, `scimath_power`

### Task 8: Scimath implementations in ops/math.rs

**Files:**
- Modify: `crates/numpy-rust-core/src/ops/math.rs`
- Modify: `crates/numpy-rust-python/src/lib.rs`
- Modify: `python/numpy/_stubs.py`
- Create: `tests/python/test_scimath.py`

These all share the same pattern: scan input for out-of-domain values; if any found, upcast entire array to Complex128 before computing.

- [ ] **Write Rust tests**

```rust
    #[test]
    fn test_scimath_sqrt_positive() {
        let a = arr(vec![4.0, 9.0]);
        let r = a.scimath_sqrt();
        // All positive: result should be real Float64
        assert!(matches!(r.data(), ArrayData::Float64(_)));
        let vals = f64_vals(&r);
        assert!((vals[0] - 2.0).abs() < 1e-10);
    }

    #[test]
    fn test_scimath_sqrt_negative() {
        let a = arr(vec![-4.0, 9.0]);
        let r = a.scimath_sqrt();
        // Has negative: result must be Complex128
        assert!(matches!(r.data(), ArrayData::Complex128(_)));
        // sqrt(-4) = 2i
        if let ArrayData::Complex128(arr) = r.data() {
            let v: Vec<_> = arr.iter().collect();
            assert!((v[0].im - 2.0).abs() < 1e-10, "sqrt(-4).im = {}", v[0].im);
        }
    }

    #[test]
    fn test_scimath_log_negative() {
        let a = arr(vec![-1.0]);
        let r = a.scimath_log();
        // log(-1) = iπ
        assert!(matches!(r.data(), ArrayData::Complex128(_)));
        if let ArrayData::Complex128(arr) = r.data() {
            let v = arr.iter().next().unwrap();
            assert!((v.re - 0.0).abs() < 1e-10);
            assert!((v.im - std::f64::consts::PI).abs() < 1e-10);
        }
    }
```

- [ ] **Implement scimath functions in math.rs**

Add a helper first, then the functions:

```rust
use num_complex::Complex;

/// Helper: scan Float32/Float64 array for out-of-domain values per predicate.
/// If any found, casts to Complex128; otherwise returns the float data unchanged.
fn maybe_complex(data: &ArrayData, is_out_of_domain: impl Fn(f64) -> bool) -> ArrayData {
    let should_complex = match data {
        ArrayData::Float32(a) => a.iter().any(|&x| is_out_of_domain(x as f64)),
        ArrayData::Float64(a) => a.iter().any(|&x| is_out_of_domain(x)),
        _ => false,
    };
    if should_complex {
        cast_array_data(data, DType::Complex128)
    } else {
        data.clone()
    }
}

impl NdArray {
    pub fn scimath_sqrt(&self) -> NdArray {
        let data = ensure_float(&self.data);
        let data = maybe_complex(&data, |x| x < 0.0);
        let result = match data {
            ArrayData::Float32(a) => ArrayData::Float32(a.mapv(|x| x.sqrt()).into_shared()),
            ArrayData::Float64(a) => ArrayData::Float64(a.mapv(|x| x.sqrt()).into_shared()),
            ArrayData::Complex128(a) => ArrayData::Complex128(a.mapv(|z| z.sqrt()).into_shared()),
            _ => unreachable!(),
        };
        NdArray::from_data(result)
    }

    pub fn scimath_log(&self) -> NdArray {
        let data = ensure_float(&self.data);
        let data = maybe_complex(&data, |x| x < 0.0);
        let result = match data {
            ArrayData::Float32(a) => ArrayData::Float32(a.mapv(|x| x.ln()).into_shared()),
            ArrayData::Float64(a) => ArrayData::Float64(a.mapv(|x| x.ln()).into_shared()),
            ArrayData::Complex128(a) => ArrayData::Complex128(a.mapv(|z| z.ln()).into_shared()),
            _ => unreachable!(),
        };
        NdArray::from_data(result)
    }

    pub fn scimath_log2(&self) -> NdArray {
        let data = ensure_float(&self.data);
        let data = maybe_complex(&data, |x| x < 0.0);
        let ln2 = std::f64::consts::LN_2;
        let result = match data {
            ArrayData::Float32(a) => ArrayData::Float32(a.mapv(|x| x.log2()).into_shared()),
            ArrayData::Float64(a) => ArrayData::Float64(a.mapv(|x| x.log2()).into_shared()),
            ArrayData::Complex128(a) => ArrayData::Complex128(
                a.mapv(|z| z.ln() / Complex::new(ln2, 0.0)).into_shared(),
            ),
            _ => unreachable!(),
        };
        NdArray::from_data(result)
    }

    pub fn scimath_log10(&self) -> NdArray {
        let data = ensure_float(&self.data);
        let data = maybe_complex(&data, |x| x < 0.0);
        let ln10 = std::f64::consts::LN_10;
        let result = match data {
            ArrayData::Float32(a) => ArrayData::Float32(a.mapv(|x| x.log10()).into_shared()),
            ArrayData::Float64(a) => ArrayData::Float64(a.mapv(|x| x.log10()).into_shared()),
            ArrayData::Complex128(a) => ArrayData::Complex128(
                a.mapv(|z| z.ln() / Complex::new(ln10, 0.0)).into_shared(),
            ),
            _ => unreachable!(),
        };
        NdArray::from_data(result)
    }

    pub fn scimath_arcsin(&self) -> NdArray {
        let data = ensure_float(&self.data);
        let data = maybe_complex(&data, |x| x.abs() > 1.0);
        let result = match data {
            ArrayData::Float32(a) => ArrayData::Float32(a.mapv(|x| x.asin()).into_shared()),
            ArrayData::Float64(a) => ArrayData::Float64(a.mapv(|x| x.asin()).into_shared()),
            ArrayData::Complex128(a) => ArrayData::Complex128(a.mapv(|z| z.asin()).into_shared()),
            _ => unreachable!(),
        };
        NdArray::from_data(result)
    }

    pub fn scimath_arccos(&self) -> NdArray {
        let data = ensure_float(&self.data);
        let data = maybe_complex(&data, |x| x.abs() > 1.0);
        let result = match data {
            ArrayData::Float32(a) => ArrayData::Float32(a.mapv(|x| x.acos()).into_shared()),
            ArrayData::Float64(a) => ArrayData::Float64(a.mapv(|x| x.acos()).into_shared()),
            ArrayData::Complex128(a) => ArrayData::Complex128(a.mapv(|z| z.acos()).into_shared()),
            _ => unreachable!(),
        };
        NdArray::from_data(result)
    }

    pub fn scimath_arctanh(&self) -> NdArray {
        let data = ensure_float(&self.data);
        let data = maybe_complex(&data, |x| x.abs() > 1.0);
        let result = match data {
            ArrayData::Float32(a) => ArrayData::Float32(a.mapv(|x| x.atanh()).into_shared()),
            ArrayData::Float64(a) => ArrayData::Float64(a.mapv(|x| x.atanh()).into_shared()),
            ArrayData::Complex128(a) => ArrayData::Complex128(a.mapv(|z| z.atanh()).into_shared()),
            _ => unreachable!(),
        };
        NdArray::from_data(result)
    }

    /// Complex-safe power: negative base → complex via exp(p * ln(x)).
    pub fn scimath_power(&self, exp_arr: &NdArray) -> Result<NdArray> {
        let data_a = ensure_float(&self.data);
        let data_e = ensure_float(&exp_arr.data);
        let out_shape = broadcast_shape(self.shape(), exp_arr.shape())?;
        let data_a = broadcast_array_data(&data_a, &out_shape);
        let data_e = broadcast_array_data(&data_e, &out_shape);
        // Check if any base is negative (needs complex)
        let needs_complex = match &data_a {
            ArrayData::Float32(a) => a.iter().any(|&x| x < 0.0),
            ArrayData::Float64(a) => a.iter().any(|&x| x < 0.0),
            _ => false,
        };
        if needs_complex {
            let c_a = cast_array_data(&data_a, DType::Complex128);
            let c_e = cast_array_data(&data_e, DType::Complex128);
            if let (ArrayData::Complex128(a), ArrayData::Complex128(e)) = (c_a, c_e) {
                return Ok(NdArray::from_data(ArrayData::Complex128(
                    ndarray::Zip::from(&a)
                        .and(&e)
                        .map_collect(|&b, &p| b.powc(p))
                        .into_shared(),
                )));
            }
        }
        // Normal float power
        let result = match (data_a, data_e) {
            (ArrayData::Float32(a), ArrayData::Float32(e)) => ArrayData::Float32(
                ndarray::Zip::from(&a).and(&e).map_collect(|&x, &p| x.powf(p)).into_shared(),
            ),
            (ArrayData::Float64(a), ArrayData::Float64(e)) => ArrayData::Float64(
                ndarray::Zip::from(&a).and(&e).map_collect(|&x, &p| x.powf(p)).into_shared(),
            ),
            _ => unreachable!(),
        };
        Ok(NdArray::from_data(result))
    }
}
```

- [ ] **Run Rust tests**

```bash
cargo test --release -p numpy-rust-core 2>&1 | grep -E "test_scimath|FAILED"
```

- [ ] **Add lib.rs bindings**

```rust
    // --- Scimath complex-safe functions ---

    #[pyfunction]
    fn scimath_sqrt(a: vm::PyRef<PyNdArray>, _vm: &VirtualMachine) -> PyNdArray {
        PyNdArray::from_core(a.inner().scimath_sqrt())
    }

    #[pyfunction]
    fn scimath_log(a: vm::PyRef<PyNdArray>, _vm: &VirtualMachine) -> PyNdArray {
        PyNdArray::from_core(a.inner().scimath_log())
    }

    #[pyfunction]
    fn scimath_log2(a: vm::PyRef<PyNdArray>, _vm: &VirtualMachine) -> PyNdArray {
        PyNdArray::from_core(a.inner().scimath_log2())
    }

    #[pyfunction]
    fn scimath_log10(a: vm::PyRef<PyNdArray>, _vm: &VirtualMachine) -> PyNdArray {
        PyNdArray::from_core(a.inner().scimath_log10())
    }

    #[pyfunction]
    fn scimath_arcsin(a: vm::PyRef<PyNdArray>, _vm: &VirtualMachine) -> PyNdArray {
        PyNdArray::from_core(a.inner().scimath_arcsin())
    }

    #[pyfunction]
    fn scimath_arccos(a: vm::PyRef<PyNdArray>, _vm: &VirtualMachine) -> PyNdArray {
        PyNdArray::from_core(a.inner().scimath_arccos())
    }

    #[pyfunction]
    fn scimath_arctanh(a: vm::PyRef<PyNdArray>, _vm: &VirtualMachine) -> PyNdArray {
        PyNdArray::from_core(a.inner().scimath_arctanh())
    }

    #[pyfunction]
    fn scimath_power(x1: PyObjectRef, x2: PyObjectRef, vm: &VirtualMachine) -> PyResult<PyNdArray> {
        let a = obj_to_ndarray(&x1, vm)?;
        let b = obj_to_ndarray(&x2, vm)?;
        a.scimath_power(&b)
            .map(PyNdArray::from_core)
            .map_err(|e| vm.new_value_error(e.to_string()))
    }
```

- [ ] **Build**

```bash
cargo build --release 2>&1 | grep "error" | head -5
```

- [ ] **Create `tests/python/test_scimath.py`**

```python
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

sm = np.lib.scimath

# sqrt of positive: real result
r = sm.sqrt(np.array([4.0, 9.0]))
check("sqrt(4)", float(r.flatten().tolist()[0]), 2.0)

# sqrt of negative: complex result
r = sm.sqrt(np.array([-4.0]))
v = r.flatten().tolist()[0]
assert hasattr(v, 'imag') or isinstance(v, complex), f"sqrt(-4) should be complex, got {v!r}"
check("sqrt(-4).imag", abs(v.imag if hasattr(v, 'imag') else v.imag), 2.0)
passed += 1

# log of negative: complex
r = sm.log(np.array([-1.0]))
v = r.flatten().tolist()[0]
import math as _m
check("log(-1).imag", abs(v.imag if hasattr(v, 'imag') else 0), _m.pi)

# arcsin out-of-domain: complex
r = sm.arcsin(np.array([2.0]))
v = r.flatten().tolist()[0]
assert hasattr(v, 'imag') or isinstance(v, complex), f"arcsin(2) should be complex"
passed += 1

# arccos out-of-domain: complex
r = sm.arccos(np.array([2.0]))
v = r.flatten().tolist()[0]
assert hasattr(v, 'imag') or isinstance(v, complex), f"arccos(2) should be complex"
passed += 1

# power with negative base
r = sm.power(np.array([-1.0]), np.array([0.5]))
v = r.flatten().tolist()[0]
assert hasattr(v, 'imag') or isinstance(v, complex), f"(-1)^0.5 should be complex"
passed += 1

print(f"test_scimath: {passed} passed, {failed} failed")
if failed:
    raise SystemExit(1)
```

- [ ] **Replace `_ScimathModule` in `_stubs.py`**

Find the `class _ScimathModule:` definition and replace its methods:

```python
class _ScimathModule:
    """Complex-safe math functions (numpy.lib.scimath)."""

    @staticmethod
    def sqrt(x):
        from ._creation import asarray
        return _native.scimath_sqrt(asarray(x))

    @staticmethod
    def log(x):
        from ._creation import asarray
        return _native.scimath_log(asarray(x))

    @staticmethod
    def log2(x):
        from ._creation import asarray
        return _native.scimath_log2(asarray(x))

    @staticmethod
    def log10(x):
        from ._creation import asarray
        return _native.scimath_log10(asarray(x))

    @staticmethod
    def arcsin(x):
        from ._creation import asarray
        return _native.scimath_arcsin(asarray(x))

    @staticmethod
    def arccos(x):
        from ._creation import asarray
        return _native.scimath_arccos(asarray(x))

    @staticmethod
    def arctanh(x):
        from ._creation import asarray
        return _native.scimath_arctanh(asarray(x))

    @staticmethod
    def power(x, p):
        from ._creation import asarray
        return _native.scimath_power(asarray(x), asarray(p))
```

- [ ] **Run scimath tests + compat**

```bash
./target/release/numpy-python tests/python/test_scimath.py
./target/release/numpy-python tests/numpy_compat/run_compat.py --ci 2>&1 | tail -5
```

- [ ] **Commit**

```bash
git add crates/numpy-rust-core/src/ops/math.rs crates/numpy-rust-python/src/lib.rs python/numpy/_stubs.py tests/python/test_scimath.py
git commit -m "feat(math): add scimath complex-safe ops in Rust (sqrt, log, arcsin, arccos, arctanh, power)"
```

---

## Chunk 5: Group 6 — Reductions + Full Suite

Functions: `trapz`, `cumulative_trapezoid` (new Rust). `gradient`: Python wrapper cleanup only (already in Rust).

### Task 9: trapz and cumulative_trapezoid

**Files:**
- Modify: `crates/numpy-rust-core/src/ops/numerical.rs`
- Modify: `crates/numpy-rust-python/src/lib.rs`
- Modify: `python/numpy/_reductions.py`

- [ ] **Write Rust tests**

```rust
// At end of numerical.rs:
#[cfg(test)]
mod tests {
    use super::*;
    use crate::array_data::ArrayData;
    use crate::creation::from_vec;

    fn f64_vals(r: &crate::NdArray) -> Vec<f64> {
        let ArrayData::Float64(a) = r.data() else { panic!("expected Float64") };
        a.iter().copied().collect()
    }

    #[test]
    fn test_trapz_basic() {
        // trapz([1, 2, 3], dx=1) = 4.0
        let y = from_vec(vec![1.0_f64, 2.0, 3.0]);
        let r = trapz(&y, None, 1.0, None).unwrap();
        let vals = f64_vals(&r);
        assert!((vals[0] - 4.0).abs() < 1e-10, "trapz = {}", vals[0]);
    }

    #[test]
    fn test_trapz_with_x() {
        // trapz([1,2,3], x=[0,1,3]) = 0.5*(1+2)*1 + 0.5*(2+3)*2 = 1.5 + 5 = 6.5
        let y = from_vec(vec![1.0_f64, 2.0, 3.0]);
        let x = from_vec(vec![0.0_f64, 1.0, 3.0]);
        let r = trapz(&y, Some(&x), 1.0, None).unwrap();
        let vals = f64_vals(&r);
        assert!((vals[0] - 6.5).abs() < 1e-10, "trapz = {}", vals[0]);
    }

    #[test]
    fn test_cumulative_trapezoid() {
        // cumtrapz([1,2,3], dx=1) = [1.5, 4.0]
        let y = from_vec(vec![1.0_f64, 2.0, 3.0]);
        let r = cumulative_trapezoid(&y, None, 1.0, None).unwrap();
        let vals = f64_vals(&r);
        assert_eq!(vals.len(), 2);
        assert!((vals[0] - 1.5).abs() < 1e-10);
        assert!((vals[1] - 4.0).abs() < 1e-10);
    }
}
```

**Note:** Scalar results from `trapz` are returned as a 1-element Float64 array; use `f64_vals(&r)[0]` to extract the value.

- [ ] **Implement in numerical.rs**

```rust
/// Trapezoidal numerical integration along axis.
/// y: values array
/// x: optional x-coordinates (if None, uses uniform spacing dx)
/// dx: spacing when x is None (default 1.0)
/// axis: integration axis (default last axis, -1)
pub fn trapz(y: &NdArray, x: Option<&NdArray>, dx: f64, axis: Option<i64>) -> Result<NdArray> {
    use crate::array_data::ArrayData;
    use crate::casting::cast_array_data;
    use crate::dtype::DType;

    let ndim = y.ndim() as i64;
    let axis_idx = match axis {
        Some(a) if a < 0 => (ndim + a) as usize,
        Some(a) => a as usize,
        None => if ndim > 0 { (ndim - 1) as usize } else { 0 },
    };

    let y_f = y.astype(DType::Float64);
    let ArrayData::Float64(y_arr) = &y_f.data else { unreachable!() };

    let n = y.shape()[axis_idx];
    if n < 2 {
        return Err(NumpyError::ValueError(
            "trapz requires at least 2 elements along integration axis".into(),
        ));
    }

    // Build dx array along axis_idx
    let dx_vals: Vec<f64> = if let Some(x_arr) = x {
        let x_f = x_arr.astype(DType::Float64);
        let ArrayData::Float64(xa) = &x_f.data else { unreachable!() };
        // dx[i] = x[i+1] - x[i]  for i in 0..n-1
        let x_flat: Vec<f64> = xa.iter().copied().collect();
        x_flat.windows(2).map(|w| w[1] - w[0]).collect()
    } else {
        vec![dx; n - 1]
    };

    // For each slice along axis_idx: sum 0.5 * (y[i] + y[i+1]) * dx[i]
    // For 1-D case:
    if y.ndim() == 1 {
        let y_flat: Vec<f64> = y_arr.iter().copied().collect();
        let result: f64 = y_flat.windows(2)
            .zip(dx_vals.iter())
            .map(|(w, &d)| 0.5 * (w[0] + w[1]) * d)
            .sum();
        return Ok(NdArray::from_data(ArrayData::Float64(
            ndarray::ArrayD::from_elem(ndarray::IxDyn(&[]), result).into_shared(),
        )));
    }

    // Multi-dimensional: reduce along axis_idx using index_axis operations
    // Use ndarray's axis_iter to iterate slices perpendicular to axis_idx
    let shape = y.shape().to_vec();
    let mut out_shape = shape.clone();
    out_shape.remove(axis_idx);
    let out_n: usize = out_shape.iter().product();
    let mut result_flat = vec![0.0_f64; out_n];

    // Iterate over all positions in the output, accumulate trapz
    for (out_idx, out_val) in result_flat.iter_mut().enumerate() {
        // Map out_idx to multi-index in output shape
        let mut rem = out_idx;
        let mut in_idx_base = vec![0usize; y.ndim()];
        let mut ax_out = out_shape.len();
        for d in (0..y.ndim()).rev() {
            if d == axis_idx {
                continue;
            }
            ax_out -= 1;
            let dim = out_shape[ax_out];
            in_idx_base[d] = rem % dim;
            rem /= dim;
        }
        let mut sum = 0.0;
        for i in 0..n - 1 {
            in_idx_base[axis_idx] = i;
            let v0 = y_arr[ndarray::IxDyn(&in_idx_base)];
            in_idx_base[axis_idx] = i + 1;
            let v1 = y_arr[ndarray::IxDyn(&in_idx_base)];
            sum += 0.5 * (v0 + v1) * dx_vals[i];
        }
        *out_val = sum;
    }

    let result = ndarray::ArrayD::from_shape_vec(out_shape, result_flat)
        .map_err(|e| NumpyError::ValueError(e.to_string()))?
        .into_shared();
    Ok(NdArray::from_data(ArrayData::Float64(result)))
}

/// Cumulative trapezoidal integration. Returns array with length n-1 along axis.
pub fn cumulative_trapezoid(y: &NdArray, x: Option<&NdArray>, dx: f64, axis: Option<i64>) -> Result<NdArray> {
    use crate::array_data::ArrayData;
    use crate::dtype::DType;

    let ndim = y.ndim() as i64;
    let axis_idx = match axis {
        Some(a) if a < 0 => (ndim + a) as usize,
        Some(a) => a as usize,
        None => if ndim > 0 { (ndim - 1) as usize } else { 0 },
    };

    let y_f = y.astype(DType::Float64);
    let ArrayData::Float64(y_arr) = &y_f.data else { unreachable!() };
    let n = y.shape()[axis_idx];
    if n < 2 {
        return Err(NumpyError::ValueError(
            "cumulative_trapezoid requires at least 2 elements".into(),
        ));
    }

    let dx_vals: Vec<f64> = if let Some(x_arr) = x {
        let x_f = x_arr.astype(DType::Float64);
        let ArrayData::Float64(xa) = &x_f.data else { unreachable!() };
        let x_flat: Vec<f64> = xa.iter().copied().collect();
        x_flat.windows(2).map(|w| w[1] - w[0]).collect()
    } else {
        vec![dx; n - 1]
    };

    // Output shape: same as input but axis_idx dimension = n-1
    let mut out_shape = y.shape().to_vec();
    out_shape[axis_idx] = n - 1;
    let out_n: usize = out_shape.iter().product();
    let mut result_flat = vec![0.0_f64; out_n];

    // For 1-D:
    if y.ndim() == 1 {
        let y_flat: Vec<f64> = y_arr.iter().copied().collect();
        let mut cumsum = 0.0;
        for (i, (&d, w)) in dx_vals.iter().zip(y_flat.windows(2)).enumerate() {
            cumsum += 0.5 * (w[0] + w[1]) * d;
            result_flat[i] = cumsum;
        }
        let result = ndarray::ArrayD::from_shape_vec(out_shape, result_flat)
            .map_err(|e| NumpyError::ValueError(e.to_string()))?
            .into_shared();
        return Ok(NdArray::from_data(ArrayData::Float64(result)));
    }

    // Multi-dimensional: similar iteration as trapz but accumulate
    let in_shape = y.shape().to_vec();
    let mut out_iter_shape = out_shape.clone();
    let _ = out_iter_shape; // used below for indexing

    // Iterate over all output positions
    for out_flat in 0..out_n {
        let mut rem = out_flat;
        let mut out_multi = vec![0usize; y.ndim()];
        for d in (0..y.ndim()).rev() {
            out_multi[d] = rem % out_shape[d];
            rem /= out_shape[d];
        }
        // out_multi[axis_idx] = segment index i (0..n-2)
        // Find cumulative sum up to this segment
        let seg = out_multi[axis_idx];
        let mut cumsum = 0.0;
        let mut in_idx = out_multi.clone();
        for k in 0..=seg {
            in_idx[axis_idx] = k;
            let v0 = y_arr[ndarray::IxDyn(&in_idx)];
            in_idx[axis_idx] = k + 1;
            let v1 = y_arr[ndarray::IxDyn(&in_idx)];
            cumsum += 0.5 * (v0 + v1) * dx_vals[k];
        }
        result_flat[out_flat] = cumsum;
    }

    let result = ndarray::ArrayD::from_shape_vec(out_shape, result_flat)
        .map_err(|e| NumpyError::ValueError(e.to_string()))?
        .into_shared();
    Ok(NdArray::from_data(ArrayData::Float64(result)))
}
```

**Note:** The multi-dimensional implementations above are correct but O(n²) for `cumulative_trapezoid` (recomputes from scratch for each segment). This is acceptable for now since the Python version was also O(n²). A future optimization can make it O(n) with prefix sums.

- [ ] **Run Rust tests**

```bash
cargo test --release -p numpy-rust-core 2>&1 | grep -E "test_trapz|test_cumulative|FAILED"
```

- [ ] **Add lib.rs bindings**

```rust
    // --- Reductions ---

    #[pyfunction]
    fn trapz(
        y: vm::PyRef<PyNdArray>,
        x: vm::function::OptionalArg<vm::PyRef<PyNdArray>>,
        dx: vm::function::OptionalArg<f64>,
        axis: vm::function::OptionalArg<i64>,
        vm: &VirtualMachine,
    ) -> PyResult<PyNdArray> {
        let x_ref = x.into_option();
        let dx_val = dx.unwrap_or(1.0);
        let axis_val = axis.into_option();
        let x_inner = x_ref.as_ref().map(|r| r.inner());
        numpy_rust_core::ops::numerical::trapz(
            &y.inner(),
            x_inner.as_deref(),
            dx_val,
            axis_val,
        )
        .map(PyNdArray::from_core)
        .map_err(|e| vm.new_value_error(e.to_string()))
    }

    #[pyfunction]
    fn cumulative_trapezoid(
        y: vm::PyRef<PyNdArray>,
        x: vm::function::OptionalArg<vm::PyRef<PyNdArray>>,
        dx: vm::function::OptionalArg<f64>,
        axis: vm::function::OptionalArg<i64>,
        vm: &VirtualMachine,
    ) -> PyResult<PyNdArray> {
        let x_ref = x.into_option();
        let dx_val = dx.unwrap_or(1.0);
        let axis_val = axis.into_option();
        let x_inner = x_ref.as_ref().map(|r| r.inner());
        numpy_rust_core::ops::numerical::cumulative_trapezoid(
            &y.inner(),
            x_inner.as_deref(),
            dx_val,
            axis_val,
        )
        .map(PyNdArray::from_core)
        .map_err(|e| vm.new_value_error(e.to_string()))
    }
```

**Note:** `numpy_rust_core::ops::numerical::trapz` — verify the path; if `trapz` and `cumulative_trapezoid` are not `pub` in `numerical.rs`, make them `pub` and ensure `ops/mod.rs` exposes `numerical`.

- [ ] **Make numerical module public if needed**

Check `crates/numpy-rust-core/src/ops/mod.rs`:
```bash
cat crates/numpy-rust-core/src/ops/mod.rs
```
If `numerical` is listed, ensure it's `pub mod numerical;`.

- [ ] **Build**

```bash
cargo build --release 2>&1 | grep "error" | head -5
```

- [ ] **Simplify `_reductions.py` wrappers**

```python
def trapz(y, x=None, dx=1.0, axis=-1):
    """Trapezoidal numerical integration."""
    y = asarray(y)
    # Pass x_arr as the array when present; when None, omit it from the positional call
    # to avoid shifting dx into the x slot (lib.rs uses OptionalArg<PyRef<PyNdArray>>).
    if x is not None:
        return _native.trapz(y, asarray(x), float(dx), int(axis))
    return _native.trapz(y, float(dx), int(axis))
```

**Important:** The lib.rs binding signature is `(y, x: OptionalArg<PyRef>, dx: OptionalArg<f64>, axis: OptionalArg<i64>)`. Because `x` is a positional `OptionalArg`, passing `dx` as the second arg when x is absent will put `dx` in the `x` slot and break. The two-branch pattern above is correct: call with 4 args when x is present, 3 args when absent (x is simply omitted).

```python
def cumulative_trapezoid(y, x=None, dx=1.0, axis=-1):
    """Cumulative trapezoidal integration."""
    y = asarray(y)
    if x is not None:
        return _native.cumulative_trapezoid(y, asarray(x), float(dx), int(axis))
    return _native.cumulative_trapezoid(y, float(dx), int(axis))
```

**Note on gradient:** The Python wrapper in `_reductions.py` already calls `_native.gradient()`. Verify it's clean — if there's any Python fallback logic around the `_native.gradient` call, remove it.

- [ ] **Run tests**

```bash
./target/release/numpy-python tests/python/test_math_special.py
./target/release/numpy-python tests/python/test_scimath.py
bash tests/python/run_tests.sh 2>&1 | tail -10
```

---

### Task 10: Full suite verification

- [ ] **Run full compat suite**

```bash
./target/release/numpy-python tests/numpy_compat/run_compat.py --ci 2>&1 | tail -10
```
Expected: 1207 passed, 3 xfails (same as before). If new failures appear, debug before committing.

- [ ] **Run vendored tests**

```bash
bash tests/python/run_tests.sh 2>&1 | tail -5
```
Expected: 1106+ passed, 0 failed.

- [ ] **Run Rust tests**

```bash
cargo test --release 2>&1 | tail -5
```
Expected: 426+ passed, 0 failed.

- [ ] **Final commit**

```bash
git add crates/numpy-rust-core/src/ops/numerical.rs crates/numpy-rust-python/src/lib.rs python/numpy/_reductions.py tests/python/
git commit -m "feat(reductions): add trapz and cumulative_trapezoid in Rust; complete math migration"
```
