# Tier 5: Operator Gaps & Element-wise Rust Migrations — Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Fill remaining operator protocol gaps and migrate pure-Python loop-based functions to Rust.

**Architecture:** Core logic in `numpy-rust-core`, Python bindings in `numpy-rust-python`, thin wrappers in `python/numpy/__init__.py`.

**Tech Stack:** Rust + ndarray + RustPython (PyO3-style macros)

---

## Task 1: Bitwise xor, left shift, right shift — Core

**Files:**
- Modify: `crates/numpy-rust-core/src/ops/logical.rs`

Add three methods to `impl NdArray` using the existing `prepare_bitwise` helper:

```rust
pub fn bitwise_xor(&self, other: &NdArray) -> Result<NdArray> {
    let (a, b) = prepare_bitwise(self, other)?;
    let result = match (a, b) {
        (ArrayData::Bool(a), ArrayData::Bool(b)) => {
            ArrayData::Bool(ndarray::Zip::from(&a).and(&b).map_collect(|&x, &y| x ^ y))
        }
        (ArrayData::Int32(a), ArrayData::Int32(b)) => {
            ArrayData::Int32(ndarray::Zip::from(&a).and(&b).map_collect(|&x, &y| x ^ y))
        }
        (ArrayData::Int64(a), ArrayData::Int64(b)) => {
            ArrayData::Int64(ndarray::Zip::from(&a).and(&b).map_collect(|&x, &y| x ^ y))
        }
        _ => unreachable!("promotion ensures matching types"),
    };
    Ok(NdArray::from_data(result))
}

pub fn left_shift(&self, other: &NdArray) -> Result<NdArray> {
    let (a, b) = prepare_bitwise(self, other)?;
    let result = match (a, b) {
        (ArrayData::Bool(a), ArrayData::Bool(b)) => {
            // Bool shift: cast to int first
            let ai = a.mapv(|x| x as i64);
            let bi = b.mapv(|x| x as i64);
            ArrayData::Int64(ndarray::Zip::from(&ai).and(&bi).map_collect(|&x, &y| x << y))
        }
        (ArrayData::Int32(a), ArrayData::Int32(b)) => {
            ArrayData::Int32(ndarray::Zip::from(&a).and(&b).map_collect(|&x, &y| x << y))
        }
        (ArrayData::Int64(a), ArrayData::Int64(b)) => {
            ArrayData::Int64(ndarray::Zip::from(&a).and(&b).map_collect(|&x, &y| x << y))
        }
        _ => unreachable!(),
    };
    Ok(NdArray::from_data(result))
}

pub fn right_shift(&self, other: &NdArray) -> Result<NdArray> {
    // Same as left_shift but with >>
}
```

Add unit tests:
- `test_bitwise_xor_bool` — `[T,T,F,F] ^ [T,F,T,F]` = `[F,T,T,F]`
- `test_bitwise_xor_int` — `0b1100 ^ 0b1010` = `0b0110`
- `test_left_shift_int` — `1 << 3` = `8`
- `test_right_shift_int` — `8 >> 2` = `2`

**Verify:** `cargo test --workspace --all-features`

---

## Task 2: Bitwise xor/shift Python bindings + abs/bool protocol slots

**Files:**
- Modify: `crates/numpy-rust-python/src/py_array.rs`

Wire in AS_NUMBER:

```rust
xor: Some(|a, b, vm| number_bin_op(a, b, |x, y| x.bitwise_xor(y), vm)),
lshift: Some(|a, b, vm| number_bin_op(a, b, |x, y| x.left_shift(y), vm)),
rshift: Some(|a, b, vm| number_bin_op(a, b, |x, y| x.right_shift(y), vm)),
inplace_xor: Some(|a, b, vm| number_inplace_bin_op(a, b, |x, y| x.bitwise_xor(y), vm)),
inplace_lshift: Some(|a, b, vm| number_inplace_bin_op(a, b, |x, y| x.left_shift(y), vm)),
inplace_rshift: Some(|a, b, vm| number_inplace_bin_op(a, b, |x, y| x.right_shift(y), vm)),
```

Add `absolute` slot:
```rust
absolute: Some(|a, vm| {
    let arr = a.downcast_ref::<PyNdArray>()
        .ok_or_else(|| vm.new_type_error("expected ndarray".to_owned()))?;
    let data = arr.data.read().unwrap();
    Ok(PyNdArray::from_core(data.abs()).into_pyobject(vm))
}),
```

Add `bool` slot:
```rust
bool: Some(|a, vm| {
    let arr = a.downcast_ref::<PyNdArray>()
        .ok_or_else(|| vm.new_type_error("expected ndarray".to_owned()))?;
    let data = arr.data.read().unwrap();
    if data.size() != 1 {
        return Err(vm.new_value_error(
            "The truth value of an array with more than one element is ambiguous".to_owned()));
    }
    let val = data.flatten().get_f64(0);
    Ok(vm.ctx.new_bool(val != 0.0).into())
}),
```

**Verify:** `cargo build --workspace && cargo test --workspace --all-features`

---

## Task 3: isnan, isinf, isfinite — Core + Python bindings

**Files:**
- Modify: `crates/numpy-rust-core/src/utility.rs` — add `isinf` method
- Modify: `crates/numpy-rust-python/src/py_array.rs` — add `isnan`, `isinf`, `isfinite` pymethods
- Modify: `crates/numpy-rust-python/src/lib.rs` — add `isnan`, `isinf`, `isfinite` pyfunctions
- Modify: `python/numpy/__init__.py` — replace Python loops with native calls

Add `isinf` to utility.rs (alongside existing `isnan`/`isfinite`):
```rust
pub fn isinf(&self) -> NdArray {
    let data = match &self.data {
        ArrayData::Float32(a) => ArrayData::Bool(a.mapv(|x| x.is_infinite())),
        ArrayData::Float64(a) => ArrayData::Bool(a.mapv(|x| x.is_infinite())),
        ArrayData::Complex64(a) => ArrayData::Bool(a.mapv(|x| x.re.is_infinite() || x.im.is_infinite())),
        ArrayData::Complex128(a) => ArrayData::Bool(a.mapv(|x| x.re.is_infinite() || x.im.is_infinite())),
        _ => ArrayData::Bool(ArrayD::from_elem(IxDyn(self.shape()), false)),
    };
    NdArray::from_data(data)
}
```

Add pymethods for all three:
```rust
#[pymethod]
fn isnan(&self) -> PyNdArray { PyNdArray::from_core(self.inner().isnan()) }
#[pymethod]
fn isinf(&self) -> PyNdArray { PyNdArray::from_core(self.inner().isinf()) }
#[pymethod]
fn isfinite(&self) -> PyNdArray { PyNdArray::from_core(self.inner().isfinite()) }
```

Add pyfunctions:
```rust
#[pyfunction]
fn isnan(x: PyObjectRef, vm: &VirtualMachine) -> PyResult {
    let arr = obj_to_ndarray(&x, vm)?;
    Ok(PyNdArray::from_core(arr.isnan()).into_pyobject(vm))
}
// Same for isinf, isfinite
```

Replace Python stubs:
```python
def isnan(x):
    if isinstance(x, ndarray):
        return x.isnan()
    return _math.isnan(x)
```

Add Rust unit test: `test_isinf`

**Verify:** full test suite

---

## Task 4: around (round with decimals) and signbit — Core + Python bindings

**Files:**
- Modify: `crates/numpy-rust-core/src/ops/math.rs` — add `around(decimals)` and `signbit` methods
- Modify: `crates/numpy-rust-python/src/py_array.rs` — add pymethods
- Modify: `crates/numpy-rust-python/src/lib.rs` — add pyfunctions
- Modify: `python/numpy/__init__.py` — replace Python loops

Add `around` to math.rs:
```rust
pub fn around(&self, decimals: i32) -> NdArray {
    let factor = 10.0_f64.powi(decimals);
    let data = ensure_float(&self.data);
    let result = match data {
        ArrayData::Float32(a) => {
            let f = factor as f32;
            ArrayData::Float32(a.mapv(|x| (x * f).round() / f))
        }
        ArrayData::Float64(a) => ArrayData::Float64(a.mapv(|x| (x * factor).round() / factor)),
        _ => data, // complex: pass through
    };
    NdArray::from_data(result)
}
```

Add `signbit` to math.rs:
```rust
pub fn signbit(&self) -> NdArray {
    let data = match &self.data {
        ArrayData::Float32(a) => ArrayData::Bool(a.mapv(|x| x.is_sign_negative())),
        ArrayData::Float64(a) => ArrayData::Bool(a.mapv(|x| x.is_sign_negative())),
        ArrayData::Int32(a) => ArrayData::Bool(a.mapv(|x| x < 0)),
        ArrayData::Int64(a) => ArrayData::Bool(a.mapv(|x| x < 0)),
        _ => ArrayData::Bool(ArrayD::from_elem(IxDyn(self.shape()), false)),
    };
    NdArray::from_data(data)
}
```

Pymethod + pyfunction for both. Replace Python stubs.

Add Rust tests: `test_around_decimals`, `test_signbit`

**Verify:** full test suite

---

## Task 5: logical_not and power module function — Core + Python bindings

**Files:**
- Modify: `crates/numpy-rust-core/src/ops/logical.rs` — add `logical_not` method
- Modify: `crates/numpy-rust-python/src/lib.rs` — add `power` and `logical_not` pyfunctions
- Modify: `python/numpy/__init__.py` — replace Python loops

Add `logical_not` to logical.rs:
```rust
pub fn logical_not(&self) -> NdArray {
    let data = match &self.data {
        ArrayData::Bool(a) => ArrayData::Bool(a.mapv(|x| !x)),
        ArrayData::Int32(a) => ArrayData::Bool(a.mapv(|x| x == 0)),
        ArrayData::Int64(a) => ArrayData::Bool(a.mapv(|x| x == 0)),
        ArrayData::Float32(a) => ArrayData::Bool(a.mapv(|x| x == 0.0)),
        ArrayData::Float64(a) => ArrayData::Bool(a.mapv(|x| x == 0.0)),
        ArrayData::Complex64(a) => ArrayData::Bool(a.mapv(|x| x.re == 0.0 && x.im == 0.0)),
        ArrayData::Complex128(a) => ArrayData::Bool(a.mapv(|x| x.re == 0.0 && x.im == 0.0)),
        ArrayData::Str(a) => ArrayData::Bool(a.mapv(|x| x.is_empty())),
    };
    NdArray::from_data(data)
}
```

Add `power` pyfunction (delegates to existing `pow` core method):
```rust
#[pyfunction]
fn power(x1: PyObjectRef, x2: PyObjectRef, vm: &VirtualMachine) -> PyResult {
    let a = obj_to_ndarray(&x1, vm)?;
    let b = obj_to_ndarray(&x2, vm)?;
    a.pow(&b)
        .map(|r| PyNdArray::from_core(r).into_pyobject(vm))
        .map_err(|e| vm.new_value_error(e.to_string()))
}
```

Replace Python stubs:
```python
def power(x1, x2):
    return asarray(x1) ** asarray(x2)

def logical_not(x):
    if isinstance(x, ndarray):
        return _numpy_rust.logical_not(x)
    return not x
```

Add Rust test: `test_logical_not`

**Verify:** full test suite

---

## Task 6: nonzero and count_nonzero — Core + Python bindings

**Files:**
- Modify: `crates/numpy-rust-core/src/utility.rs` — add `nonzero` and `count_nonzero` methods
- Modify: `crates/numpy-rust-python/src/lib.rs` — add pyfunctions
- Modify: `python/numpy/__init__.py` — replace Python loops

Add `nonzero` to utility.rs. NumPy returns a tuple of arrays, one per dimension:
```rust
pub fn nonzero(a: &NdArray) -> Vec<NdArray> {
    let flat = a.astype(crate::DType::Float64);
    let ArrayData::Float64(arr) = &flat.data else { unreachable!() };
    let shape = a.shape();
    let ndim = a.ndim().max(1); // 1-D for scalars

    let mut indices: Vec<Vec<i64>> = vec![Vec::new(); ndim];
    for (linear_idx, &val) in arr.iter().enumerate() {
        if val != 0.0 {
            let mut remaining = linear_idx;
            for d in (0..ndim).rev() {
                let dim_size = if d < shape.len() { shape[d] } else { 1 };
                indices[d].push((remaining % dim_size) as i64);
                remaining /= dim_size;
            }
        }
    }
    indices.into_iter().map(|idx| {
        NdArray::from_data(ArrayData::Int64(
            ArrayD::from_shape_vec(IxDyn(&[idx.len()]), idx).unwrap()
        ))
    }).collect()
}

pub fn count_nonzero(a: &NdArray) -> usize {
    let flat = a.astype(crate::DType::Float64);
    let ArrayData::Float64(arr) = &flat.data else { unreachable!() };
    arr.iter().filter(|&&x| x != 0.0).count()
}
```

Pyfunction for `nonzero` returns a Python tuple of arrays:
```rust
#[pyfunction]
fn nonzero(a: PyObjectRef, vm: &VirtualMachine) -> PyResult {
    let arr = obj_to_ndarray(&a, vm)?;
    let result = numpy_rust_core::nonzero(&arr);
    let py_arrays: Vec<PyObjectRef> = result.into_iter()
        .map(|r| PyNdArray::from_core(r).into_pyobject(vm))
        .collect();
    Ok(vm.ctx.new_tuple(py_arrays).into())
}
```

Replace Python stubs.

Add Rust tests: `test_nonzero_1d`, `test_nonzero_2d`, `test_count_nonzero`

**Verify:** full test suite

---

## Task 7: Python integration tests

**Files:**
- Modify: `tests/python/test_numeric.py`

Add tests for all new features:

```python
# Bitwise
def test_bitwise_xor(): a = np.array([True, False]); b = np.array([False, True]); assert (a ^ b).tolist() == [True, True]
def test_left_shift(): a = np.array([1, 2, 4]); assert (a << 2).tolist() == [4, 8, 16]
def test_right_shift(): a = np.array([4, 8, 16]); assert (a >> 2).tolist() == [1, 2, 4]

# abs/bool
def test_abs_operator(): a = np.array([-1.0, 2.0, -3.0]); assert abs(a).tolist() == [1.0, 2.0, 3.0]
def test_bool_scalar(): assert bool(np.array([1.0]))
def test_bool_multi_raises(): caught ValueError for bool(np.array([1.0, 2.0]))

# isnan/isinf/isfinite
def test_isnan(): np.isnan(np.array([1.0, float('nan')])) returns Bool array
def test_isinf(): np.isinf(np.array([1.0, float('inf')])) returns Bool array
def test_isfinite(): np.isfinite(np.array([1.0, float('inf'), float('nan')])) returns Bool array

# around/signbit
def test_around(): np.around(np.array([1.234, 5.678]), 2) → [1.23, 5.68]
def test_signbit(): np.signbit(np.array([-1.0, 0.0, 1.0])) → [True, False, False]

# logical_not/power
def test_logical_not(): np.logical_not(np.array([True, False])).tolist() == [False, True]
def test_power_func(): np.power(np.array([2.0, 3.0]), 2.0).tolist() == [4.0, 9.0]

# nonzero/count_nonzero
def test_nonzero(): np.nonzero(np.array([0, 1, 0, 3])) returns tuple with indices [1, 3]
def test_count_nonzero(): np.count_nonzero(np.array([0, 1, 0, 3])) == 2

# In-place bitwise
def test_inplace_xor(): a ^= b works
def test_inplace_lshift(): a <<= 1 works
def test_inplace_rshift(): a >>= 1 works
```

**Verify:** full test suite including `./tests/python/run_tests.sh target/debug/numpy-python`

---

## Verification

After each task:
1. `cargo fmt --all -- --check`
2. `cargo clippy --workspace --all-features -- -D warnings`
3. `cargo test --workspace --all-features`
4. `./tests/python/run_tests.sh target/debug/numpy-python`
