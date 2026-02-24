# numpy-rust Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build a NumPy-compatible pure Rust array library, then wrap it for RustPython/WASM.

**Architecture:** 3-crate workspace (core/python/wasm). Enum-based dtype dispatch over 5 types (f32, f64, i32, i64, bool). Core lib built and tested first in pure Rust, Python bindings added later.

**Tech Stack:** ndarray, faer, rustfft, rand, num-traits, thiserror, rustpython-vm

**Key ndarray API notes:**
- `ArrayD<T>` for dynamic-dimension arrays
- `Array::zeros(IxDyn(&[m, n]))` for creation
- `Array::range(start, end, step)` for arange equivalent
- `.sum_axis(Axis(n))` for axis reductions
- `.broadcast(shape)` for broadcasting (right-hand only — we must handle symmetric broadcasting ourselves)
- `s![]` macro for slicing
- `concatenate(Axis(n), &[a.view(), b.view()])` for joining

---

## Phase 1: Project Scaffolding

### Task 1: Initialize workspace and core crate

**Files:**
- Create: `Cargo.toml` (workspace root)
- Create: `crates/numpy-rust-core/Cargo.toml`
- Create: `crates/numpy-rust-core/src/lib.rs`

**Step 1: Create workspace Cargo.toml**

```toml
[workspace]
members = ["crates/*"]
resolver = "2"
```

**Step 2: Create core crate Cargo.toml**

```toml
[package]
name = "numpy-rust-core"
version = "0.1.0"
edition = "2021"

[dependencies]
ndarray = { version = "0.16", default-features = false }
num-traits = "0.2"
thiserror = "2"

[features]
default = ["linalg", "fft", "random"]
linalg = ["dep:faer"]
fft = ["dep:rustfft"]
random = ["dep:rand", "dep:rand_distr"]

[dependencies.faer]
version = "0.21"
default-features = false
optional = true

[dependencies.rustfft]
version = "6.2"
optional = true

[dependencies.rand]
version = "0.9"
optional = true

[dependencies.rand_distr]
version = "0.5"
optional = true
```

**Step 3: Create minimal lib.rs**

```rust
pub mod dtype;
pub mod error;

pub use dtype::DType;
pub use error::NumpyError;
```

**Step 4: Verify it compiles**

Run: `cargo check -p numpy-rust-core`
Expected: compiles (will fail until we create dtype.rs and error.rs in Task 2)

**Step 5: Commit**

```bash
git add Cargo.toml crates/
git commit -m "chore: initialize workspace and core crate"
```

---

### Task 2: Error types and DType enum

**Files:**
- Create: `crates/numpy-rust-core/src/error.rs`
- Create: `crates/numpy-rust-core/src/dtype.rs`
- Modify: `crates/numpy-rust-core/src/lib.rs`

**Step 1: Write tests for DType**

Add to `crates/numpy-rust-core/src/dtype.rs`:

```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_promote_same_type() {
        assert_eq!(DType::Float64.promote(DType::Float64), DType::Float64);
        assert_eq!(DType::Int32.promote(DType::Int32), DType::Int32);
    }

    #[test]
    fn test_promote_bool_widens() {
        assert_eq!(DType::Bool.promote(DType::Int32), DType::Int32);
        assert_eq!(DType::Bool.promote(DType::Float64), DType::Float64);
    }

    #[test]
    fn test_promote_int_to_float() {
        assert_eq!(DType::Int32.promote(DType::Float32), DType::Float32);
        assert_eq!(DType::Int64.promote(DType::Float64), DType::Float64);
    }

    #[test]
    fn test_promote_int32_int64() {
        assert_eq!(DType::Int32.promote(DType::Int64), DType::Int64);
    }

    #[test]
    fn test_promote_mixed_int_float() {
        // i64 + f32 -> f64 (to avoid precision loss)
        assert_eq!(DType::Int64.promote(DType::Float32), DType::Float64);
    }

    #[test]
    fn test_promote_is_symmetric() {
        let pairs = [
            (DType::Int32, DType::Float64),
            (DType::Bool, DType::Int64),
            (DType::Float32, DType::Int32),
        ];
        for (a, b) in pairs {
            assert_eq!(a.promote(b), b.promote(a), "promote({a:?}, {b:?}) not symmetric");
        }
    }
}
```

**Step 2: Run tests to verify they fail**

Run: `cargo test -p numpy-rust-core`
Expected: compilation error — `DType` not defined yet

**Step 3: Implement error.rs**

```rust
use thiserror::Error;

#[derive(Error, Debug)]
pub enum NumpyError {
    #[error("shape mismatch: {0}")]
    ShapeMismatch(String),

    #[error("invalid axis {axis} for array with {ndim} dimensions")]
    InvalidAxis { axis: usize, ndim: usize },

    #[error("cannot broadcast shapes {0:?} and {1:?}")]
    BroadcastError(Vec<usize>, Vec<usize>),

    #[error("cannot reshape array of size {from} into shape {to:?}")]
    ReshapeError { from: usize, to: Vec<usize> },

    #[error("type error: {0}")]
    TypeError(String),

    #[error("value error: {0}")]
    ValueError(String),
}

pub type Result<T> = std::result::Result<T, NumpyError>;
```

**Step 4: Implement dtype.rs**

```rust
/// Supported data types, mirroring NumPy's core numeric dtypes.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum DType {
    Bool,
    Int32,
    Int64,
    Float32,
    Float64,
}

impl DType {
    /// Returns the common type that both `self` and `other` can be promoted to,
    /// following NumPy's type promotion rules (simplified for 5 types).
    ///
    /// Promotion lattice:
    ///   Bool -> Int32 -> Int64 -> Float64
    ///                    Float32 -> Float64
    ///   i64 + f32 -> f64 (to avoid precision loss)
    pub fn promote(self, other: DType) -> DType {
        use DType::*;
        if self == other {
            return self;
        }
        match (self.rank(), other.rank()) {
            _ if self == other => self,
            _ => {
                let (a, b) = if self.rank() >= other.rank() {
                    (self, other)
                } else {
                    (other, self)
                };
                // Special case: i64 + f32 -> f64
                if (a == Int64 && b == Float32) || (a == Float32 && b == Int64) {
                    return Float64;
                }
                a
            }
        }
    }

    /// Numeric rank for promotion ordering.
    /// Higher rank wins in promotion (except special cases).
    fn rank(self) -> u8 {
        match self {
            DType::Bool => 0,
            DType::Int32 => 1,
            DType::Int64 => 2,
            DType::Float32 => 3,
            DType::Float64 => 4,
        }
    }

    /// Size in bytes of a single element.
    pub fn itemsize(self) -> usize {
        match self {
            DType::Bool => 1,
            DType::Int32 => 4,
            DType::Int64 => 8,
            DType::Float32 => 4,
            DType::Float64 => 8,
        }
    }

    /// Returns true if this is a floating-point type.
    pub fn is_float(self) -> bool {
        matches!(self, DType::Float32 | DType::Float64)
    }

    /// Returns true if this is an integer type.
    pub fn is_integer(self) -> bool {
        matches!(self, DType::Int32 | DType::Int64)
    }
}

impl std::fmt::Display for DType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            DType::Bool => write!(f, "bool"),
            DType::Int32 => write!(f, "int32"),
            DType::Int64 => write!(f, "int64"),
            DType::Float32 => write!(f, "float32"),
            DType::Float64 => write!(f, "float64"),
        }
    }
}

// tests at bottom (from Step 1)
```

**Step 5: Update lib.rs**

```rust
pub mod dtype;
pub mod error;

pub use dtype::DType;
pub use error::{NumpyError, Result};
```

**Step 6: Run tests**

Run: `cargo test -p numpy-rust-core`
Expected: all DType tests pass

**Step 7: Commit**

```bash
git add -A crates/numpy-rust-core/src/
git commit -m "feat: add DType enum with promotion rules and error types"
```

---

## Phase 2: ArrayData and NdArray Core

### Task 3: ArrayData enum and NdArray struct

**Files:**
- Create: `crates/numpy-rust-core/src/array_data.rs`
- Create: `crates/numpy-rust-core/src/array.rs`
- Modify: `crates/numpy-rust-core/src/lib.rs`

**Step 1: Write tests for NdArray basics**

In `crates/numpy-rust-core/src/array.rs` (at bottom):

```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_from_f64_vec() {
        let a = NdArray::from_vec(vec![1.0_f64, 2.0, 3.0]);
        assert_eq!(a.shape(), &[3]);
        assert_eq!(a.ndim(), 1);
        assert_eq!(a.dtype(), DType::Float64);
        assert_eq!(a.size(), 3);
    }

    #[test]
    fn test_from_i32_vec() {
        let a = NdArray::from_vec(vec![1_i32, 2, 3]);
        assert_eq!(a.dtype(), DType::Int32);
        assert_eq!(a.size(), 3);
    }

    #[test]
    fn test_zeros() {
        let a = NdArray::zeros(&[2, 3], DType::Float64);
        assert_eq!(a.shape(), &[2, 3]);
        assert_eq!(a.ndim(), 2);
        assert_eq!(a.size(), 6);
    }

    #[test]
    fn test_ones() {
        let a = NdArray::ones(&[3], DType::Int32);
        assert_eq!(a.shape(), &[3]);
        assert_eq!(a.dtype(), DType::Int32);
    }
}
```

**Step 2: Run tests to verify they fail**

Run: `cargo test -p numpy-rust-core`
Expected: compilation error

**Step 3: Implement array_data.rs**

```rust
use ndarray::{ArrayD, IxDyn};
use crate::dtype::DType;

/// Type-erased array storage. Each variant holds a concrete `ArrayD<T>`.
#[derive(Debug, Clone)]
pub enum ArrayData {
    Bool(ArrayD<bool>),
    Int32(ArrayD<i32>),
    Int64(ArrayD<i64>),
    Float32(ArrayD<f32>),
    Float64(ArrayD<f64>),
}

impl ArrayData {
    pub fn dtype(&self) -> DType {
        match self {
            ArrayData::Bool(_) => DType::Bool,
            ArrayData::Int32(_) => DType::Int32,
            ArrayData::Int64(_) => DType::Int64,
            ArrayData::Float32(_) => DType::Float32,
            ArrayData::Float64(_) => DType::Float64,
        }
    }

    pub fn shape(&self) -> &[usize] {
        match self {
            ArrayData::Bool(a) => a.shape(),
            ArrayData::Int32(a) => a.shape(),
            ArrayData::Int64(a) => a.shape(),
            ArrayData::Float32(a) => a.shape(),
            ArrayData::Float64(a) => a.shape(),
        }
    }

    pub fn ndim(&self) -> usize {
        match self {
            ArrayData::Bool(a) => a.ndim(),
            ArrayData::Int32(a) => a.ndim(),
            ArrayData::Int64(a) => a.ndim(),
            ArrayData::Float32(a) => a.ndim(),
            ArrayData::Float64(a) => a.ndim(),
        }
    }

    pub fn size(&self) -> usize {
        self.shape().iter().product()
    }
}

/// Macro to reduce boilerplate when dispatching over all ArrayData variants.
/// Calls $body with the inner ArrayD bound to $name.
#[macro_export]
macro_rules! dispatch_unary {
    ($data:expr, $name:ident, $body:expr) => {
        match $data {
            ArrayData::Bool($name) => $body,
            ArrayData::Int32($name) => $body,
            ArrayData::Int64($name) => $body,
            ArrayData::Float32($name) => $body,
            ArrayData::Float64($name) => $body,
        }
    };
}
```

**Step 4: Implement array.rs**

```rust
use ndarray::{ArrayD, IxDyn};
use crate::array_data::ArrayData;
use crate::dtype::DType;

/// The main N-dimensional array type, analogous to `numpy.ndarray`.
#[derive(Debug, Clone)]
pub struct NdArray {
    pub(crate) data: ArrayData,
}

// --- Constructors ---

impl NdArray {
    /// Create an NdArray from existing ArrayData.
    pub fn from_data(data: ArrayData) -> Self {
        Self { data }
    }

    /// Create a 1-D array from a Vec<f64>.
    pub fn from_vec<T: Into<ArrayData1D>>(vec: Vec<T>) -> Self
    where
        Vec<T>: IntoArrayData,
    {
        Self { data: vec.into_array_data() }
    }

    /// Create an array filled with zeros.
    pub fn zeros(shape: &[usize], dtype: DType) -> Self {
        let sh = IxDyn(shape);
        let data = match dtype {
            DType::Bool => ArrayData::Bool(ArrayD::from_elem(sh, false)),
            DType::Int32 => ArrayData::Int32(ArrayD::zeros(sh)),
            DType::Int64 => ArrayData::Int64(ArrayD::zeros(sh)),
            DType::Float32 => ArrayData::Float32(ArrayD::zeros(sh)),
            DType::Float64 => ArrayData::Float64(ArrayD::zeros(sh)),
        };
        Self { data }
    }

    /// Create an array filled with ones.
    pub fn ones(shape: &[usize], dtype: DType) -> Self {
        let sh = IxDyn(shape);
        let data = match dtype {
            DType::Bool => ArrayData::Bool(ArrayD::from_elem(sh, true)),
            DType::Int32 => ArrayData::Int32(ArrayD::ones(sh)),
            DType::Int64 => ArrayData::Int64(ArrayD::ones(sh)),
            DType::Float32 => ArrayData::Float32(ArrayD::ones(sh)),
            DType::Float64 => ArrayData::Float64(ArrayD::ones(sh)),
        };
        Self { data }
    }

    /// Create an array filled with a given value.
    pub fn full_f64(shape: &[usize], value: f64) -> Self {
        let sh = IxDyn(shape);
        Self { data: ArrayData::Float64(ArrayD::from_elem(sh, value)) }
    }
}

// --- Attributes ---

impl NdArray {
    pub fn shape(&self) -> &[usize] {
        self.data.shape()
    }

    pub fn ndim(&self) -> usize {
        self.data.ndim()
    }

    pub fn dtype(&self) -> DType {
        self.data.dtype()
    }

    pub fn size(&self) -> usize {
        self.data.size()
    }

    /// Reference to the inner ArrayData.
    pub fn data(&self) -> &ArrayData {
        &self.data
    }
}

// --- Trait for converting Vec<T> to ArrayData ---

pub trait IntoArrayData {
    fn into_array_data(self) -> ArrayData;
}

impl IntoArrayData for Vec<f64> {
    fn into_array_data(self) -> ArrayData {
        ArrayData::Float64(ArrayD::from_shape_vec(IxDyn(&[self.len()]), self).unwrap())
    }
}

impl IntoArrayData for Vec<f32> {
    fn into_array_data(self) -> ArrayData {
        ArrayData::Float32(ArrayD::from_shape_vec(IxDyn(&[self.len()]), self).unwrap())
    }
}

impl IntoArrayData for Vec<i32> {
    fn into_array_data(self) -> ArrayData {
        ArrayData::Int32(ArrayD::from_shape_vec(IxDyn(&[self.len()]), self).unwrap())
    }
}

impl IntoArrayData for Vec<i64> {
    fn into_array_data(self) -> ArrayData {
        ArrayData::Int64(ArrayD::from_shape_vec(IxDyn(&[self.len()]), self).unwrap())
    }
}

impl IntoArrayData for Vec<bool> {
    fn into_array_data(self) -> ArrayData {
        ArrayData::Bool(ArrayD::from_shape_vec(IxDyn(&[self.len()]), self).unwrap())
    }
}

impl<T> NdArray
where
    Vec<T>: IntoArrayData,
{
    pub fn from_vec(vec: Vec<T>) -> Self {
        Self { data: vec.into_array_data() }
    }
}

// tests at bottom...
```

Note: The `from_vec` has a duplicate definition — remove the non-generic one and keep only the generic impl block at the bottom.

**Step 5: Update lib.rs**

```rust
pub mod array;
pub mod array_data;
pub mod dtype;
pub mod error;

pub use array::NdArray;
pub use array_data::ArrayData;
pub use dtype::DType;
pub use error::{NumpyError, Result};
```

**Step 6: Run tests**

Run: `cargo test -p numpy-rust-core`
Expected: all tests pass

**Step 7: Commit**

```bash
git add -A crates/numpy-rust-core/
git commit -m "feat: add ArrayData enum and NdArray struct with zeros/ones/from_vec"
```

---

### Task 4: Type casting (astype)

**Files:**
- Create: `crates/numpy-rust-core/src/casting.rs`
- Modify: `crates/numpy-rust-core/src/array.rs` (add `astype` method)
- Modify: `crates/numpy-rust-core/src/lib.rs`

**Step 1: Write tests**

In `crates/numpy-rust-core/src/casting.rs`:

```rust
#[cfg(test)]
mod tests {
    use crate::{NdArray, DType};

    #[test]
    fn test_cast_i32_to_f64() {
        let a = NdArray::from_vec(vec![1_i32, 2, 3]);
        let b = a.astype(DType::Float64);
        assert_eq!(b.dtype(), DType::Float64);
    }

    #[test]
    fn test_cast_f64_to_i32_truncates() {
        let a = NdArray::from_vec(vec![1.7_f64, 2.3, 3.9]);
        let b = a.astype(DType::Int32);
        assert_eq!(b.dtype(), DType::Int32);
    }

    #[test]
    fn test_cast_same_type_is_clone() {
        let a = NdArray::from_vec(vec![1.0_f64, 2.0]);
        let b = a.astype(DType::Float64);
        assert_eq!(b.dtype(), DType::Float64);
        assert_eq!(b.shape(), a.shape());
    }

    #[test]
    fn test_cast_bool_to_int() {
        let a = NdArray::from_vec(vec![true, false, true]);
        let b = a.astype(DType::Int32);
        assert_eq!(b.dtype(), DType::Int32);
    }
}
```

**Step 2: Implement casting.rs**

Implement `cast_array_data(data: &ArrayData, target: DType) -> ArrayData` that uses `mapv` to convert element types. Each source variant maps to each target dtype via `as` casts. Use a macro to reduce the 25-arm match.

**Step 3: Add `astype` to NdArray**

```rust
impl NdArray {
    pub fn astype(&self, dtype: DType) -> Self {
        Self { data: cast_array_data(&self.data, dtype) }
    }
}
```

**Step 4: Run tests, verify pass, commit**

```bash
git commit -m "feat: add type casting (astype) between all dtype pairs"
```

---

## Phase 3: Broadcasting

### Task 5: Broadcasting shape computation

**Files:**
- Create: `crates/numpy-rust-core/src/broadcasting.rs`
- Modify: `crates/numpy-rust-core/src/lib.rs`

**Step 1: Write tests**

```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_same_shape() {
        assert_eq!(broadcast_shape(&[3, 4], &[3, 4]).unwrap(), vec![3, 4]);
    }

    #[test]
    fn test_scalar_broadcast() {
        assert_eq!(broadcast_shape(&[3, 4], &[]).unwrap(), vec![3, 4]);
    }

    #[test]
    fn test_1d_broadcast() {
        assert_eq!(broadcast_shape(&[3, 4], &[4]).unwrap(), vec![3, 4]);
    }

    #[test]
    fn test_2d_broadcast() {
        assert_eq!(broadcast_shape(&[3, 1], &[1, 4]).unwrap(), vec![3, 4]);
    }

    #[test]
    fn test_incompatible_shapes() {
        assert!(broadcast_shape(&[3, 4], &[3, 5]).is_err());
    }

    #[test]
    fn test_higher_rank() {
        assert_eq!(
            broadcast_shape(&[2, 1, 5], &[3, 1]).unwrap(),
            vec![2, 3, 5]
        );
    }
}
```

**Step 2: Implement broadcast_shape**

```rust
use crate::error::{NumpyError, Result};

/// Compute the broadcast-compatible output shape for two input shapes,
/// following NumPy's broadcasting rules.
pub fn broadcast_shape(a: &[usize], b: &[usize]) -> Result<Vec<usize>> {
    let ndim = a.len().max(b.len());
    let mut result = Vec::with_capacity(ndim);
    for i in 0..ndim {
        let da = if i < ndim - a.len() { 1 } else { a[i - (ndim - a.len())] };
        let db = if i < ndim - b.len() { 1 } else { b[i - (ndim - b.len())] };
        if da == db {
            result.push(da);
        } else if da == 1 {
            result.push(db);
        } else if db == 1 {
            result.push(da);
        } else {
            return Err(NumpyError::BroadcastError(a.to_vec(), b.to_vec()));
        }
    }
    Ok(result)
}
```

**Step 3: Add a helper to broadcast ArrayData to a target shape**

This uses ndarray's `.broadcast(shape)` to create a view, then `.to_owned()` to materialize it when the shapes differ. Important: ndarray's broadcast only works when the array can be broadcast to the target — we've already validated this via `broadcast_shape`.

**Step 4: Run tests, verify pass, commit**

```bash
git commit -m "feat: add NumPy-compatible broadcasting shape computation"
```

---

## Phase 4: Arithmetic Operations

### Task 6: Element-wise arithmetic with broadcasting

**Files:**
- Create: `crates/numpy-rust-core/src/ops/mod.rs`
- Create: `crates/numpy-rust-core/src/ops/arithmetic.rs`
- Modify: `crates/numpy-rust-core/src/lib.rs`

**Step 1: Write tests**

In `crates/numpy-rust-core/src/ops/arithmetic.rs`:

```rust
#[cfg(test)]
mod tests {
    use crate::{NdArray, DType};

    #[test]
    fn test_add_same_shape() {
        let a = NdArray::from_vec(vec![1.0_f64, 2.0, 3.0]);
        let b = NdArray::from_vec(vec![4.0_f64, 5.0, 6.0]);
        let c = (&a + &b).unwrap();
        assert_eq!(c.shape(), &[3]);
        assert_eq!(c.dtype(), DType::Float64);
    }

    #[test]
    fn test_add_type_promotion() {
        let a = NdArray::from_vec(vec![1_i32, 2, 3]);
        let b = NdArray::from_vec(vec![1.0_f64, 2.0, 3.0]);
        let c = (&a + &b).unwrap();
        assert_eq!(c.dtype(), DType::Float64);
    }

    #[test]
    fn test_add_broadcast() {
        let a = NdArray::zeros(&[3, 4], DType::Float64);
        let b = NdArray::ones(&[4], DType::Float64);
        let c = (&a + &b).unwrap();
        assert_eq!(c.shape(), &[3, 4]);
    }

    #[test]
    fn test_sub() {
        let a = NdArray::from_vec(vec![5.0_f64, 3.0]);
        let b = NdArray::from_vec(vec![1.0_f64, 1.0]);
        let c = (&a - &b).unwrap();
        assert_eq!(c.dtype(), DType::Float64);
    }

    #[test]
    fn test_mul() {
        let a = NdArray::from_vec(vec![2.0_f64, 3.0]);
        let b = NdArray::from_vec(vec![4.0_f64, 5.0]);
        let _c = (&a * &b).unwrap();
    }

    #[test]
    fn test_div() {
        let a = NdArray::from_vec(vec![10.0_f64, 20.0]);
        let b = NdArray::from_vec(vec![2.0_f64, 5.0]);
        let _c = (&a / &b).unwrap();
    }

    #[test]
    fn test_broadcast_incompatible_fails() {
        let a = NdArray::zeros(&[3, 4], DType::Float64);
        let b = NdArray::zeros(&[5], DType::Float64);
        assert!((&a + &b).is_err());
    }
}
```

**Step 2: Implement arithmetic**

The pattern for each binary op (add, sub, mul, div):
1. Compute `broadcast_shape` for the two operands
2. Promote both to their common dtype via `astype`
3. Broadcast both to the output shape via ndarray's `.broadcast().to_owned()`
4. Match on the (now same) dtype and apply the operation

Use a macro `impl_binary_op!` that generates an `impl std::ops::Add<&NdArray> for &NdArray` (returning `Result<NdArray>`) for each of `Add/Sub/Mul/Div`.

**Step 3: Run tests, verify pass, commit**

```bash
git commit -m "feat: add element-wise arithmetic ops with broadcasting and type promotion"
```

---

### Task 7: Comparison operations

**Files:**
- Create: `crates/numpy-rust-core/src/ops/comparison.rs`
- Modify: `crates/numpy-rust-core/src/ops/mod.rs`

Implement `eq`, `ne`, `lt`, `gt`, `le`, `ge` as methods on NdArray that return `NdArray` with `DType::Bool`. Same broadcast + promote pattern as arithmetic. Tests cover same-shape, broadcasting, and cross-dtype comparison.

**Commit:** `feat: add comparison operations returning bool arrays`

---

### Task 8: Unary math functions

**Files:**
- Create: `crates/numpy-rust-core/src/ops/math.rs`
- Modify: `crates/numpy-rust-core/src/ops/mod.rs`

Implement as methods on NdArray: `abs`, `neg`, `sqrt`, `exp`, `log`, `sin`, `cos`, `tan`, `floor`, `ceil`, `round`.

For integer types, `sqrt`/`exp`/`log`/trig should cast to f64 first (matching NumPy behavior). `abs` works on int and float. `neg` works on int and float (not bool).

Use `mapv(|x| x.abs())` pattern on the inner ArrayD.

Tests: verify output dtype, verify known values (e.g., `sqrt([4.0, 9.0]) == [2.0, 3.0]`).

**Commit:** `feat: add unary math functions (abs, sqrt, exp, log, trig)`

---

## Phase 5: Reductions

### Task 9: Reduction operations (sum, mean, min, max, etc.)

**Files:**
- Create: `crates/numpy-rust-core/src/ops/reduction.rs`
- Modify: `crates/numpy-rust-core/src/ops/mod.rs`

**Key tests:**

```rust
#[test]
fn test_sum_all() {
    let a = NdArray::from_vec(vec![1.0_f64, 2.0, 3.0]);
    let s = a.sum(None).unwrap(); // sum over all elements
    assert_eq!(s.size(), 1); // scalar result
}

#[test]
fn test_sum_axis() {
    let a = NdArray::zeros(&[3, 4], DType::Float64);
    let s = a.sum(Some(0)).unwrap(); // sum along axis 0
    assert_eq!(s.shape(), &[4]);
}

#[test]
fn test_mean() {
    let a = NdArray::from_vec(vec![1.0_f64, 2.0, 3.0]);
    let m = a.mean(None).unwrap();
    assert_eq!(m.size(), 1);
}
```

Implement: `sum`, `mean`, `min`, `max`, `std`, `var` — each takes `Option<usize>` for axis (None = reduce all). Also `argmin`, `argmax`, `all`, `any`.

For axis reduction, use ndarray's `.sum_axis(Axis(n))`. For full reduction, use `.sum()` or `.iter().sum()`. `std` and `var` compute manually: `var = mean(x^2) - mean(x)^2`, `std = sqrt(var)`.

**Commit:** `feat: add reduction operations (sum, mean, min, max, std, var, argmin, argmax)`

---

## Phase 6: Array Creation Functions

### Task 10: arange, linspace, eye, full, zeros_like, ones_like

**Files:**
- Create: `crates/numpy-rust-core/src/creation.rs`
- Modify: `crates/numpy-rust-core/src/lib.rs`

**Key tests:**

```rust
#[test]
fn test_arange() {
    let a = arange(0.0, 5.0, 1.0);
    assert_eq!(a.shape(), &[5]);
    assert_eq!(a.dtype(), DType::Float64);
}

#[test]
fn test_linspace() {
    let a = linspace(0.0, 1.0, 5);
    assert_eq!(a.shape(), &[5]);
}

#[test]
fn test_eye() {
    let a = eye(3, DType::Float64);
    assert_eq!(a.shape(), &[3, 3]);
}

#[test]
fn test_zeros_like() {
    let a = NdArray::ones(&[2, 3], DType::Int32);
    let b = zeros_like(&a);
    assert_eq!(b.shape(), &[2, 3]);
    assert_eq!(b.dtype(), DType::Int32);
}
```

Implement as free functions: `arange(start, stop, step) -> NdArray`, `linspace(start, stop, num) -> NdArray`, `eye(n, dtype) -> NdArray`, `full(shape, value, dtype) -> NdArray`, `zeros_like(&NdArray) -> NdArray`, `ones_like(&NdArray) -> NdArray`.

**Commit:** `feat: add creation functions (arange, linspace, eye, full, zeros_like, ones_like)`

---

## Phase 7: Shape Manipulation

### Task 11: reshape, transpose, flatten, concatenate, stack

**Files:**
- Create: `crates/numpy-rust-core/src/manipulation.rs`
- Modify: `crates/numpy-rust-core/src/lib.rs`

**Key tests:**

```rust
#[test]
fn test_reshape() {
    let a = NdArray::from_vec(vec![1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0]);
    let b = a.reshape(&[2, 3]).unwrap();
    assert_eq!(b.shape(), &[2, 3]);
}

#[test]
fn test_reshape_invalid() {
    let a = NdArray::from_vec(vec![1.0_f64, 2.0, 3.0]);
    assert!(a.reshape(&[2, 2]).is_err());
}

#[test]
fn test_transpose() {
    let a = NdArray::zeros(&[2, 3], DType::Float64);
    let b = a.transpose();
    assert_eq!(b.shape(), &[3, 2]);
}

#[test]
fn test_flatten() {
    let a = NdArray::zeros(&[2, 3], DType::Float64);
    let b = a.flatten();
    assert_eq!(b.shape(), &[6]);
}

#[test]
fn test_concatenate() {
    let a = NdArray::zeros(&[2, 3], DType::Float64);
    let b = NdArray::ones(&[2, 3], DType::Float64);
    let c = concatenate(&[&a, &b], 0).unwrap();
    assert_eq!(c.shape(), &[4, 3]);
}
```

Use ndarray's `.into_shape()` for reshape, `.t()` for transpose, `ndarray::concatenate()` for joining. `stack` creates a new axis.

**Commit:** `feat: add shape manipulation (reshape, transpose, flatten, concatenate, stack)`

---

## Phase 8: Indexing

### Task 12: Basic slicing and integer indexing

**Files:**
- Create: `crates/numpy-rust-core/src/indexing.rs`
- Modify: `crates/numpy-rust-core/src/lib.rs`

Implement:
- `get(&[usize]) -> Scalar` — single element access
- `slice(ranges: &[SliceArg]) -> NdArray` — slicing with ranges/steps
- `index_select(axis, indices: &[usize]) -> NdArray` — fancy indexing along axis

Define a `SliceArg` enum: `Index(usize)`, `Range(start, stop, step)`, `Full`.

**Commit:** `feat: add basic slicing and integer indexing`

---

### Task 13: Boolean masking

**Files:**
- Modify: `crates/numpy-rust-core/src/indexing.rs`

Implement `mask_select(mask: &NdArray) -> NdArray` — returns 1-D array of elements where mask is true (like `a[a > 0]`).

**Commit:** `feat: add boolean mask indexing`

---

## Phase 9: Utility Functions

### Task 14: where, isnan, isfinite, dot, copy

**Files:**
- Create: `crates/numpy-rust-core/src/utility.rs`
- Modify: `crates/numpy-rust-core/src/lib.rs`

- `where_cond(cond: &NdArray, x: &NdArray, y: &NdArray) -> NdArray` — element-wise ternary
- `isnan`, `isfinite` — returns Bool array (always false for int types)
- `dot` — 1-D dot product or matrix multiply (delegates to ndarray's `.dot()`)
- `copy` — deep clone

**Commit:** `feat: add utility functions (where, isnan, isfinite, dot, copy)`

---

## Phase 10: Linear Algebra (faer)

### Task 15: numpy.linalg module

**Files:**
- Create: `crates/numpy-rust-core/src/linalg.rs`
- Modify: `crates/numpy-rust-core/src/lib.rs`

Implement behind `linalg` feature flag. Functions convert NdArray to `faer::Mat<f64>` then back:

- `matmul(a, b) -> NdArray` — matrix multiply
- `inv(a) -> NdArray` — matrix inverse via LU
- `solve(a, b) -> NdArray` — solve Ax=b
- `det(a) -> f64` — determinant via LU
- `eig(a) -> (NdArray, NdArray)` — eigenvalues, eigenvectors
- `svd(a) -> (NdArray, NdArray, NdArray)` — U, S, Vt
- `qr(a) -> (NdArray, NdArray)` — Q, R
- `norm(a) -> f64` — Frobenius norm
- `cholesky(a) -> NdArray` — Cholesky decomposition

Use `faer` API: `a.svd()`, `a.eigen()`, `a.llt()`, `a.partial_piv_lu()`, `a.qr()`.

**Commit:** `feat: add numpy.linalg backed by faer`

---

## Phase 11: FFT (rustfft)

### Task 16: numpy.fft module

**Files:**
- Create: `crates/numpy-rust-core/src/fft.rs`
- Modify: `crates/numpy-rust-core/src/lib.rs`

Behind `fft` feature flag. Implement `fft`, `ifft`, `rfft`, `irfft`.

Convert NdArray to `Vec<Complex<f64>>`, run RustFFT, convert back. `rfft` returns only positive frequencies.

**Commit:** `feat: add numpy.fft backed by RustFFT`

---

## Phase 12: Random (rand)

### Task 17: numpy.random module

**Files:**
- Create: `crates/numpy-rust-core/src/random.rs`
- Modify: `crates/numpy-rust-core/src/lib.rs`

Behind `random` feature flag. Implement:
- `seed(n)` — set global RNG seed
- `rand(shape) -> NdArray` — uniform [0,1)
- `randn(shape) -> NdArray` — standard normal
- `randint(low, high, shape) -> NdArray` — uniform integers
- `normal(mean, std, shape) -> NdArray`
- `uniform(low, high, shape) -> NdArray`
- `choice(a, size, replace) -> NdArray`

Use `rand::rngs::StdRng` with `SeedableRng`. Use `rand_distr` for Normal, Uniform.

**Commit:** `feat: add numpy.random backed by rand`

---

## Phase 13: RustPython Bindings

### Task 18: Set up numpy-rust-python crate

**Files:**
- Create: `crates/numpy-rust-python/Cargo.toml`
- Create: `crates/numpy-rust-python/src/lib.rs`

```toml
[package]
name = "numpy-rust-python"
version = "0.1.0"
edition = "2021"

[dependencies]
numpy-rust-core = { path = "../numpy-rust-core" }
rustpython-vm = "0.4"
```

Implement `#[pymodule] mod numpy` with `#[pyattr]` for the ndarray class and module-level functions.

**Commit:** `chore: scaffold numpy-rust-python crate`

---

### Task 19: PyNdArray class

**Files:**
- Create: `crates/numpy-rust-python/src/py_array.rs`
- Modify: `crates/numpy-rust-python/src/lib.rs`

Implement `#[pyclass] PyNdArray` wrapping `NdArray`:
- Properties: `shape`, `ndim`, `dtype`, `size`, `T`
- Methods: `reshape`, `flatten`, `sum`, `mean`, `min`, `max`, `astype`, `copy`
- Operators: `__add__`, `__sub__`, `__mul__`, `__truediv__`, `__eq__`, `__lt__`, etc.
- `__repr__` and `__str__`
- `__getitem__` for indexing

**Commit:** `feat: add PyNdArray class with operators and methods`

---

### Task 20: Module-level functions and submodules

**Files:**
- Create: `crates/numpy-rust-python/src/py_creation.rs`
- Create: `crates/numpy-rust-python/src/py_linalg.rs`
- Create: `crates/numpy-rust-python/src/py_fft.rs`
- Create: `crates/numpy-rust-python/src/py_random.rs`
- Modify: `crates/numpy-rust-python/src/lib.rs`

Expose: `numpy.array()`, `numpy.zeros()`, `numpy.ones()`, `numpy.arange()`, `numpy.linspace()`, `numpy.eye()`, `numpy.concatenate()`, `numpy.where()`, `numpy.dot()`, `numpy.linalg.*`, `numpy.fft.*`, `numpy.random.*`.

**Commit:** `feat: expose all numpy functions and submodules to Python`

---

## Phase 14: WASM Entry Point

### Task 21: Set up numpy-rust-wasm crate

**Files:**
- Create: `crates/numpy-rust-wasm/Cargo.toml`
- Create: `crates/numpy-rust-wasm/src/main.rs`

Build RustPython interpreter with numpy module registered. Test with a simple Python script:

```python
import numpy as np
a = np.zeros((3, 3))
print(a.shape)
```

Verify it runs on native first, then compile to wasm32-wasip1.

**Commit:** `feat: add WASM entry point with RustPython + numpy`

---

## Phase 15: Vendored NumPy Tests

### Task 22: Vendor and adapt NumPy test subset

**Files:**
- Create: `tests/python/test_array_creation.py`
- Create: `tests/python/test_numeric.py`
- Create: `tests/python/test_indexing.py`
- Create: `tests/python/test_linalg.py`
- Create: `tests/python/conftest.py`

Clone NumPy repo, extract relevant test cases from `numpy/core/tests/`, adapt:
1. Remove tests for unsupported dtypes (complex, uint, datetime, etc.)
2. Remove tests using unsupported features (structured dtypes, ufunc protocol, etc.)
3. Keep tests that exercise our Tier 0/1/2 API surface
4. Replace `numpy.testing` helpers with simple assert-based equivalents

Create a test runner script that uses the numpy-rust-wasm binary to execute each test file.

**Commit:** `test: vendor and adapt NumPy test subset`

---

## Phase 16: CI & Polish

### Task 23: GitHub Actions CI

**Files:**
- Create: `.github/workflows/ci.yml`

Jobs:
- `cargo test` (all crates, native)
- `cargo clippy`
- `cargo fmt --check`
- Build wasm32-wasip1 target
- Run vendored Python tests via RustPython

**Commit:** `ci: add GitHub Actions workflow`

---

## Execution Order Summary

| Phase | Tasks | What it delivers |
|---|---|---|
| 1. Scaffolding | 1-2 | Compiling workspace with DType + errors |
| 2. Core types | 3-4 | NdArray struct, ArrayData enum, type casting |
| 3. Broadcasting | 5 | NumPy-compatible shape broadcasting |
| 4. Arithmetic | 6-8 | +, -, *, /, comparisons, unary math |
| 5. Reductions | 9 | sum, mean, min, max, std, var |
| 6. Creation | 10 | arange, linspace, eye, full |
| 7. Shape | 11 | reshape, transpose, concatenate, stack |
| 8. Indexing | 12-13 | Slicing, boolean masking |
| 9. Utilities | 14 | where, isnan, dot, copy |
| 10. Linalg | 15 | inv, solve, eig, svd, qr (faer) |
| 11. FFT | 16 | fft, ifft, rfft (RustFFT) |
| 12. Random | 17 | rand, randn, normal, uniform |
| 13-14. Python/WASM | 18-21 | RustPython bindings, WASM binary |
| 15. Tests | 22 | Vendored NumPy tests |
| 16. CI | 23 | GitHub Actions |
