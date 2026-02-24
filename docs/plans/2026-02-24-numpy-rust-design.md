# numpy-rust: NumPy Backend in Rust for WASM/RustPython

**Date**: 2026-02-24
**Status**: Approved

## Goal

Build a NumPy-compatible Python library with a pure Rust backend, targeting WASM via RustPython. No C/C++ dependencies. Users `import numpy` and get a drop-in replacement that works in browser and WASI runtimes.

## Decisions

| Decision | Choice | Rationale |
|---|---|---|
| Test strategy | Vendored NumPy test subset | Copy & adapt NumPy's Python tests; run in RustPython |
| Deploy targets | Browser + WASI (both) | Feature-flagged compilation targets |
| Module name | `import numpy` | Drop-in replacement; registered as built-in module in RustPython |
| Dtype scope (MVP) | f32, f64, i32, i64, bool | 5 types covers most real usage without excessive dispatch code |
| Build order | Rust core lib first, Python bindings later | Validate correctness with Rust tests before adding RustPython complexity |
| Dtype dispatch | Enum-based | Match on `ArrayData` enum variants; macros handle boilerplate; compiler-checked exhaustiveness |

## Architecture

### Workspace Structure

```
numpy-rust/
├── Cargo.toml                    # Workspace root
├── crates/
│   ├── numpy-rust-core/          # Pure Rust array library (no Python deps)
│   ├── numpy-rust-python/        # RustPython bindings (thin layer)
│   └── numpy-rust-wasm/          # WASM entry point (RustPython + native module)
├── tests/
│   ├── rust/                     # Rust-level tests
│   └── python/                   # Vendored NumPy test subset
└── docs/plans/
```

**numpy-rust-core**: Pure Rust, zero Python dependencies, testable with `cargo test`. Contains all array logic.

**numpy-rust-python**: Thin binding layer using RustPython's `#[pymodule]`/`#[pyclass]` macros. Wraps core types as Python objects.

**numpy-rust-wasm**: Entry point that builds RustPython interpreter with the numpy module registered. Compiles to wasm32-unknown-unknown (browser) or wasm32-wasip1 (WASI).

### Core Data Model

```rust
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DType {
    Float32, Float64, Int32, Int64, Bool,
}

pub enum ArrayData {
    Float32(ArrayD<f32>),
    Float64(ArrayD<f64>),
    Int32(ArrayD<i32>),
    Int64(ArrayD<i64>),
    Bool(ArrayD<bool>),
}

pub struct NdArray {
    data: ArrayData,
}
```

### Type Promotion

Follows NumPy's simplified rules for 5 types:

```
bool -> i32 -> i64 -> f64
               |
               v
              f32 -> f64
```

Binary operations promote both operands to their common type first, then dispatch on the (now matching) type. This reduces binary dispatch from 25 arms to 5.

### Broadcasting

Implemented as a standalone function following NumPy's rules:
1. Pad shorter shape with 1s on the left
2. For each dimension: sizes must be equal or one must be 1
3. Output size is the max of the two

Leverages `ndarray`'s `.broadcast(shape)` for zero-copy views via stride manipulation.

### Dispatch Macro

```rust
macro_rules! dispatch_binary {
    ($lhs:expr, $rhs:expr, $op:ident) => {
        match promote_pair($lhs, $rhs) {
            (ArrayData::Float64(a), ArrayData::Float64(b)) => ArrayData::Float64(a.$op(&b)),
            (ArrayData::Float32(a), ArrayData::Float32(b)) => ArrayData::Float32(a.$op(&b)),
            (ArrayData::Int64(a), ArrayData::Int64(b)) => ArrayData::Int64(a.$op(&b)),
            (ArrayData::Int32(a), ArrayData::Int32(b)) => ArrayData::Int32(a.$op(&b)),
            (ArrayData::Bool(a), ArrayData::Bool(b)) => ArrayData::Bool(a.$op(&b)),
            _ => unreachable!("promotion ensures matching types"),
        }
    };
}
```

## Dependencies

| Crate | Purpose | WASM Compatible |
|---|---|---|
| `ndarray` (no `blas` feature) | N-dimensional array container | Yes |
| `faer` | Linear algebra (pure Rust LAPACK replacement) | Yes |
| `rustfft` | FFT (has `wasm_simd` feature) | Yes |
| `rand` + `rand_distr` | Random number generation | Yes |
| `num-traits` | Generic numeric traits | Yes |
| `thiserror` | Error types | Yes |
| `rustpython-vm` (python crate only) | RustPython integration | Yes |

All dependencies are pure Rust. No C/C++ transitive dependencies.

## API Surface (MVP)

### Tier 0 - Core (must-have)

**Creation**: `array()`, `zeros()`, `ones()`, `empty()`, `arange()`, `linspace()`, `eye()`, `full()`, `zeros_like()`, `ones_like()`

**Attributes**: `shape`, `ndim`, `dtype`, `size`, `T`

**Indexing**: Basic slicing, integer indexing, boolean masking

**Arithmetic**: `+`, `-`, `*`, `/`, `//`, `%`, `**` (element-wise with broadcasting)

**Comparison**: `==`, `!=`, `<`, `>`, `<=`, `>=`

**Math**: `abs`, `sqrt`, `exp`, `log`, `sin`, `cos`, `tan`

**Reductions**: `sum`, `mean`, `min`, `max`, `std`, `var`, `argmin`, `argmax`, `all`, `any`

**Shape**: `reshape`, `transpose`, `flatten`, `ravel`, `concatenate`, `stack`, `vstack`, `hstack`

**Utility**: `where`, `isnan`, `isfinite`, `dot`, `copy`, `astype`

### Tier 1 - Extended

**Math**: full trig, `floor`, `ceil`, `round`, `clip`, `cumsum`, `cumprod`, `diff`, `sort`, `argsort`, `unique`

**Random** (`numpy.random`): `seed`, `rand`, `randn`, `randint`, `normal`, `uniform`, `choice`

### Tier 2 - Scientific

**Linear algebra** (`numpy.linalg`): `norm`, `inv`, `solve`, `det`, `eig`, `svd`, `qr`, `cholesky`, `matmul`/`@`

**FFT** (`numpy.fft`): `fft`, `ifft`, `fft2`, `rfft`, `irfft`

## Test Strategy

1. **Rust tests**: Unit tests in each core module, integration tests in `tests/rust/`
2. **Vendored NumPy tests**: Copy relevant test files from `numpy/core/tests/` (MIT-licensed), adapt minimally (remove tests for unsupported features). Priority files:
   - `test_multiarray.py` (creation, shape)
   - `test_numeric.py` (arithmetic, math)
   - `test_indexing.py` (slicing, fancy indexing)
   - `test_umath.py` (element-wise ops)
   - `test_linalg.py` (linear algebra)
3. **Runner**: `cargo test` for Rust, RustPython for Python tests

## WASM Constraints

- Single-threaded execution (no parallelism)
- 4 GB memory ceiling (wasm32)
- ~25-40 MB binary size (RustPython + frozen stdlib + numpy module)
- WASM SIMD (128-bit) available via `target-feature=+simd128`
- No C extension compatibility (SciPy, pandas won't work)

## Build Commands

```bash
# Native Rust tests
cargo test -p numpy-rust-core

# Browser WASM
cargo build -p numpy-rust-wasm --target wasm32-unknown-unknown --release

# WASI WASM
cargo build -p numpy-rust-wasm --target wasm32-wasip1 --release
```
