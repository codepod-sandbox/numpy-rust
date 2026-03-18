# numpy-rust

A NumPy implementation in Rust for Python code running in sandboxed environments (RustPython/WASM).

**7,705 tests passing (`2026-03-18`)**

## How it works

```
Python code (import numpy as np)
        │
   numpy/ package (Python wrappers + submodules)
        │
   _numpy_native (Rust ← RustPython bindings)
        │
   numpy-rust-core (ndarray, nalgebra, rustfft, rand)
```

All numerical operations run in native Rust. The Python layer handles API surface and dtype routing.

## Test coverage

| Suite | Result |
|---|---|
| `cargo test` | 454 passed |
| Python vendored tests | 1,188 passed |
| NumPy compat (`test_numeric.py`) | 1,206 passed, 8 expected failures |
| NumPy ufunc compat (`test_ufunc.py`) | 106 passed, 344 expected failures |
| NumPy I/O compat (`test_io.py`) | 23 passed, 2 skipped |
| Upstream NumPy tests (74 files, 86K lines) | 4,858 passed, 6 panics, 6 timeouts |

### Upstream test breakdown

74 vendored upstream NumPy test files cover: core array ops, math, indexing, dtypes, scalars, shape manipulation, einsum, array padding, set ops, nan functions, histograms, index tricks, stride tricks, type checking, masked arrays, polynomials, FFT, linalg, random, and more.

```bash
# Run all upstream tests (scan mode)
./target/release/numpy-python tests/numpy_compat/run_upstream.py --scan

# Run a single upstream test file
./target/release/numpy-python tests/numpy_compat/run_upstream.py upstream/core_test_numeric.py
```

---

## Submodules

| Submodule | Notes |
|-----------|-------|
| `np.linalg` | matmul, inv, solve, det, eig, svd, qr, norm, cholesky, lstsq, pinv — via nalgebra |
| `np.fft` | fft/ifft (Rust/rustfft), rfft/irfft, fft2/fftn, fftfreq, fftshift |
| `np.random` | Full distribution set, both legacy and Generator API, SeedSequence |
| `np.ma` | Complete MaskedArray (224 symbols): creation, masking, ufunc wrappers, reductions, manipulation, set ops, statistics |
| `np.testing` | assert_allclose, assert_array_equal, assert_equal, assert_raises, suppress_warnings, temppath |
| `np.polynomial` | Polynomial, Chebyshev, Legendre, Hermite, HermiteE, Laguerre classes with val/fit/add/sub/mul/der/int |
| `np.char` / `np.strings` | 35+ element-wise string operations |
| `np.lib.scimath` | Complex-safe math (sqrt, log, log2, log10, arcsin, arccos) |
| `np.lib.stride_tricks` | as_strided, broadcast_shapes, sliding_window_view |
| `np.dtypes` | DType class stubs for all numeric types |
| `np._core` | Full internal compatibility package (15 submodules) |
| `np.exceptions` | AxisError, ComplexWarning, DTypePromotionError, RankWarning, TooHardError |
| `np._utils` | asbytes, asunicode, Version |

---

## Limitations

### CPython-only features (not implementable in RustPython/WASM)

- **C-extension ufunc machinery.** The low-level strided-loop API, `PyUFunc_OO_O` generic loops, and gufunc signature introspection all require CPython internals. The 344 ufunc xfails are entirely in this category.
- **C-extension custom dtypes** (e.g. `rational`). Requires CPython's type system.
- **`np.memmap`.** Memory-mapped files aren't available in WASM.
- **`nditer`.** The N-dimensional iterator requires CPython C-level iteration protocol.
- **`errstate` with `raise` mode.** FP exception signaling requires OS-level signal handling.

### Architecture limits

- **Slice views are copies.** `arr[1:4]` returns a copy — the `ndarray` crate can't represent sub-array `ArcArray` views. `view()` and `reshape()` on the whole array work in O(1). `shares_memory()` / `may_share_memory()` work correctly via Arc pointer equality.
- **`out=` with slices.** Writes to `clip(out=arr[1:4])` don't propagate back because slices are copies.
- **Complex scalars.** Scalars extracted from complex arrays come back as `(re, im)` tuples — a RustPython limitation.
- **Fortran-order layout.** All arrays are C-contiguous. `order='F'` is accepted but data is always row-major.
- **Long double.** `np.longdouble` maps to `float64` (Rust has no 80-bit float). Same for `np.clongdouble` → `complex128`.

### Performance

- **`einsum`** uses brute-force index iteration — correct but slow for large contractions.
- **`rfft`/`irfft`** are pure Python. `fft`/`ifft` are native Rust via `rustfft`.
- No SIMD or multi-threading. All operations are single-threaded.

### Silently ignored parameters

`casting=`, `subok=` on most ufuncs. `out=` is silently ignored except for in-place operators.

---

## Usage with codepod

This repo is included as a git submodule in codepod at `packages/numpy-rust`. The codepod Python binary links it via a Cargo feature flag:

```bash
cargo build -p codepod-python --features numpy --target wasm32-wasip1
```

## Development

### Setup

```bash
# Install git hooks (formatting + clippy on commit, full tests on push)
./hooks/install.sh
```

### Build and test

```bash
# Build (native)
cargo build --release

# Build (WASM, as used by codepod)
cargo build -p numpy-rust-wasm --target wasm32-wasip1

# Rust unit tests
cargo test --release

# Python integration tests
bash tests/python/run_tests.sh

# NumPy compat tests (tracks known gaps via xfail list)
./target/release/numpy-python tests/numpy_compat/run_compat.py --ci
./target/release/numpy-python tests/numpy_compat/run_ufunc_compat.py --ci
./target/release/numpy-python tests/numpy_compat/run_io_compat.py --ci

# Upstream NumPy tests (74 vendored test files)
./target/release/numpy-python tests/numpy_compat/run_upstream.py --scan
```

### CI

GitHub Actions runs on every push and PR to `main`:

- **Test** — `cargo test` (core + workspace)
- **Lint** — `cargo fmt --check` + `cargo clippy -D warnings`
- **Python Tests** — builds the binary and runs all vendored Python test files
- **WASM Build Check** — verifies the project compiles for `wasm32-wasip1`
