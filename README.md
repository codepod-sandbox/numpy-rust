# numpy-rust

A NumPy 1.26 implementation in Rust for Python code running in sandboxed environments (RustPython/WASM).

**3,091 tests, 0 failures (`2026-03-17`)**

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
| `cargo test` | 454 passed, 0 failed |
| Python vendored tests | 1,320 passed, 0 failed |
| NumPy compat (`test_numeric.py` via RustPython) | 1,211 passed, 3 expected failures |
| NumPy ufunc compat (`test_ufunc.py` via RustPython) | 106 passed, 344 expected failures |

---

## Limitations

### CPython-only features (not implementable in RustPython/WASM)

- **C-extension ufunc machinery.** The low-level strided-loop API, `PyUFunc_OO_O` generic loops, and gufunc signature introspection all require CPython internals. The 344 ufunc xfails are entirely in this category.
- **C-extension custom dtypes** (e.g. `rational`). Requires CPython's type system.
- **`np.memmap`.** Memory-mapped files aren't available in WASM.
- **String ufuncs.** `np.char.find` and similar as ufuncs require CPython string buffer internals.
- **`errstate` with `raise` mode.** FP exception signaling requires OS-level signal handling.

### Architecture limits

- **Slice views are copies.** `arr[1:4]` returns a copy — the `ndarray` crate can't represent sub-array `ArcArray` views. `view()` and `reshape()` on the whole array work in O(1). `shares_memory()` / `may_share_memory()` work correctly via Arc pointer equality.
- **`out=` with slices.** Writes to `clip(out=arr[1:4])` don't propagate back because slices are copies.
- **Complex scalars.** Scalars extracted from complex arrays come back as `(re, im)` tuples — a RustPython limitation.
- **Binary `.npy`/`.npz`.** `np.save`/`np.load` use text. Binary format not implemented.
- **Fortran-order layout.** All arrays are C-contiguous. `order='F'` is accepted but data is always row-major.

### Performance

- **`einsum`** uses brute-force index iteration — correct but slow for large contractions.
- **`rfft`/`irfft`** are pure Python. `fft`/`ifft` are native Rust via `rustfft`.
- No SIMD or multi-threading. All operations are single-threaded.

### Silently ignored parameters

`casting=`, `subok=` on most ufuncs. `out=` is silently ignored except for in-place operators.

---

## Submodules

| Submodule | Notes |
|-----------|-------|
| `np.linalg` | matmul, inv, solve, det, eig, svd, qr, norm, cholesky, lstsq, pinv — via nalgebra |
| `np.fft` | fft/ifft (Rust/rustfft), rfft/irfft, fft2/fftn, fftfreq, fftshift |
| `np.random` | Full distribution set, both legacy and Generator API |
| `np.ma` | MaskedArray with creation, indexing, and reduction operations |
| `np.testing` | assert_allclose, assert_array_equal, assert_equal, assert_raises, and friends |
| `np.polynomial` | polyval, polyfit, polyadd, polysub, polymul, polyder, polyint |
| `np.char` | String operation array functions |
| `np.lib.scimath` | Complex-safe math (sqrt, log, arcsin, etc.) |

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
```

### CI

GitHub Actions runs on every push and PR to `main`:

- **Test** — `cargo test` (core + workspace)
- **Lint** — `cargo fmt --check` + `cargo clippy -D warnings`
- **Python Tests** — builds the binary and runs all vendored Python test files
- **WASM Build Check** — verifies the project compiles for `wasm32-wasip1`
