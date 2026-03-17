# numpy-rust

A NumPy 1.26 implementation in Rust for Python code running in sandboxed environments (RustPython/WASM). Covers the full NumPy API — array math, linalg, FFT, random distributions, structured arrays, masked arrays, string operations, and more.

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

All numerical operations run in native Rust — element-wise math (90+ functions via libm), reductions, broadcasting, sorting, linear algebra (nalgebra), FFT (rustfft), and random distributions. The Python layer handles API surface, dtype routing, and a small number of pure-Python fallbacks for non-numeric types.

## Test coverage

| Suite | Result |
|---|---|
| `cargo test` | 454 passed, 0 failed |
| Python vendored tests | 1,320 passed, 0 failed |
| NumPy compat (`test_numeric.py` via RustPython) | 1,211 passed, 3 expected failures |
| NumPy ufunc compat (`test_ufunc.py` via RustPython) | 106 passed, 346 expected failures |

The 3 numeric compat expected failures: out-parameter with overlapping slice memory (1), C-extension custom dtype (1), NaT propagation in clip — a known upstream NumPy bug (1).

The 346 ufunc expected failures are C-extension-only features: low-level strided-loop API, PyUFunc generic loop machinery, gufunc signature introspection, and string ufuncs — none implementable without CPython's C extension infrastructure.

---

## Known limitations

### Not supported

- **Array views for slices.** `arr[1:4]` returns a copy — a limitation of the `ndarray` crate (sub-array `ArcArray` views aren't representable). `view()` on the whole array works in O(1) and shares the underlying buffer; `reshape()` similarly. `shares_memory()` / `may_share_memory()` work correctly via Arc pointer equality.
- **`out=` parameter.** Writes to an output slice (`clip(out=arr[1:4])`) don't propagate back to the original because slices are copies (same root cause as above).
- **Complex scalars.** Scalars extracted from complex arrays come back as `(re, im)` tuples rather than Python `complex` — a RustPython limitation.
- **Binary `.npy`/`.npz` format.** `np.save`/`np.load` use text. Binary format is not implemented.
- **C-extension custom dtypes** (e.g. `rational`). Requires CPython's type system.
- **`np.memmap`.** Memory-mapped files aren't available in WASM.
- **Fortran-order layout.** All arrays are C-contiguous. `order='F'` is accepted and tracked but data is always stored row-major.

### Not efficient

- **`einsum`** uses brute-force index iteration — correct but slow for large contractions.
- **`rfft`/`irfft`** are pure Python. `fft`/`ifft` are native Rust via `rustfft`.
- No SIMD or multi-threading. All operations are single-threaded.

### Parameters silently ignored

`casting=`, `subok=`, `where=` on most ufuncs and array functions. `out=` is silently ignored except for in-place operators.

---

## Submodules

| Submodule | Notes |
|-----------|-------|
| `np.linalg` | matmul, inv, solve, det, eig, svd, qr, norm, cholesky, lstsq, pinv, and more — core ops via nalgebra |
| `np.fft` | fft/ifft (Rust/rustfft), rfft/irfft, fft2/fftn, fftfreq, fftshift |
| `np.random` | Full distribution set (normal, uniform, poisson, …), both legacy and Generator API |
| `np.ma` | MaskedArray with standard creation, indexing, and reduction operations |
| `np.testing` | assert_allclose, assert_array_equal, assert_equal, assert_raises, and friends |
| `np.polynomial` | polyval, polyfit, polyadd, polysub, polymul, polyder, polyint |
| `np.char` | String operation array functions |
| `np.lib.scimath` | Complex-safe math — sqrt, log, arcsin, etc. return complex for out-of-domain inputs |

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
```

### CI

GitHub Actions runs on every push and PR to `main`:

- **Test** — `cargo test` (core + workspace)
- **Lint** — `cargo fmt --check` + `cargo clippy -D warnings`
- **Python Tests** — builds the binary and runs all vendored Python test files
- **WASM Build Check** — verifies the project compiles for `wasm32-wasip1`
