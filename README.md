# numpy-rust

A NumPy implementation in Rust, compiled to WebAssembly. Provides core ndarray operations — array creation, slicing, broadcasting, linear algebra, FFT, and random number generation — for Python code running inside [wasmsand](https://github.com/sunnymar/wasmsand).

## How it works

```
Python code (import numpy as np)
        │
   numpy/ package (Python shims + dtype stubs)
        │
   _numpy_native (Rust ← RustPython bindings)
        │
   numpy-rust-core (ndarray, nalgebra, rustfft, rand)
```

The Rust core (`numpy-rust-core`) implements n-dimensional arrays on top of the `ndarray` crate. The Python bindings (`numpy-rust-python`) expose these as a native RustPython module. A thin Python package (`python/numpy/`) provides the familiar `import numpy as np` interface with dtype aliases, constants, and stub functions for API coverage.

## Crates

| Crate | Description |
|---|---|
| `numpy-rust-core` | Core ndarray implementation (dtypes, indexing, slicing, math, broadcasting) |
| `numpy-rust-python` | RustPython bindings — exposes `_numpy_native` module |
| `numpy-rust-wasm` | Binary entry point (RustPython + numpy, compiles to WASI) |

## Supported operations

- **Array creation**: `array`, `zeros`, `ones`, `arange`, `linspace`, `eye`
- **Indexing**: integer, multi-dimensional, boolean masks, slicing (`a[1:4]`, `a[::2]`, `a[::-1]`)
- **Shape**: `reshape`, `flatten`, `ravel`, `transpose`
- **Math**: element-wise arithmetic, `sum`, `mean`, `std`, `var`, `min`, `max`, `abs`, `sqrt`
- **Comparison**: `==`, `!=`, `<`, `<=`, `>`, `>=` (returns boolean arrays)
- **Linear algebra** (feature-gated): `dot`, `matmul`, `inv`, `det`, `eig`, `svd`, `solve`, `norm`
- **FFT** (feature-gated): `fft`, `ifft`, `fft2`, `ifft2`, `fftfreq`
- **Random** (feature-gated): `random`, `rand`, `randn`, `randint`, `uniform`, `normal`, `seed`

## Usage with wasmsand

This repo is included as a git submodule in wasmsand at `packages/numpy-rust`. The wasmsand Python binary links it via a Cargo feature flag:

```bash
cargo build -p wasmsand-python --features numpy --target wasm32-wasip1
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
cargo build -p numpy-rust-wasm

# Build (WASM, as used by wasmsand)
cargo build -p numpy-rust-wasm --target wasm32-wasip1

# Rust unit tests
cargo test --workspace --all-features

# Python integration tests
cargo run -p numpy-rust-wasm -- tests/python/test_indexing.py
cargo run -p numpy-rust-wasm -- tests/python/test_numeric.py

# All Python tests
./tests/python/run_tests.sh target/debug/numpy-python
```

### CI

GitHub Actions runs on every push and PR to `main`:

- **Test** — `cargo test` (core + workspace)
- **Lint** — `cargo fmt --check` + `cargo clippy -D warnings`
- **Python Tests** — builds the binary and runs all vendored Python test files
- **WASM Build Check** — verifies the project compiles for `wasm32-wasip1`
