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

The Rust core (`numpy-rust-core`) implements n-dimensional arrays on top of the `ndarray` crate. The Python bindings (`numpy-rust-python`) expose these as a native RustPython module. A thin Python package (`python/numpy/`) provides the familiar `import numpy as np` interface with dtype aliases, constants, and wrapper functions for API coverage.

## Crates

| Crate | Description |
|---|---|
| `numpy-rust-core` | Core ndarray implementation (dtypes, indexing, slicing, math, broadcasting) |
| `numpy-rust-python` | RustPython bindings — exposes `_numpy_native` module |
| `numpy-rust-wasm` | Binary entry point (RustPython + numpy, compiles to WASI) |

## Supported features

### Data types

| Type | Aliases | Notes |
|------|---------|-------|
| `bool` | `bool_` | |
| `int32` | `intc` | |
| `int64` | `int_`, `intp` | Default integer type |
| `float32` | `single` | |
| `float64` | `double`, `float_` | Default float type |
| `complex64` | `csingle` | |
| `complex128` | `cdouble` | |
| `str` | `str_`, `unicode_` | Variable-length strings |

Other dtype aliases (`float16`, `uint8`, `uint16`, etc.) are accepted for API compatibility but are stored internally as the nearest supported type.

### Array creation

| Function | Status | Notes |
|----------|--------|-------|
| `np.array(data)` | Full | Nested lists, mixed int/float |
| `np.zeros(shape)` | Full | |
| `np.ones(shape)` | Full | |
| `np.full(shape, value)` | Full | |
| `np.arange(start, stop, step)` | Full | `dtype` param supported in core |
| `np.linspace(start, stop, num)` | Full | `retstep=True` supported |
| `np.eye(N, M, k)` | Full | Rectangular + diagonal offset |
| `np.zeros_like(a)` | Full | |
| `np.ones_like(a)` | Full | |
| `np.full_like(a, value)` | Full | |
| `np.copy(a)` | Full | |
| `np.asarray(a)` | Full | |
| `np.fromiter(iterable, dtype)` | Full | |
| `np.empty(shape)` | Stub | Returns zeros |
| `np.empty_like(a)` | Stub | Returns zeros |

### Indexing and slicing

| Feature | Status | Example |
|---------|--------|---------|
| Integer indexing | Full | `a[0]`, `a[1, 2]` |
| Slice indexing | Full | `a[1:4]`, `a[::2]`, `a[::-1]` |
| Boolean mask indexing | Full | `a[a > 3]` |
| Fancy indexing (list) | Full | `a[[0, 2, 4]]` |
| Fancy indexing (array) | Full | `a[np.array([1, 3])]` |
| Assignment | Full | `a[0] = 5`, `a[[0, 2]] = arr` |

### Arithmetic operators

All operators support broadcasting and in-place variants.

| Operator | In-place | Notes |
|----------|----------|-------|
| `+` `-` `*` `/` | `+=` `-=` `*=` `/=` | Element-wise |
| `**` | `**=` | Power |
| `//` | `//=` | Floor division (toward -inf, NumPy semantics) |
| `%` | `%=` | Modulo (sign of divisor, NumPy semantics) |
| `@` | `@=` | Matrix multiplication |
| `-a` | | Unary negation |

### Comparison operators

`==`, `!=`, `<`, `<=`, `>`, `>=` — all return boolean arrays and support broadcasting.

### Bitwise / logical operators

| Operator | Notes |
|----------|-------|
| `&` (and) | Works on boolean arrays |
| `\|` (or) | Works on boolean arrays |
| `~` (not) | Works on boolean arrays |

### Reduction operations

All reductions support `axis` and `keepdims` parameters.

| Function | Method | Notes |
|----------|--------|-------|
| `np.sum(a)` | `a.sum()` | |
| `np.mean(a)` | `a.mean()` | Returns Float64 or Complex128 |
| `np.min(a)` | `a.min()` | Not supported for complex |
| `np.max(a)` | `a.max()` | Not supported for complex |
| `np.std(a, ddof=0)` | `a.std()` | `ddof` for sample std |
| `np.var(a, ddof=0)` | `a.var()` | `ddof` for sample var |
| `np.argmin(a)` | `a.argmin()` | Supports `axis` parameter |
| `np.argmax(a)` | `a.argmax()` | Supports `axis` parameter |
| `a.all()` | | Boolean test |
| `a.any()` | | Boolean test |
| `np.prod(a)` | | Python-level implementation |

### Element-wise math

Available as both module functions (`np.sqrt(a)`) and array methods (`a.sqrt()`).

| Function | Notes |
|----------|-------|
| `abs` / `absolute` | Returns magnitude for complex |
| `sqrt` | |
| `exp` | |
| `log` | Natural logarithm |
| `sin`, `cos`, `tan` | |
| `floor`, `ceil`, `round` | Not supported for complex |
| `clip(a, min, max)` | |

### Shape manipulation

| Function | Method | Notes |
|----------|--------|-------|
| `np.reshape(a, shape)` | `a.reshape(shape)` | |
| | `a.flatten()` | |
| `np.ravel(a)` | `a.ravel()` | |
| `np.squeeze(a, axis)` | `a.squeeze(axis)` | |
| `np.expand_dims(a, axis)` | `a.expand_dims(axis)` | |
| `np.transpose(a)` | `a.T` | |
| `np.concatenate(arrays, axis)` | | |
| `np.stack(arrays, axis)` | | |
| `np.vstack(arrays)` | | |
| `np.hstack(arrays)` | | |

### Sorting and searching

| Function | Method | Notes |
|----------|--------|-------|
| `np.sort(a, axis)` | `a.sort(axis)` | Returns new sorted array |
| `np.argsort(a, axis)` | `a.argsort(axis)` | Returns Int64 indices |
| `np.searchsorted(a, v, side)` | | Binary search, `"left"` / `"right"` |
| `np.where(cond, x, y)` | | |
| `np.nonzero(a)` | | |
| `np.compress(cond, a, axis)` | | |
| `np.choose(a, choices)` | | |

### Complex number support

| Feature | Notes |
|---------|-------|
| `a.real` | Property — real part |
| `a.imag` | Property — imaginary part |
| `a.conj()` / `np.conj(a)` | Complex conjugate |
| `np.angle(a)` | Phase angle (radians) |
| Arithmetic (`+`, `-`, `*`, `/`, `**`) | Full support |
| `abs(a)` | Returns float magnitude |
| `sum`, `mean` | Work on complex arrays |
| `exp`, `sqrt`, `log` | Work on complex arrays |
| `==`, `!=` | Work on complex arrays |

Operations that don't support complex (raise `TypeError`): `min`, `max`, `std`, `var`, `<`, `<=`, `>`, `>=`, `//`, `%`, `floor`, `ceil`, `round`, `sort`, `argsort`, bitwise ops.

### String operations (`np.char`)

| Function | Notes |
|----------|-------|
| `np.char.upper(a)` | |
| `np.char.lower(a)` | |
| `np.char.capitalize(a)` | |
| `np.char.strip(a)` | |
| `np.char.str_len(a)` | Unicode character count |
| `np.char.startswith(a, prefix)` | Returns bool array |
| `np.char.endswith(a, suffix)` | Returns bool array |
| `np.char.replace(a, old, new)` | |

### Einstein summation

`np.einsum(subscripts, *operands)` — supports explicit subscript notation (`"ij,jk->ik"`).

Works for matrix multiplication, trace, transpose, outer products, and general contractions. All operands are cast to Float64.

### Linear algebra (`np.linalg`) — feature-gated

| Function | Notes |
|----------|-------|
| `np.dot(a, b)` | Dot product / matmul |
| `np.linalg.matmul(a, b)` | Matrix multiplication |
| `np.linalg.inv(a)` | Matrix inverse |
| `np.linalg.solve(a, b)` | Solve Ax = b |
| `np.linalg.det(a)` | Determinant |
| `np.linalg.eig(a)` | Eigenvalues + eigenvectors |
| `np.linalg.svd(a)` | Singular value decomposition |
| `np.linalg.qr(a)` | QR decomposition |
| `np.linalg.norm(a)` | Matrix/vector norm |
| `np.linalg.cholesky(a)` | Cholesky decomposition |

### FFT (`np.fft`) — feature-gated

| Function | Notes |
|----------|-------|
| `np.fft.fft(a)` | 1-D FFT |
| `np.fft.ifft(a)` | 1-D inverse FFT |
| `np.fft.rfft(a)` | 1-D real FFT |
| `np.fft.irfft(a, n)` | 1-D inverse real FFT |
| `np.fft.fftfreq(n, d)` | Frequency bins |

### Random (`np.random`) — feature-gated

| Function | Notes |
|----------|-------|
| `np.random.seed(n)` | Set RNG seed |
| `np.random.rand(*shape)` | Uniform [0, 1) |
| `np.random.randn(*shape)` | Standard normal |
| `np.random.randint(low, high, size)` | Random integers |
| `np.random.normal(loc, scale, size)` | Normal distribution |
| `np.random.uniform(low, high, size)` | Uniform distribution |
| `np.random.choice(a, size, replace)` | Random selection |

### Constants

`np.pi`, `np.e`, `np.inf`, `np.nan`, `np.PINF`, `np.NINF`, `np.PZERO`, `np.NZERO`, `np.newaxis`, `np.True_`, `np.False_`

### Type introspection

`np.iinfo(dtype)` — integer type info (min, max, bits)
`np.finfo(dtype)` — float type info (eps, max, min, tiny, resolution)

## Known limitations

### Dtype handling
- Most functions ignore the `dtype` parameter and operate in Float64 internally. `astype()` works for explicit conversion.
- `result_type()` and `promote_types()` are stubs that return `float64`.
- Integer arithmetic may promote to Float64 unexpectedly.

### Parameters accepted but ignored
- `order` (memory layout) — always C-contiguous.
- `out` (output array) — no in-place output support for functions.
- `casting`, `subok`, `where` — silently ignored on most functions.
- `endpoint=False` on `linspace` — endpoint is always included.

### Complex numbers
- Scalars extracted from complex arrays are returned as `(re, im)` tuples, not Python `complex` objects (RustPython limitation).
- `var`/`std` reject complex inputs; use `np.abs(a)` first.

### einsum performance
- Uses brute-force iteration over all index combinations. Fine for small-to-medium arrays; will be slow for large contractions. Only explicit subscript notation supported (no implicit or `->` omission).

### Not implemented
- **Structured / record arrays** — no compound dtypes.
- **Datetime / timedelta** — dtype aliases exist but no operations.
- **Advanced fancy indexing** — no multi-axis fancy indexing (`a[[0,1], [2,3]]`).
- **np.ufunc** — no universal function protocol.
- **np.nditer** — no multi-array iterator.
- **np.memmap** — no memory-mapped files.
- **np.polynomial** — no polynomial module.
- **np.ma** (masked arrays) — not supported.
- **Fortran-order arrays** — everything is C-order.

### Stubs (accept calls but return approximate/incorrect results)
- `empty()` / `empty_like()` — return zeros instead of uninitialized memory.
- `seterr()` / `geterr()` / `errstate()` — no-ops.
- `may_share_memory()` / `shares_memory()` — always return False.
- `moveaxis()` / `rollaxis()` — return input unchanged.
- `logical_and()` / `logical_or()` — use arithmetic approximation.

## Test coverage

| Suite | Tests | Description |
|-------|-------|-------------|
| Rust unit tests | 229 | Core operations, dtypes, math, sorting, einsum, strings |
| Python integration | 180 | End-to-end through RustPython (4 test files) |
| NumPy compat (reference) | 256 | Official NumPy `test_numeric.py` adapted for pytest |

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
./tests/python/run_tests.sh target/debug/numpy-python

# NumPy compat tests (requires CPython + pytest + numpy)
uv venv .venv && source .venv/bin/activate
uv pip install pytest numpy
python -m pytest tests/numpy_compat/ -q
```

### CI

GitHub Actions runs on every push and PR to `main`:

- **Test** — `cargo test` (core + workspace)
- **Lint** — `cargo fmt --check` + `cargo clippy -D warnings`
- **Python Tests** — builds the binary and runs all vendored Python test files
- **WASM Build Check** — verifies the project compiles for `wasm32-wasip1`
