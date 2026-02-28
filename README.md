# numpy-rust

A NumPy implementation in Rust, compiled to WebAssembly. Provides ~95% of the NumPy API surface commonly used in data science and ML code — array creation, manipulation, linear algebra, FFT, random distributions, masked arrays, and more — for Python code running inside sandboxed environments.

**425 Rust + 877 Python = 1,302 tests, 0 failures**

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

The Rust core (`numpy-rust-core`) implements n-dimensional arrays on top of the `ndarray` crate with support for 8 dtypes, broadcasting, and 400+ operations. The Python bindings (`numpy-rust-python`) expose these as a native RustPython module with 77 ndarray methods. A Python package (`python/numpy/`) provides the familiar `import numpy as np` interface with ~420 functions across 8 submodules.

## Crates

| Crate | Description |
|---|---|
| `numpy-rust-core` | Core ndarray implementation (dtypes, broadcasting, math, linalg, FFT, random) |
| `numpy-rust-python` | RustPython bindings — exposes `_numpy_native` module |
| `numpy-rust-wasm` | Binary entry point (RustPython + numpy, compiles to WASI) |

---

## Supported features

### Data types

| Type | Aliases | Storage |
|------|---------|---------|
| `bool` | `bool_` | Native bool |
| `int32` | `intc` | Native i32 |
| `int64` | `int_`, `intp` | Native i64 (default integer) |
| `float32` | `single` | Native f32 |
| `float64` | `double`, `float_` | Native f64 (default float) |
| `complex64` | `csingle` | Native (f32, f32) |
| `complex128` | `cdouble` | Native (f64, f64) |
| `str` | `str_`, `unicode_` | Variable-length strings |

Additional dtype aliases (`float16`, `int8`, `int16`, `uint8`, `uint16`, `uint32`, `uint64`) are accepted and stored as the nearest native type. `iinfo()` and `finfo()` provide type introspection for all numeric types.

Full type hierarchy for `issubdtype` checks: `generic > number > integer/inexact > signedinteger/unsignedinteger/floating/complexfloating` with concrete types (`int32`, `float64`, etc.) as leaves.

### Array creation (17 functions)

| Function | Notes |
|----------|-------|
| `array`, `asarray`, `ascontiguousarray`, `asarray_chkfinite` | From data/lists |
| `zeros`, `ones`, `full`, `empty` | Constant fill (`empty` returns zeros) |
| `zeros_like`, `ones_like`, `full_like`, `empty_like` | Match shape/dtype |
| `arange` | Evenly spaced values with step |
| `linspace`, `logspace`, `geomspace` | Linear/log/geometric spacing, `endpoint` and `dtype` params |
| `eye`, `identity` | Identity matrices with offset |
| `fromiter`, `fromstring`, `fromfunction`, `frombuffer` | From iterables/strings/callables/bytes |
| `copy` | Deep copy |

### Indexing and slicing

| Feature | Example |
|---------|---------|
| Integer indexing | `a[0]`, `a[1, 2]` |
| Slice indexing | `a[1:4]`, `a[::2]`, `a[::-1]` |
| Boolean mask indexing | `a[a > 3]` |
| Fancy indexing (list/array) | `a[[0, 2, 4]]`, `a[np.array([1, 3])]` |
| Assignment | `a[0] = 5`, `a[a > 0] = 0`, `a[[0, 2]] = arr` |
| Ellipsis | `a[..., 0]` |

### Operators

All operators support broadcasting and in-place variants (`+=`, `-=`, etc.).

| Category | Operators |
|----------|-----------|
| Arithmetic | `+` `-` `*` `/` `**` `//` `%` `@` (matmul) |
| Comparison | `==` `!=` `<` `<=` `>` `>=` (return bool arrays) |
| Bitwise/Logical | `&` `\|` `^` `~` (bool and int arrays) |
| Unary | `-a`, `+a`, `abs(a)` |

### Reduction operations (31 functions)

All reductions support `axis` (including tuple of axes), `keepdims`, and `ddof` where applicable.

| Category | Functions |
|----------|-----------|
| Basic | `sum`, `prod`, `cumsum`, `cumprod` |
| Statistics | `mean`, `std`, `var`, `median`, `average` |
| Extrema | `min`, `max`, `argmin`, `argmax`, `ptp` |
| Logical | `all`, `any` |
| NaN-safe | `nansum`, `nanprod`, `nanmean`, `nanstd`, `nanvar`, `nanmin`, `nanmax`, `nanargmin`, `nanargmax`, `nancumsum`, `nancumprod` |
| Quantiles | `quantile`, `percentile`, `nanquantile`, `nanpercentile`, `nanmedian` |

### Element-wise math (70+ functions)

Available as both `np.func(a)` and `a.func()` methods.

| Category | Functions |
|----------|-----------|
| Trigonometric | `sin`, `cos`, `tan`, `arcsin`, `arccos`, `arctan`, `arctan2`, `hypot` |
| Hyperbolic | `sinh`, `cosh`, `tanh`, `arcsinh`, `arccosh`, `arctanh` |
| Exponential/Log | `exp`, `exp2`, `expm1`, `log`, `log2`, `log10`, `log1p`, `logaddexp`, `logaddexp2` |
| Rounding | `floor`, `ceil`, `round`, `around`, `rint`, `trunc`, `fix` |
| Algebraic | `sqrt`, `cbrt`, `square`, `abs`, `fabs`, `sign`, `reciprocal` |
| Comparison | `maximum`, `minimum`, `fmax`, `fmin`, `clip` |
| Complex | `real`, `imag`, `conj`, `angle`, `real_if_close` |
| Angular | `deg2rad`, `rad2deg`, `unwrap`, `sinc` |
| Bitwise | `bitwise_and`, `bitwise_or`, `bitwise_xor`, `bitwise_not`, `left_shift`, `right_shift` |
| Logical | `logical_and`, `logical_or`, `logical_xor`, `logical_not` |
| Special | `copysign`, `heaviside`, `nextafter`, `spacing`, `modf`, `divmod_`, `gcd`, `lcm` |
| Floating | `isnan`, `isinf`, `isfinite`, `signbit`, `nan_to_num` |

### Shape manipulation (40+ functions)

| Category | Functions |
|----------|-----------|
| Reshape | `reshape`, `flatten`, `ravel`, `squeeze`, `expand_dims` |
| Transpose | `transpose`, `.T`, `swapaxes`, `moveaxis`, `rollaxis` |
| Join | `concatenate`, `stack`, `vstack`, `hstack`, `dstack`, `column_stack`, `block`, `append` |
| Split | `split`, `vsplit`, `hsplit`, `dsplit`, `array_split` |
| Repeat | `tile`, `repeat`, `resize` |
| Rearrange | `roll`, `flip`, `flipud`, `fliplr`, `rot90` |
| Dimension | `atleast_1d`, `atleast_2d`, `atleast_3d`, `broadcast_to`, `broadcast_arrays`, `broadcast_shapes` |
| Selection | `take`, `put`, `putmask`, `place`, `extract`, `compress`, `choose`, `select`, `piecewise` |
| Insert/Delete | `insert`, `delete`, `trim_zeros` |
| Padding | `pad` (modes: constant, edge, reflect, wrap, symmetric, linear_ramp, mean, median, minimum, maximum) |

### Sorting and searching (12 functions)

| Function | Notes |
|----------|-------|
| `sort`, `argsort` | Axis support, returns new array / indices |
| `partition`, `argpartition` | Partial sorting |
| `searchsorted` | Binary search (`"left"` / `"right"`) |
| `lexsort` | Multi-key sorting |
| `nonzero`, `argwhere`, `flatnonzero` | Element location |
| `count_nonzero` | Count non-zeros |
| `where` | Conditional selection |
| `digitize` | Bin indices |

### Set operations

`unique`, `intersect1d`, `union1d`, `setdiff1d`, `setxor1d`, `isin`, `in1d`

### Numerical utilities

| Function | Notes |
|----------|-------|
| `interp` | 1D linear interpolation with `left`/`right` boundary params |
| `gradient` | Numerical gradient, multi-axis, variable spacing |
| `trapz`/`trapezoid` | Trapezoidal integration, nD support |
| `convolve`, `correlate` | 1D convolution/correlation |
| `diff`, `ediff1d` | Discrete differences |
| `histogram`, `histogram2d`, `histogramdd` | Binning with weights/range |
| `bincount` | Integer bin counting |
| `cov`, `corrcoef` | Covariance/correlation matrices |
| `cross`, `outer`, `inner`, `kron`, `tensordot` | Vector/tensor products |
| `einsum` | Einstein summation (explicit subscripts) |
| `dot`, `vdot`, `matmul` | Dot products and matrix multiplication |
| `take_along_axis`, `put_along_axis` | Advanced axis indexing (nD) |
| `apply_along_axis`, `apply_over_axes` | Apply function along axes |

### Grid and index utilities

`meshgrid`, `mgrid`, `ogrid` (including complex step `5j`), `indices`, `ix_`, `diag_indices`, `diag_indices_from`, `tril_indices`, `triu_indices`, `ravel_multi_index`, `unravel_index`, `ndindex`, `ndenumerate`, `nditer`

### Matrix utilities

`diag`, `diagonal`, `trace`, `tri`, `tril`, `triu`, `vander`, `fill_diagonal`, `identity`, `eye`

### Window functions

`bartlett`, `blackman`, `hamming`, `hanning`, `kaiser`

### I/O

| Function | Notes |
|----------|-------|
| `loadtxt`, `savetxt` | Text format |
| `genfromtxt` | Flexible text loading with missing value handling |
| `save`, `load` | Text-based (not binary `.npy`) |
| `savez` | Compressed text archives |

### Constants

`pi`, `e`, `inf`, `nan`, `PINF`, `NINF`, `PZERO`, `NZERO`, `newaxis`, `True_`, `False_`, `__version__` (reports `"1.26.0"`)

### Special classes

| Class | Notes |
|-------|-------|
| `dtype` | Full dtype class with `name`, `kind`, `itemsize`, `char`, `str`, `fields`, `type` |
| `AxisError` | Subclass of ValueError + IndexError |
| `matrix` | Deprecated-style matrix class (string parsing, `*` for matmul, `.T`, `.I`) |
| `poly1d` | Polynomial class |
| `vectorize` | Function vectorization wrapper |
| `broadcast` | Broadcasting iterator |
| `nditer` | N-dimensional iterator with `multi_index` |
| `iinfo`, `finfo` | Integer/float type introspection |
| `errstate` | Error state context manager (no-op) |

---

## Submodules

### Linear algebra (`np.linalg`)

| Function | Implementation | Notes |
|----------|---------------|-------|
| `matmul` | Rust (nalgebra) | Matrix multiplication |
| `inv` | Rust (nalgebra) | Matrix inverse |
| `solve` | Rust (nalgebra) | Solve Ax = b |
| `det` | Rust (nalgebra) | Determinant |
| `eig` | Rust (nalgebra) | Eigenvalues + eigenvectors |
| `svd` | Rust (nalgebra) | Singular value decomposition |
| `qr` | Rust (nalgebra) | QR decomposition |
| `norm` | Rust (nalgebra) | Matrix/vector norm |
| `cholesky` | Rust (nalgebra) | Cholesky decomposition |
| `lstsq` | Rust (nalgebra) | Least squares |
| `pinv` | Python | Pseudoinverse via SVD |
| `matrix_rank` | Python | Via SVD |
| `matrix_power` | Python | Repeated matmul |
| `eigh`, `eigvals`, `eigvalsh` | Python | Symmetric/Hermitian eigenproblems |
| `slogdet` | Python | Sign + log-determinant |
| `cond` | Python | Condition number |
| `multi_dot` | Python | Chained multiplication |
| `trace` | Python | Diagonal sum |

### FFT (`np.fft`)

| Function | Implementation | Notes |
|----------|---------------|-------|
| `fft`, `ifft` | Rust (rustfft) | 1D FFT/IFFT |
| `rfft`, `irfft` | Python | Real FFT (1D only) |
| `fft2`, `ifft2` | Python | 2D FFT via iterated 1D |
| `fftn`, `ifftn` | Python | N-D FFT |
| `fftfreq`, `rfftfreq` | Python | Frequency bins |
| `fftshift`, `ifftshift` | Python | Frequency reordering |

### Random (`np.random`)

Both legacy API (`np.random.func()`) and new Generator API (`np.random.default_rng().func()`).

| Category | Functions |
|----------|-----------|
| Basic | `random`, `rand`, `randn`, `randint`, `seed` |
| Sampling | `choice`, `shuffle`, `permutation` |
| Continuous | `normal`, `uniform`, `exponential`, `beta`, `gamma`, `lognormal`, `laplace`, `rayleigh`, `weibull`, `logistic`, `gumbel`, `triangular`, `chisquare`, `vonmises`, `wald`, `power` |
| Discrete | `poisson`, `binomial`, `geometric`, `negative_binomial`, `zipf` |
| Multivariate | `multinomial`, `multivariate_normal`, `dirichlet` |
| Generator | `default_rng()` returns `Generator` with `integers`, `random`, `standard_normal`, and all distribution methods |

All distribution functions return scalars when `size=None` and arrays when `size` is specified.

### Masked arrays (`np.ma`)

| Feature | Notes |
|---------|-------|
| `MaskedArray` class | `data`, `mask`, `fill_value`, `shape`, `ndim`, `size`, `dtype` properties |
| Methods | `filled()`, `compressed()`, `count()`, `sum()`, `mean()`, `tolist()` |
| Indexing | `ma_arr[0]`, `ma_arr[1:3]` return MaskedArray |
| Creation | `masked_array()`, `array()` |
| Masking by value | `masked_equal`, `masked_greater`, `masked_less`, `masked_less_equal`, `masked_greater_equal`, `masked_not_equal` |
| Masking by range | `masked_inside`, `masked_outside` |
| Masking by condition | `masked_where`, `masked_invalid` |
| Utilities | `is_masked`, `getdata`, `getmaskarray`, `fix_invalid` |
| Constant | `masked` — the masked singleton |

### Polynomial (`np.polynomial`)

`polyval`, `polyfit`, `polyadd`, `polysub`, `polymul`, `polyder`, `polyint`

Also: `np.polyval`, `np.polyfit`, `np.polyadd`, `np.polysub`, `np.polymul`, `np.polyder`, `np.polyint`, `np.roots`, `np.polydiv`, `np.poly1d`

### String operations (`np.char`)

| Category | Functions |
|----------|-----------|
| Case | `upper`, `lower`, `capitalize`, `title`, `swapcase` |
| Testing | `startswith`, `endswith`, `isalpha`, `isdigit`, `isnumeric`, `isupper`, `islower`, `isspace` |
| Manipulation | `strip`, `replace`, `center`, `ljust`, `rjust`, `zfill`, `split`, `join`, `find`, `count` |
| Other | `str_len`, `encode`, `decode`, `multiply`, `add` |

### Testing (`np.testing`)

| Function | Notes |
|----------|-------|
| `assert_allclose` | Tolerance-based comparison with `rtol`, `atol`, `equal_nan` |
| `assert_array_equal` | Exact array equality |
| `assert_array_almost_equal` | Decimal-place comparison |
| `assert_equal` | General equality |
| `assert_raises` | Exception assertion |
| `assert_raises_regex` | Exception + message pattern matching |
| `assert_warns` | Warning assertion (no-op in this runtime) |
| `assert_approx_equal` | Significant digit comparison |
| `assert_array_less` | Element-wise less-than assertion |

---

## Known limitations

### Dtype handling
- Most operations convert to Float64 internally. `astype()` works for explicit conversion.
- Integer arithmetic may promote to Float64.
- `result_type()` and `promote_types()` have basic implementations.

### Parameters accepted but ignored
- `order` (memory layout) — always C-contiguous.
- `out` (output array) — no in-place output support for ufuncs.
- `casting`, `subok`, `where` — silently ignored on most functions.

### Complex numbers
- Scalars extracted from complex arrays are returned as `(re, im)` tuples (RustPython limitation).
- `var`/`std` reject complex inputs; use `np.abs(a)` first.
- Comparison operators (`<`, `>`, etc.), `sort`, `floor`/`ceil`, bitwise ops raise `TypeError` on complex.

### Performance
- `einsum` uses brute-force iteration. Fine for small-to-medium arrays.
- FFT 2D/nD implemented via iterated 1D transforms.
- `rfft`/`irfft` are pure Python (1D only).
- No SIMD or parallelism — single-threaded execution.

### Stubs (accept calls, approximate behavior)
- `empty()` / `empty_like()` — return zeros.
- `seterr()` / `geterr()` / `errstate()` — no-ops.
- `ndarray.view()` — returns a copy (no shared memory views).
- `may_share_memory()` / `shares_memory()` — return False.

---

## Still needs work

These items are not yet implemented but may be needed for full compatibility:

### Would break some code
- **Structured/record arrays** — no compound dtypes or field access.
- **Datetime/timedelta** — dtype aliases exist but no operations.
- **`np.ufunc` protocol** — no universal function objects (`.reduce()`, `.accumulate()`, `.outer()` work via `_UfuncWithReduce` wrapper for `maximum`, `minimum`, `add`, `multiply`).
- **`rfft`/`irfft` for nD** — currently 1D only.

### Missing submodules
- **`np.polynomial.chebyshev`**, **`hermite`**, **`laguerre`**, **`legendre`** — only basic polynomial operations.
- **`np.fft.hfft`/`ihfft`** — Hermitian FFT.

### Edge cases
- Multi-axis fancy indexing (`a[[0,1], [2,3]]`) — not supported.
- **Fortran-order arrays** — everything is C-order.
- **Memory-mapped files** (`np.memmap`) — not supported.
- **`.npy` binary format** — `save`/`load` use text format.
- Some `pad` modes not implemented (raises `NotImplementedError` with clear message).

---

## Test coverage

| Suite | Tests | Description |
|-------|-------|-------------|
| Rust unit tests | 425 | Core: dtypes, math, broadcasting, sorting, einsum, linalg, FFT, random, strings |
| Python `test_numeric.py` | 806 | Comprehensive integration: all functions, edge cases, regressions |
| Python `test_array_creation.py` | 14 | Array creation functions |
| Python `test_indexing.py` | 38 | Indexing, slicing, assignment |
| Python `test_linalg.py` | 19 | Linear algebra operations |
| **Total** | **1,302** | |

---

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
cargo build --workspace --all-features

# Build (WASM, as used by wasmsand)
cargo build -p numpy-rust-wasm --target wasm32-wasip1

# Rust unit tests
cargo test --workspace --all-features

# Python integration tests (all 4 test files)
./target/debug/numpy-python tests/python/test_numeric.py
./target/debug/numpy-python tests/python/test_array_creation.py
./target/debug/numpy-python tests/python/test_indexing.py
./target/debug/numpy-python tests/python/test_linalg.py

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
