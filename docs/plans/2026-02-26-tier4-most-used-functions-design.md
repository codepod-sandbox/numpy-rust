# Tier 4: Most-Used Missing Functions — Design

## Goal

Add the most commonly used NumPy functions that are currently missing or broken: cumulative operations, array splitting, discrete differences, element repetition, product reduction, percentile/quantile, and index utilities.

## Architecture

All new operations follow the established pattern: core logic in `numpy-rust-core` with Rust unit tests, Python bindings in `numpy-rust-python`, and thin wrappers in `python/numpy/__init__.py`. New operations go in existing module files where they fit naturally (reduction.rs for prod, manipulation.rs for split/repeat/tile) or new files where appropriate (cumulative.rs, statistics.rs).

## Features

### 1. `cumsum` and `cumprod`

Cumulative sum and product along an axis.

- `cumsum(axis=None)` — if axis is None, flatten first; returns same-size array
- `cumprod(axis=None)` — same pattern
- New file `ops/cumulative.rs`
- Implementation: cast to Float64, iterate lanes along axis, accumulate with running sum/product
- Both support all numeric dtypes (not string/complex for cumprod)

### 2. `diff`

N-th discrete difference along an axis.

- `diff(a, n=1, axis=-1)` — result shape is input shape with `shape[axis] -= n`
- Implementation: for n=1, subtract `a[1:]` from `a[:-1]` along axis using slicing
- For n>1, apply n=1 repeatedly
- Add to `ops/cumulative.rs` (related to cumulative ops)

### 3. `split`, `vsplit`, `hsplit`

Inverse of concatenate/vstack/hstack.

- `split(a, indices_or_sections, axis=0)` — if int, split into N equal parts; if list, split at those indices
- `vsplit(a, indices)` = `split(a, indices, axis=0)`
- `hsplit(a, indices)` = `split(a, indices, axis=1)` (or axis=0 for 1-D)
- Core implementation in `manipulation.rs` using `ndarray::ArrayBase::slice_axis`
- Returns `Vec<NdArray>`

### 4. `argwhere`

Indices of non-zero elements as (N, ndim) array.

- Flatten, find non-zero indices, unravel to multi-dimensional coordinates
- Returns Int64 array of shape (num_nonzero, ndim)
- Add to `utility.rs`

### 5. `repeat` and `tile` with axis support

Fix existing broken Python stubs with proper Rust core implementations.

- `repeat(a, repeats, axis=None)` — repeat each element `repeats` times along axis; axis=None flattens first
- `tile(a, reps)` — construct array by repeating `a` the number of times given by `reps` tuple
- Core implementations in `manipulation.rs`

### 6. `prod` with axis/keepdims

Move product reduction to Rust core, following the `sum`/`mean` pattern.

- `prod(axis=None, keepdims=False)` — product of elements
- Add to `reduction.rs` alongside existing sum/mean/min/max
- Supports all numeric dtypes

### 7. `percentile` and `quantile`

Statistical quantile computation.

- `quantile(a, q, axis=None)` — linear interpolation between sorted values
- `percentile(a, q, axis=None)` — same as `quantile(a, q/100, axis)`
- New file `ops/statistics.rs`
- Implementation: sort along axis, compute interpolated values
- `q` is a scalar (0.0-1.0 for quantile, 0-100 for percentile)

### 8. `ptp` with axis

Peak-to-peak (max - min) along axis.

- Thin wrapper: `ptp(axis) = max(axis) - min(axis)`
- Add method to `reduction.rs`, fix Python wrapper

## Testing

Each feature gets:
- Rust unit tests in the relevant core module
- Python integration tests in `tests/python/test_numeric.py`

## Verification

After each task:
1. `cargo fmt --all -- --check`
2. `cargo clippy --workspace --all-features -- -D warnings`
3. `cargo test --workspace --all-features`
4. `./tests/python/run_tests.sh target/debug/numpy-python`
