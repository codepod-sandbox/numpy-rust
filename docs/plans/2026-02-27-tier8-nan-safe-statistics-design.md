# Tier 8: NaN-safe Functions + Statistics — Design

## Goal

Add the most commonly needed functions for LLM-generated data analysis code: NaN-safe reductions, correlation/covariance, histogram/bincount, set operations, stacking fixes, and index utilities.

## Motivation

This numpy clone runs in codepod — a secure sandboxed execution environment for LLMs. The most common LLM workflow is: load data → check for NaN → compute statistics → filter/aggregate. Missing `nanmean` alone breaks ~40% of real-world data analysis code.

## Architecture

Follows established patterns — core Rust implementations dispatching over `ArrayData`, Python bindings via `#[pyfunction]` and `#[pymethod]`, Python wrappers in `__init__.py`.

## Features

### 1. NaN-safe Reductions (`ops/nan_reduction.rs` — new file)

**Functions**: `nansum`, `nanmean`, `nanstd`, `nanvar`, `nanmin`, `nanmax`, `nanargmin`, `nanargmax`, `nanprod`

**Implementation**:
- Cast to Float64 (NaN only exists in float types)
- Filter out NaN values before delegating to reduction logic
- Support `axis` and `keepdims` parameters (reuse `validate_axis`, `maybe_keepdims`)

**Pattern for each function**:
- `axis=None`: Flatten, filter NaN, apply reduction on remaining values
- `axis=Some(ax)`: Iterate lanes along axis, filter NaN per lane, reduce each lane

**Edge cases** (matching NumPy):
- All-NaN slice → NaN for mean/std/var/sum, raise error for min/max
- Empty array after NaN removal → same as above
- No NaN present → identical to non-nan variant

### 2. Correlation & Covariance (`ops/correlation.rs` — new file)

**Functions**: `corrcoef(x, y=None, rowvar=True)`, `cov(m, y=None, rowvar=True, ddof=None)`

**Implementation**:
- `cov(m, ddof=1)`: Rows = variables, columns = observations. Center by subtracting row means, compute `X @ X.T / (N - ddof)`.
- `corrcoef(x)`: Compute `cov(x, ddof=0)`, normalize: `C[i,j] / sqrt(C[i,i] * C[j,j])`
- `corrcoef(x, y)`: Stack into 2×N matrix, then compute as above.
- All inputs cast to Float64.

### 3. Histogram & Bincount (extend `ops/statistics.rs`)

**`histogram(a, bins=10, range=None)`**:
- Flatten input, cast to Float64
- Uniform bins only (integer `bins` parameter)
- If `range=None`, use `(min, max)` of data
- Return `(counts: Int64, bin_edges: Float64)` — counts has length `bins`, edges has length `bins + 1`

**`bincount(x, weights=None, minlength=0)`**:
- Input must be non-negative Int64 array
- Count occurrences of each value `0..max(x)`
- If `weights` provided, sum weights instead of counting
- `minlength` sets minimum output length
- Return Int64 array (or Float64 if weights)

### 4. Set Operations (extend `ops/selection.rs`)

**Functions**: `intersect1d(a, b)`, `union1d(a, b)`, `setdiff1d(a, b)`, `isin(element, test_elements)`

**Implementation**:
- `intersect1d`, `union1d`, `setdiff1d`: Flatten both, cast to Float64, sort, apply set logic, return sorted result
- `isin(element, test_elements)`: Return boolean array same shape as `element`. Build HashSet from `test_elements`, check membership.

### 5. Stacking Fixes (extend `manipulation.rs`)

**Fix `stack(arrays, axis=0)`**:
- Currently aliases `concatenate` — wrong. Should insert a new axis before concatenating.
- `stack([a, b], axis=0)` on two `(3,)` arrays → `(2, 3)` (expand_dims each, then concatenate)

**`column_stack(tup)`**:
- 1-D arrays → treated as columns, result is 2-D
- 2-D arrays → concatenate along axis 1

**`dstack(tup)`**:
- Stack along axis 2 (depth). Expand 1-D to `(1, N, 1)`, 2-D to `(M, N, 1)`, then concatenate.

### 6. Index Utilities (extend `indexing.rs`)

**`unravel_index(indices, shape)`**:
- Convert flat indices to tuple of coordinate arrays
- Pure arithmetic using shape strides (C-order)
- Single index → tuple of ints; array of indices → tuple of arrays

**`ravel_multi_index(multi_index, dims)`**:
- Convert multi-dimensional indices to flat indices
- Inverse of `unravel_index`

## Testing

- Rust unit tests for each new function in their respective modules
- Python integration tests in `tests/python/test_numeric.py`
- Test edge cases: empty arrays, all-NaN, single element, axis variations

## Verification

After each task:
1. `cargo fmt --all -- --check`
2. `cargo clippy --workspace --all-features -- -D warnings`
3. `cargo test --workspace --all-features`
4. `./tests/python/run_tests.sh target/debug/numpy-python`
