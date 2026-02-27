# Tier 7: Array Manipulation Utilities — Design

## Goal

Add commonly-used array manipulation functions and migrate 5 Python-loop implementations to Rust: flip/flipud/fliplr, rot90, unique, diagonal, outer, roll, take.

## Architecture

Core implementations in `manipulation.rs` (flip, rot90, unique, roll, take) and `utility.rs` (diagonal, outer). Python bindings follow established patterns.

## Features

### 1. `flip`, `flipud`, `fliplr`
- `flip(a, axis=None)` — reverse elements along axis; if None, flip all axes
- `flipud(a)` = `flip(a, axis=0)`, `fliplr(a)` = `flip(a, axis=1)`
- Implementation: use ndarray `slice_axis` with negative step or build reversed index

### 2. `rot90`
- `rot90(a, k=1, axes=(0,1))` — rotate 90 degrees k times in the plane of axes
- k=1: transpose then flip axis 0; k=2: flip both; k=3: transpose then flip axis 1

### 3. `unique`
- `unique(a)` — return sorted unique values (flattened)
- Cast to Float64, sort, dedup

### 4. `diagonal`
- `diagonal(a, offset=0)` — extract diagonal from 2D array
- Replace Python loop with Rust implementation

### 5. `outer`
- `outer(a, b)` — outer product of two 1-D arrays
- Flatten both, compute element-wise products, return (M, N) array

### 6. `roll`
- `roll(a, shift, axis=None)` — circular shift of elements
- If axis=None, flatten, roll, reshape

### 7. `take`
- `take(a, indices, axis=None)` — select elements by index
- If axis=None, flatten first; else use index_select along axis

## Testing

Rust unit tests + Python integration tests for all functions.

## Verification

1. `cargo fmt --all -- --check`
2. `cargo clippy --workspace --all-features -- -D warnings`
3. `cargo test --workspace --all-features`
4. `./tests/python/run_tests.sh target/debug/numpy-python`
