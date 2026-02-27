# Tier 9: Dtype Correctness — Design

## Goal

Make the `dtype` parameter work correctly across all creation functions, fix `promote_types`/`result_type` stubs, and implement `full`/`full_like` natively in Rust.

## Motivation

LLMs frequently write `np.zeros(5, dtype=np.int32)` or `np.array([1,2,3], dtype='float32')`. Currently these silently produce Float64 arrays. This causes subtle bugs in ML preprocessing pipelines.

## Architecture

The Rust core already supports dtype in `NdArray::zeros()`, `NdArray::ones()`, `creation::full()`, `creation::eye()`, and `creation::arange()`. The fix is primarily in the Python binding layer (`py_creation.rs`, `lib.rs`) and Python wrappers (`__init__.py`).

## Changes

### 1. Add dtype to `py_zeros`/`py_ones` in `py_creation.rs`

Currently: `NdArray::zeros(&shape, DType::Float64)` hardcoded.
Fix: Accept optional dtype string, parse via `parse_dtype`, pass to core.

### 2. Add dtype to `array()` Python wrapper

When `dtype` is a numeric string/ScalarType: create array normally, then `.astype(dtype)`.

### 3. Add dtype to `arange`/`linspace`/`eye` bindings

- `arange`: Core already accepts `Option<DType>` — wire it through
- `linspace`: Add `.astype()` call on result
- `eye`: Core already accepts `DType` — wire it through

### 4. Implement `full`/`full_like` natively

Replace Python workaround (`ones * value`) with native binding calling `creation::full()`.
Add `full_like` that gets shape from source array.

### 5. Fix `promote_types`/`result_type`

Expose `DType::promote()` to Python. `promote_types(a, b)` returns the promoted dtype.
`result_type` uses the same logic on arrays' dtypes.

### 6. Wire dtype for `zeros_like`/`ones_like`/`empty_like`

Accept optional dtype, use it to override the source array's dtype.

## Testing

Python integration tests verifying dtype parameter produces correct array dtype.
