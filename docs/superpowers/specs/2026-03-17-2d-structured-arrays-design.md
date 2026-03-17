# 2D Structured Arrays Design

## Goal

Extend `StructuredArray` from 1D-only to multi-D (primarily 2D), enabling `np.resize`/`np.reshape` on structured arrays. Required for pandas compatibility and to fix `TestResize.test_reshape_from_zero`.

## Approach

Python-side shape tracking (Approach A). Rust stores data as 1D flat columns; Python `StructuredArray` tracks a `_py_shape` overlay. Vectorized field access (the hot path) is unaffected — it goes straight to Rust. Per-element overhead is one Python index computation, acceptable because element-by-element iteration is an anti-pattern in NumPy code.

## Architecture

### Rust (2 files, minimal changes)

**`crates/numpy-rust-core/src/struct_array.rs`**
Add `slice_rows(start: usize, end: usize) -> StructArrayData`: slices each field column `[start:end]`, returns a new `StructArrayData` with `shape=[end-start]`. All field columns are 1D, so this is a simple element extraction on each `ArrayData`.

**`crates/numpy-rust-python/src/py_struct_array.rs`**
Add `PySlice` branch in `__getitem__`: resolve start/end against `self.len()`, call `slice_rows`, return a new `PyStructuredArray`.

### Python (`python/numpy/__init__.py`)

`StructuredArray` gains:

- `_py_shape: Optional[tuple]` — `None` = use Rust shape (1D, existing behavior); tuple = Python-tracked multi-D shape
- `shape` property: return `_py_shape` if set, else Rust shape
- `ndim` property: `len(self.shape)`
- `reshape(*new_shape)` — return new `StructuredArray` wrapping same `_native_arr` with new `_py_shape`. No data copy. Validate `product(new_shape) == product(current_shape)`.
- `__getitem__` additions:
  - string key + ndim > 1 → get column from Rust, reshape result to `_py_shape`
  - int key + ndim > 1 → `flat = i * cols`, call `native[flat:flat+cols]` (PySlice), wrap as `StructuredArray` with shape `(cols,)`
  - tuple key `(i, j)` → flat index `i*cols + j` → void scalar via `native[flat_idx]`
- `__len__`: `self.shape[0]` (first dim)
- `__iter__`: for ndim > 1, yield rows as 1D StructuredArrays

### Python (`python/numpy/_creation.py`)

`zeros` / `empty` / `ones` with structured dtype + multi-D shape:
- Create flat StructuredArray of `product(shape)` elements
- Wrap with `_py_shape = shape`

### Python (`python/numpy/_manipulation.py`)

`resize` for StructuredArray:
- Detect `StructuredArray` input
- Tile each field independently: `ndarray.resize(flat_field, (total,))` to get tiled column of length `product(new_shape)`
- Build new `StructuredArray` wrapping tiled fields, set `_py_shape = new_shape`

## Data Flow

```
np.zeros(0, dtype=[('a', float32)])  →  StructuredArray, _py_shape=None, shape=(0,)
np.resize(A, (2, 1))
  → detect StructuredArray
  → tile each field to length 2 (zeros)
  → StructuredArray(_native=flat[2], _py_shape=(2,1))
  → .shape == (2, 1)  ✓

Ar['a']          →  Rust returns 1D ndarray[2], Python reshapes to (2,1)  ✓
Ar[0]            →  native[0:1], StructuredArray(_py_shape=None, shape=(1,))  ✓
Ar[0, 0]         →  native[0] → void scalar  ✓
Ar.dtype == A.dtype  ✓
```

## Invariants

- Rust always stores 1D flat data; `_native_arr.shape` is always `(n,)` for flat storage
- `_py_shape` is always `None` or a tuple whose product equals `len(_native_arr)`
- `set_field` on Rust expects 1D shape `(n,)` — Python must flatten before passing
- All existing 1D behavior unchanged: when `_py_shape is None`, all paths fall through to existing code

## Testing

- `TestResize.test_reshape_from_zero` must pass and be removed from `xfail.txt`
- Compat test suite: 0 new unexpected failures
- Unit tests in `tests/python/` covering:
  - `zeros((2,3), dtype=structured)` shape and field access
  - `resize(1d_structured, (2,3))` tiling
  - `arr[i]` row access returning shape `(cols,)`
  - `arr[i, j]` element access returning void scalar
  - `arr['field']` returning shape matching `_py_shape`
  - `reshape` preserving data, changing shape
  - Rust `slice_rows` unit test
