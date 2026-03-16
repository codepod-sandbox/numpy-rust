# Structured / Record Array Support — Design Spec

**Goal:** Add structured (compound-dtype) array support backed by columnar Rust storage, enabling the pandas operations that depend on `np.dtype([('x', float), ('y', int)])`, `arr['field']`, and `np.recarray`.

**Architecture:** Columnar Rust storage — one `ArrayData` column per field, each a full Rust-backed `ArcArray`. Field extraction is O(1). Row access assembles a Python `numpy.void` from scalar values. Creation is orchestrated in Python; field operations live in Rust.

**Tech Stack:** Rust (`ndarray`, existing `ArrayData` enum), RustPython bindings (`pyo3`-style via RustPython's `pyclass`/`pymethods`), Python wrappers (`_core_types.py`, `__init__.py`).

---

## 1. Data Model (Rust Core)

**File:** `crates/numpy-rust-core/src/struct_array.rs` (new)

```rust
pub struct FieldSpec {
    pub name: String,
    pub data: ArrayData,   // existing homogeneous column
}

pub struct StructArrayData {
    pub fields: Vec<FieldSpec>,   // ordered named columns
    pub shape: Vec<usize>,        // logical shape, e.g. [3] for 3-record array
}
```

Key methods on `StructArrayData`:
- `field(name) -> Option<&ArrayData>` — get column by name
- `field_names() -> Vec<&str>` — ordered names
- `get_row(flat_idx) -> Vec<ScalarValue>` — extract one record as scalar vec
- `set_field(name, ArrayData) -> Result<(), NumpyError>` — replace column

`ScalarValue` is a new enum mirroring the `ArrayData` variants but for single values (reuse or extend the existing scalar extraction logic in `py_array.rs`).

`StructArrayData` is exported from `crates/numpy-rust-core/src/lib.rs`.

---

## 2. Python Binding Layer

**File:** `crates/numpy-rust-python/src/py_struct_array.rs` (new)
**Registered in:** `crates/numpy-rust-python/src/lib.rs` as `_native.StructuredArray`

### Class definition

```rust
#[pyclass(name = "StructuredArray")]
pub struct PyStructuredArray {
    inner: RwLock<StructArrayData>,
    dtype_str: String,   // e.g. "[('x', '<f8'), ('y', '<i4')]" — for dtype property
}
```

### Operations

| Python operation | Rust implementation |
|---|---|
| `arr['fieldname']` | `field(name)` → wrap `ArrayData` as `PyNdArray` |
| `arr[['x','y']]` | extract subset of fields → new `PyStructuredArray` |
| `arr[i]` | `get_row(i)` → return `numpy.void` Python object |
| `arr['field'] = values` | `set_field(name, ...)` after converting values to `ArrayData` |
| `arr.shape` | `inner.shape.clone()` |
| `arr.ndim` | `inner.shape.len()` |
| `arr.dtype` | return `dtype_str` for Python to parse into `StructuredDtype` |
| `len(arr)` | `inner.shape[0]` |
| `iter(arr)` | iterate integer indices, yield `numpy.void` per row |
| `arr.field_names()` | `inner.field_names()` as `Vec<String>` |

### Constructor (called from Python)

```rust
// _native.StructuredArray(fields, shape)
// fields: list of (name: str, array: PyNdArray)
// shape: list of int
fn __init__(fields: Vec<(String, PyRef<PyNdArray>)>, shape: Vec<usize>) -> Self
```

The constructor validates that every field's array shape matches `shape`.

---

## 3. Python Wrapper Layer

### 3a. Fix `StructuredDtype` (`python/numpy/_core_types.py`)

Add missing properties to `StructuredDtype`:

```python
@property
def names(self) -> tuple:
    """Ordered tuple of field names."""
    return tuple(self.fields_list)   # already have ordered field list

@property
def fields(self) -> dict:
    """Dict of {name: (dtype_obj, byte_offset)}."""
    return {name: (dtype(dt_str), offset)
            for name, (dt_str, offset) in self._fields_meta.items()}

@property
def descr(self) -> list:
    """List of (name, typestr) tuples — numpy compat."""
    return [(name, str(dtype(dt_str))) for name, (dt_str, _) in self._fields_meta.items()]
```

Also fix `dtype.__eq__` and `dtype.__repr__` for structured dtypes so `str(arr.dtype)` round-trips.

### 3b. `numpy.void` scalar (`python/numpy/__init__.py` or `_core_types.py`)

```python
class void:
    """Scalar returned by arr[i] on a structured array."""
    def __init__(self, data: dict, dtype):
        self._data = data   # {fieldname: scalar_value}
        self.dtype = dtype

    def __getitem__(self, key):
        return self._data[key]

    def __getattr__(self, name):
        if name in self._data:
            return self._data[name]
        raise AttributeError(name)

    def __repr__(self):
        vals = tuple(self._data[n] for n in self.dtype.names)
        return repr(vals)

    def __iter__(self):
        return iter(self._data[n] for n in self.dtype.names)
```

### 3c. `numpy.recarray` (`python/numpy/__init__.py`)

```python
class recarray:
    """Structured array with attribute-style field access."""
    def __init__(self, shape, dtype):
        # create an empty StructuredArray of given shape + dtype
        ...

    def __getattr__(self, name):
        if hasattr(self, '_arr') and name in self._arr.dtype.names:
            return self._arr[name]
        raise AttributeError(name)

    def __setattr__(self, name, value):
        if name not in ('_arr', 'dtype') and hasattr(self, '_arr') and name in self._arr.dtype.names:
            self._arr[name] = value
        else:
            object.__setattr__(self, name, value)

    # delegate all other ops to self._arr
    def __getitem__(self, key): return self._arr[key]
    def __setitem__(self, key, val): self._arr[key] = val
    def __len__(self): return len(self._arr)
    def __iter__(self): return iter(self._arr)
```

`np.array(data, dtype=structured_dtype)` returns a plain `StructuredArray` (not recarray).
`arr.view(np.recarray)` or `np.rec.array(data)` returns a `recarray`.

### 3d. Creation routing (`python/numpy/__init__.py` and `_creation.py`)

Replace the `_ObjectArray` fallback for structured dtypes:

```python
# in _array_core(), where compound dtype is detected:
if is_structured_dtype(dt):
    return _create_structured_array(data, parsed_dtype)
```

```python
def _create_structured_array(data, sdt: StructuredDtype):
    """data: sequence of tuples; sdt: StructuredDtype"""
    names = sdt.names
    n = len(data)
    fields = []
    for i, name in enumerate(names):
        field_dtype = sdt.fields[name][0]
        col_values = [row[i] for row in data]
        col_arr = array(col_values, dtype=field_dtype)
        fields.append((name, col_arr._native))
    return _native.StructuredArray(fields, [n])
```

2D structured arrays (`shape=(m,n)`) are deferred — out of scope for v1.

---

## 4. Scope

### In scope

- Create 1D structured arrays from list-of-tuples + structured dtype
- `arr['field']` → Rust-backed `PyNdArray` column
- `arr[['f1','f2']]` → new `StructuredArray` with subset fields
- `arr[i]` → `numpy.void` scalar with field access
- `arr['field'] = values` → field assignment
- `for row in arr` → yields `numpy.void`
- `dtype.names`, `dtype.fields`, `dtype.descr`, `dtype.itemsize`
- `numpy.void` scalar class
- `numpy.recarray` wrapper class
- `np.rec.array(data, dtype=...)` convenience constructor

### Out of scope (v1)

- 2D structured arrays
- `np.lib.recfunctions` (merge_arrays, stack_arrays, etc.)
- Structured array sorting (`np.sort` by field)
- Boolean fancy indexing into structured arrays
- `.view()` with structured dtype
- Structured array I/O (`.npy` format for structured dtypes)

---

## 5. Testing

**New file:** `tests/python/test_structured.py`

Cover:
- Construction: list-of-tuples + structured dtype
- Field access: `arr['x']` returns correct values and dtype
- Multi-field: `arr[['x','y']]` returns correct subset
- Row access: `arr[0]` is `numpy.void` with `row['x']` and `row.x` working
- Field assignment: `arr['x'] = new_values` updates correctly
- Iteration: `list(arr)` gives list of `numpy.void`
- `dtype.names`: correct tuple
- `dtype.fields`: correct dict with offsets
- `np.recarray`: `arr.x` attribute access works
- Shape/ndim on the structured array itself
- Structured array of different numeric dtypes (int, float, bool)

Compatibility tests via `tests/numpy_compat/run_compat.py` — any existing structured array compat tests that were previously xfail'd should pass.
