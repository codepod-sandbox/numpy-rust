# Structured / Record Array Support — Design Spec

**Goal:** Add structured (compound-dtype) array support backed by columnar Rust storage, enabling the pandas operations that depend on `np.dtype([('x', float), ('y', int)])`, `arr['field']`, and `np.recarray`.

**Architecture:** Columnar Rust storage — one `ArrayData` column per field, each a full Rust-backed `ArcArray`. Field extraction is O(1). Row access assembles a Python `numpy.void` from scalar values. Creation is orchestrated in Python; field operations live in Rust.

**Tech Stack:** Rust (`ndarray`, existing `ArrayData` enum), RustPython bindings (`pyclass`/`pymethods`), Python wrappers (`_core_types.py`, `__init__.py`).

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
- `field_mut(name) -> Option<&mut ArrayData>` — get mutable column by name
- `field_names() -> Vec<&str>` — ordered names
- `get_row(idx: isize) -> Result<Vec<ScalarValue>, NumpyError>` — extract one record, supports negative indexing (converts `idx` to `0..n` range, errors if out of bounds)
- `set_field(name: &str, data: ArrayData) -> Result<(), NumpyError>` — replace column, validates shape

`ScalarValue` reuses the existing scalar extraction logic already present in `py_array.rs` (the pattern used for scalar returns from indexing).

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
    /// StructuredDtype repr string, e.g. "dtype([('x', 'float64'), ('y', 'int32')])"
    /// Produced by Python before construction; stored verbatim; returned by .dtype property
    /// so Python can reconstruct the StructuredDtype via dtype.__init__(existing_structured_dtype).
    dtype_repr: String,
}
```

**Locking strategy:** follows the existing `PyNdArray` pattern — take a read lock for getitem/property access, write lock for setitem. During `__init__`, each `PyNdArray` argument's inner `NdArray` is read-locked briefly to extract `ArrayData` (via `clone()`, which is O(1) for `ArcArray`), then released immediately before the `StructArrayData` is constructed. No two locks are held simultaneously.

### Operations

| Python operation | Rust implementation |
|---|---|
| `arr['fieldname']` | `field(name)` → wrap `ArrayData` clone as `PyNdArray` |
| `arr[['x','y']]` | extract subset of fields → new `PyStructuredArray` |
| `arr[i]` (int) | `get_row(i)` → `Vec<ScalarValue>` zipped with `field_names()` → Python dict `{name: scalar}` → `numpy.void` |
| `arr['field'] = values` | `set_field(name, ...)` after converting values to `ArrayData` |
| `arr.shape` | `inner.shape.clone()` as Python tuple |
| `arr.ndim` | `inner.shape.len()` |
| `arr.dtype` | return `dtype_repr` string; Python parses via `dtype._structured` path |
| `len(arr)` | `inner.shape[0]` (0 for empty) |
| `iter(arr)` | iterate integer indices 0..len, yield `numpy.void` per row |
| `arr.field_names()` | `inner.field_names()` as `Vec<String>` |

### Constructor

```rust
// _native.StructuredArray(fields, shape, dtype_repr)
// fields: list of (name: str, array: PyNdArray)
// shape: list of int
// dtype_repr: str — repr of the Python StructuredDtype
fn new(
    fields: Vec<(String, PyRef<PyNdArray>)>,
    shape: Vec<usize>,
    dtype_repr: String,
    vm: &VirtualMachine,
) -> PyResult<Self>
```

Validates that every field's array shape equals `shape`. Returns `ValueError` if mismatched or if `fields` is empty with non-zero shape.

---

## 3. Python Wrapper Layer

### 3a. Fix `StructuredDtype` (`python/numpy/_core_types.py`)

`StructuredDtype.__init__` already sets `self.names` (tuple), `self.fields` (dict `{name: (dtype_obj, offset)}`), and `self.itemsize` as plain instance attributes. **No changes needed for those.**

The only missing property is `descr`. Add it as a plain instance attribute in `__init__` (not a `@property`, to avoid shadowing):

```python
# add at end of StructuredDtype.__init__, after self.str = ...
self.descr = [(name, str(dt_obj)) for name, dt_obj in self._fields]
```

`dtype.__init__` copies `sd.names` and `sd.fields` from `StructuredDtype` when constructing from a list-of-tuples or a `StructuredDtype` instance — add `self.descr = sd.descr` to **both** those branches.

Also fix the `dtype(existing_dtype)` copy path (`elif isinstance(tp, dtype):`): if `tp` is a structured dtype (i.e. has a `_structured` attribute), copy `self.descr = tp.descr` there too.

### 3b. `numpy.void` scalar (`python/numpy/__init__.py`)

```python
class void:
    """Scalar returned by arr[i] on a structured array."""
    def __init__(self, data: dict, dtype):
        # Use object.__setattr__ to avoid any __setattr__ override issues
        object.__setattr__(self, '_data', data)   # {fieldname: scalar_value}
        object.__setattr__(self, 'dtype', dtype)

    def __getitem__(self, key):
        return object.__getattribute__(self, '_data')[key]

    def __getattr__(self, name):
        # Only called when normal lookup fails (i.e. name not in instance __dict__)
        data = object.__getattribute__(self, '_data')
        if name in data:
            return data[name]
        raise AttributeError(name)

    def __repr__(self):
        data = object.__getattribute__(self, '_data')
        dt = object.__getattribute__(self, 'dtype')
        vals = tuple(data[n] for n in dt.names)
        return repr(vals)

    def __iter__(self):
        data = object.__getattribute__(self, '_data')
        dt = object.__getattribute__(self, 'dtype')
        return iter(data[n] for n in dt.names)
```

Using `object.__getattribute__` inside `__getattr__` prevents infinite recursion when `_data` is accessed before being set.

### 3c. `numpy.recarray` (`python/numpy/__init__.py`)

```python
class recarray:
    """Structured array with attribute-style field access."""

    def __init__(self, shape, dtype):
        """Create empty structured array of given shape and structured dtype."""
        from numpy import zeros
        # Normalise shape
        if isinstance(shape, int):
            shape = (shape,)
        if len(shape) != 1:
            raise ValueError(
                "recarray only supports 1D arrays in this implementation; "
                "got shape {}".format(shape)
            )
        arr = zeros(shape[0], dtype=dtype)   # returns StructuredArray
        object.__setattr__(self, '_arr', arr)
        object.__setattr__(self, 'dtype', arr.dtype)

    def __getattr__(self, name):
        arr = object.__getattribute__(self, '_arr')
        dt = object.__getattribute__(self, 'dtype')
        if name in dt.names:
            return arr[name]
        raise AttributeError(name)

    def __setattr__(self, name, value):
        try:
            arr = object.__getattribute__(self, '_arr')
            dt = object.__getattribute__(self, 'dtype')
        except AttributeError:
            object.__setattr__(self, name, value)
            return
        if name in dt.names:
            arr[name] = value
        else:
            object.__setattr__(self, name, value)

    def __getitem__(self, key):
        return object.__getattribute__(self, '_arr')[key]
    def __setitem__(self, key, val):
        object.__getattribute__(self, '_arr')[key] = val
    def __len__(self):
        return len(object.__getattribute__(self, '_arr'))
    def __iter__(self):
        return iter(object.__getattribute__(self, '_arr'))
    @property
    def shape(self):
        return object.__getattribute__(self, '_arr').shape
    @property
    def ndim(self):
        return object.__getattribute__(self, '_arr').ndim
```

`np.array(data, dtype=structured_dtype)` returns a plain `StructuredArray` (not recarray).
`np.rec.array(data, dtype=...)` or `arr.view(np.recarray)` returns a `recarray`.

### 3d. Python-level `StructuredArray` wrapper (`python/numpy/__init__.py`)

`_native.StructuredArray` is a Rust object, not a Python ndarray. Code that receives the result of `np.array(...)` expects a Python object with a standard interface. Wrap it in a thin Python class:

```python
class StructuredArray:
    """Python wrapper for _native.StructuredArray."""

    def __init__(self, native_arr):
        object.__setattr__(self, '_native_arr', native_arr)
        # Parse dtype_repr back into a StructuredDtype for .dtype property
        # dtype_repr is "dtype([('x', 'float64'), ...])" — eval or parse
        # Use _parse_structured_dtype_repr(native_arr.dtype) helper (see below)
        dt = _parse_structured_dtype_repr(native_arr.dtype)
        object.__setattr__(self, 'dtype', dt)

    def __getitem__(self, key):
        result = object.__getattribute__(self, '_native_arr')[key]
        if isinstance(result, _native.StructuredArray):
            return StructuredArray(result)  # multi-field selection
        if isinstance(result, _native.NdArray):
            return _wrap_native(result)     # single field → ndarray
        return result   # numpy.void scalar

    def __setitem__(self, key, val):
        object.__getattribute__(self, '_native_arr')[key] = _unwrap(val)

    def __len__(self):
        return len(object.__getattribute__(self, '_native_arr'))

    def __iter__(self):
        # yields numpy.void per row
        native = object.__getattribute__(self, '_native_arr')
        dt = object.__getattribute__(self, 'dtype')
        for i in range(len(self)):
            row_data = native[i]   # dict from Rust
            yield void(row_data, dt)

    @property
    def shape(self):
        return tuple(object.__getattribute__(self, '_native_arr').shape)

    @property
    def ndim(self):
        return len(self.shape)

    @property
    def field_names(self):
        return object.__getattribute__(self, '_native_arr').field_names()
```

`_parse_structured_dtype_repr(s)` is a small helper that converts the `dtype_repr` string back to a `StructuredDtype`. The simplest approach: `StructuredDtype.__repr__` produces `"dtype([('x', 'float64'), ('y', 'int32')])"`, so we can store the raw field list as JSON instead and parse it trivially. **Decision: `dtype_repr` field in `PyStructuredArray` stores a JSON array of `[name, dtype_str]` pairs**, e.g. `[["x","float64"],["y","int32"]]`. Python parses with `json.loads()` and reconstructs `StructuredDtype`. This avoids fragile repr parsing.

The Rust constructor receives this JSON string from Python (Python serialises the dtype before calling `_native.StructuredArray(...)`).

### 3e. Creation routing (`python/numpy/_creation.py` and `__init__.py`)

Replace the `_ObjectArray` fallback for structured dtypes:

```python
# in _array_core(), where compound dtype is detected:
if is_structured_dtype(dt):
    return _create_structured_array(data, parsed_dtype)
```

```python
def _create_structured_array(data, sdt):
    """data: sequence of tuples; sdt: StructuredDtype"""
    import json
    names = sdt.names
    nrows = len(data)
    fields = []
    for i, name in enumerate(names):
        field_dtype_obj, _ = sdt.fields[name]
        col_values = [row[i] for row in data]
        col_arr = array(col_values, dtype=field_dtype_obj)
        fields.append((name, col_arr._native))
    dtype_json = json.dumps([[nm, str(sdt.fields[nm][0])] for nm in names])
    native = _native.StructuredArray(fields, [nrows], dtype_json)
    return StructuredArray(native)
```

**Empty arrays:** `np.zeros(n, dtype=structured_dtype)` and `np.empty(n, dtype=structured_dtype)` must also route through structured array creation. These create zero-filled columns per field:

```python
def _create_empty_structured(n, sdt, fill=0.0):
    import json
    fields = []
    for name in sdt.names:
        field_dtype_obj, _ = sdt.fields[name]
        col_arr = full(n, fill, dtype=field_dtype_obj)
        fields.append((name, col_arr._native))
    dtype_repr = json.dumps([[n2, str(sdt.fields[n2][0])] for n2 in sdt.names])
    native = _native.StructuredArray(fields, [n], dtype_repr)
    return StructuredArray(native)
```

`zeros`, `empty`, and `full` in `__init__.py` check for structured dtype and delegate to `_create_empty_structured`.

---

## 4. Scope

### In scope

- Create 1D structured arrays from list-of-tuples + structured dtype
- `arr['field']` → Rust-backed `PyNdArray` column
- `arr[['f1','f2']]` → new `StructuredArray` with subset fields
- `arr[i]` → `numpy.void` scalar with field access (positive and negative indexing)
- `arr['field'] = values` → field assignment
- `for row in arr` → yields `numpy.void`
- `dtype.names`, `dtype.fields`, `dtype.itemsize` (already exist), `dtype.descr` (new)
- `numpy.void` scalar class
- `numpy.recarray` wrapper class
- `np.rec.array(data, dtype=...)` convenience constructor
- `np.zeros(n, dtype=structured_dtype)` and `np.empty(n, dtype=structured_dtype)`

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
- Construction from list-of-tuples + structured dtype
- `arr['x']` returns correct values and dtype
- `arr[['x','y']]` returns correct subset structured array
- `arr[0]` is `numpy.void` with `row['x']` working
- `arr[-1]` negative indexing works
- `arr['x'] = new_values` updates correctly
- `for row in arr` yields `numpy.void` values
- `dtype.names` is correct tuple
- `dtype.fields` is correct dict with offsets
- `dtype.descr` is correct list of (name, typestr)
- `np.zeros(3, dtype=sdt)` creates zero-filled structured array
- `np.recarray`: `arr.x` attribute access works
- `arr.shape` and `arr.ndim` correct
- Various numeric field dtypes (int32, float64, bool)
- Empty structured array (n=0)
- `dtype.itemsize` is sum of field itemsizes

Compatibility tests via `tests/numpy_compat/run_compat.py` — any existing structured array compat tests that were previously xfail'd should pass.
