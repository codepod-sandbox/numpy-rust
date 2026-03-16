# Structured / Record Array Support — Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add columnar-Rust-backed structured array support (`np.dtype([('x', float), ('y', int)])`, `arr['field']`, `np.recarray`) to enable pandas-critical workflows.

**Architecture:** One `ArrayData` column per field in a new `StructArrayData` Rust type. A `PyStructuredArray` binding exposes it to Python. A Python `StructuredArray` wrapper + `void` scalar + `recarray` class complete the public API. Creation is routed in `_creation.py`; field extraction is O(1) via ArcArray clone.

**Tech Stack:** Rust (ndarray, existing ArrayData/NdArray/Scalar), RustPython bindings (`#[vm::pyclass]`), Python wrappers (`_core_types.py`, `_creation.py`, `__init__.py`).

---

## File Map

| File | Action | Responsibility |
|---|---|---|
| `crates/numpy-rust-core/src/struct_array.rs` | **Create** | `StructArrayData` + `FieldSpec` + `get_row`/`set_field` |
| `crates/numpy-rust-core/src/lib.rs` | **Modify** | export `StructArrayData` |
| `crates/numpy-rust-python/src/py_array.rs` | **Modify** | make `numpy_err`, `scalar_to_py`, `py_obj_to_scalar` `pub(crate)` |
| `crates/numpy-rust-python/src/py_struct_array.rs` | **Create** | `PyStructuredArray` bindings |
| `crates/numpy-rust-python/src/lib.rs` | **Modify** | register `PyStructuredArray` as `_native.StructuredArray` |
| `python/numpy/_core_types.py` | **Modify** | add `descr` to `StructuredDtype` + both `dtype.__init__` structured paths |
| `python/numpy/_creation.py` | **Modify** | `_create_structured_array`, `_create_empty_structured`, route structured dtypes |
| `python/numpy/__init__.py` | **Modify** | `void`, `StructuredArray`, `recarray`, `rec` module stub |
| `tests/python/test_structured.py` | **Create** | full structured array test suite |

---

## Chunk 1: Rust Core + Bindings

### Task 1: `StructArrayData` in Rust core

**Files:**
- Create: `crates/numpy-rust-core/src/struct_array.rs`
- Modify: `crates/numpy-rust-core/src/lib.rs`

- [ ] **Step 1: Write the failing test (Rust)**

Add a `#[cfg(test)]` module at the bottom of `struct_array.rs` as you write the file. The test will only compile once the file exists.

- [ ] **Step 2: Create `crates/numpy-rust-core/src/struct_array.rs`**

```rust
use crate::array::NdArray;
use crate::array_data::ArrayData;
use crate::error::{NumpyError, Result};
use crate::indexing::Scalar;

/// A single named column in a structured array.
pub struct FieldSpec {
    pub name: String,
    pub data: ArrayData,
}

/// Columnar structured array: each field is a separate homogeneous column.
/// `shape` is the logical shape of the record array (not including fields).
pub struct StructArrayData {
    pub fields: Vec<FieldSpec>,
    pub shape: Vec<usize>,
}

impl StructArrayData {
    pub fn new(fields: Vec<FieldSpec>, shape: Vec<usize>) -> Self {
        Self { fields, shape }
    }

    /// Get a column's ArrayData by field name.
    pub fn field(&self, name: &str) -> Option<&ArrayData> {
        self.fields.iter().find(|f| f.name == name).map(|f| &f.data)
    }

    /// Get a mutable reference to a column's ArrayData by field name.
    pub fn field_mut(&mut self, name: &str) -> Option<&mut ArrayData> {
        self.fields.iter_mut().find(|f| f.name == name).map(|f| &mut f.data)
    }

    /// Ordered list of field names.
    pub fn field_names(&self) -> Vec<&str> {
        self.fields.iter().map(|f| f.name.as_str()).collect()
    }

    /// Number of records (first dimension of shape, 0 if shape is empty).
    pub fn len(&self) -> usize {
        self.shape.first().copied().unwrap_or(0)
    }

    /// Extract one row as a Vec of Scalars (one per field, in field order).
    /// Supports negative indexing: -1 = last row.
    pub fn get_row(&self, idx: isize) -> Result<Vec<Scalar>> {
        let n = self.len();
        if n == 0 {
            return Err(NumpyError::IndexError(
                "index out of bounds: array has 0 rows".into(),
            ));
        }
        let actual_idx = if idx < 0 {
            let pos = n as isize + idx;
            if pos < 0 {
                return Err(NumpyError::IndexError(format!(
                    "index {} is out of bounds for axis 0 with size {}",
                    idx, n
                )));
            }
            pos as usize
        } else {
            let pos = idx as usize;
            if pos >= n {
                return Err(NumpyError::IndexError(format!(
                    "index {} is out of bounds for axis 0 with size {}",
                    idx, n
                )));
            }
            pos
        };
        let mut row = Vec::with_capacity(self.fields.len());
        for field in &self.fields {
            let scalar = NdArray::from_data(field.data.clone()).get(&[actual_idx])?;
            row.push(scalar);
        }
        Ok(row)
    }

    /// Replace a column. New data must have the same shape as `self.shape`.
    pub fn set_field(&mut self, name: &str, data: ArrayData) -> Result<()> {
        if data.shape() != self.shape.as_slice() {
            return Err(NumpyError::ValueError(format!(
                "shape mismatch: field data has shape {:?}, expected {:?}",
                data.shape(),
                self.shape
            )));
        }
        match self.fields.iter_mut().find(|f| f.name == name) {
            Some(field) => {
                field.data = data;
                Ok(())
            }
            None => Err(NumpyError::ValueError(format!(
                "no field named '{}'",
                name
            ))),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::array_data::ArrayD;
    use ndarray::IxDyn;

    fn make_int_col(vals: Vec<i64>) -> ArrayData {
        let n = vals.len();
        let arr = ArrayD::from_shape_vec(IxDyn(&[n]), vals)
            .unwrap()
            .into_shared();
        ArrayData::Int64(arr)
    }

    fn make_float_col(vals: Vec<f64>) -> ArrayData {
        let n = vals.len();
        let arr = ArrayD::from_shape_vec(IxDyn(&[n]), vals)
            .unwrap()
            .into_shared();
        ArrayData::Float64(arr)
    }

    #[test]
    fn test_field_access() {
        let sa = StructArrayData::new(
            vec![
                FieldSpec { name: "x".into(), data: make_int_col(vec![1, 2, 3]) },
                FieldSpec { name: "y".into(), data: make_float_col(vec![1.5, 2.5, 3.5]) },
            ],
            vec![3],
        );
        assert!(sa.field("x").is_some());
        assert!(sa.field("y").is_some());
        assert!(sa.field("z").is_none());
        assert_eq!(sa.field_names(), vec!["x", "y"]);
        assert_eq!(sa.len(), 3);
    }

    #[test]
    fn test_get_row() {
        let sa = StructArrayData::new(
            vec![
                FieldSpec { name: "x".into(), data: make_int_col(vec![10, 20, 30]) },
                FieldSpec { name: "y".into(), data: make_float_col(vec![1.1, 2.2, 3.3]) },
            ],
            vec![3],
        );
        let row = sa.get_row(0).unwrap();
        assert_eq!(row.len(), 2);
        assert!(matches!(row[0], Scalar::Int64(10)));
        let row = sa.get_row(-1).unwrap();
        assert!(matches!(row[0], Scalar::Int64(30)));
        assert!(sa.get_row(3).is_err());
        assert!(sa.get_row(-4).is_err());
    }

    #[test]
    fn test_set_field() {
        let mut sa = StructArrayData::new(
            vec![
                FieldSpec { name: "x".into(), data: make_int_col(vec![1, 2, 3]) },
            ],
            vec![3],
        );
        let new_col = make_int_col(vec![10, 20, 30]);
        sa.set_field("x", new_col).unwrap();
        let row = sa.get_row(0).unwrap();
        assert!(matches!(row[0], Scalar::Int64(10)));
        // wrong shape → error
        assert!(sa.set_field("x", make_int_col(vec![1, 2])).is_err());
        // unknown field → error
        assert!(sa.set_field("z", make_int_col(vec![1, 2, 3])).is_err());
    }
}
```

- [ ] **Step 3: Export from `crates/numpy-rust-core/src/lib.rs`**

Add after line `pub mod utility;`:
```rust
pub mod struct_array;
pub use struct_array::{FieldSpec, StructArrayData};
```

- [ ] **Step 4: Run Rust tests to verify**

```bash
cd /path/to/repo && cargo test -p numpy-rust-core struct_array --release 2>&1 | tail -20
```

Expected: `test struct_array::tests::test_field_access ... ok`, all 3 pass.

- [ ] **Step 5: Commit**

```bash
git add crates/numpy-rust-core/src/struct_array.rs crates/numpy-rust-core/src/lib.rs
git commit -m "feat(core): add StructArrayData for columnar structured arrays"
```

---

### Task 2: Make helpers pub(crate) in py_array.rs

**Files:**
- Modify: `crates/numpy-rust-python/src/py_array.rs` (~lines 239–355)

`py_struct_array.rs` needs access to `numpy_err`, `scalar_to_py`, and `py_obj_to_scalar` which are currently private. Make them `pub(crate)`.

- [ ] **Step 1: Change visibility of the three helpers**

Find these three function definitions in `py_array.rs` and add `pub(crate)`:

```rust
// line ~239 (was `fn numpy_err`)
pub(crate) fn numpy_err(
    e: numpy_rust_core::NumpyError,
    vm: &VirtualMachine,
) -> vm::builtins::PyBaseExceptionRef {
    vm.new_value_error(e.to_string())
}

// line ~246 (was `fn scalar_to_py`)
pub(crate) fn scalar_to_py(s: Scalar, vm: &VirtualMachine) -> PyObjectRef { ... }

// line ~272 (was `fn py_obj_to_scalar`)
pub(crate) fn py_obj_to_scalar(obj: &PyObjectRef, dtype: DType, vm: &VirtualMachine) -> PyResult<Scalar> { ... }
```

Only add `pub(crate)` to the `fn` keyword — do not change any function body.

- [ ] **Step 2: Verify it compiles**

```bash
cargo build -p numpy-rust-python --release 2>&1 | grep -E "error|warning: unused" | head -20
```

Expected: no errors (warnings about unused is fine).

- [ ] **Step 3: Commit**

```bash
git add crates/numpy-rust-python/src/py_array.rs
git commit -m "refactor(bindings): expose numpy_err/scalar_to_py/py_obj_to_scalar as pub(crate)"
```

---

### Task 3: `PyStructuredArray` binding

**Files:**
- Create: `crates/numpy-rust-python/src/py_struct_array.rs`
- Modify: `crates/numpy-rust-python/src/lib.rs`

- [ ] **Step 1: Create `crates/numpy-rust-python/src/py_struct_array.rs`**

```rust
use std::sync::RwLock;

use rustpython_vm as vm;
use vm::builtins::PyStr;
use vm::protocol::PyMappingMethods;
use vm::types::AsMapping;
use vm::{atomic_func, AsObject, PyObjectRef, PyPayload, PyResult, VirtualMachine};

use numpy_rust_core::{FieldSpec, NdArray, StructArrayData};

use crate::py_array::{numpy_err, scalar_to_py, PyNdArray};

/// Python-visible structured array class backed by columnar Rust storage.
#[vm::pyclass(module = "numpy", name = "StructuredArray")]
#[derive(Debug, PyPayload)]
pub struct PyStructuredArray {
    inner: RwLock<StructArrayData>,
    /// JSON string: [[name, dtype_str], ...] e.g. [["x","float64"],["y","int32"]]
    /// Passed in from Python at construction. Returned verbatim by .dtype property
    /// so Python can reconstruct the StructuredDtype via json.loads().
    dtype_json: String,
}

impl PyStructuredArray {
    fn new_from_data(sa: StructArrayData, dtype_json: String) -> Self {
        Self {
            inner: RwLock::new(sa),
            dtype_json,
        }
    }
}

#[vm::pyclass(with(AsMapping))]
impl PyStructuredArray {
    /// Constructor: _native.StructuredArray(fields, shape, dtype_json)
    ///   fields: list of [name_str, PyNdArray]
    ///   shape: list of int
    ///   dtype_json: JSON string [[name, dtype_str], ...]
    ///
    /// Uses #[pyslot] slot_new pattern (same as PyNdArray).
    #[pyslot]
    fn slot_new(
        cls: vm::builtins::PyTypeRef,
        args: vm::function::FuncArgs,
        vm: &VirtualMachine,
    ) -> PyResult {
        if args.args.len() < 3 {
            return Err(vm.new_type_error(
                "StructuredArray requires 3 positional args: fields, shape, dtype_json".to_owned(),
            ));
        }
        let fields_obj = args.args[0].clone();
        let shape_obj = args.args[1].clone();
        let dtype_json_obj = args.args[2].clone();

        // Parse shape: list of int
        let shape: Vec<usize> = {
            let list = shape_obj.try_into_value::<Vec<PyObjectRef>>(vm)?;
            list.iter()
                .map(|o| o.clone().try_into_value::<usize>(vm))
                .collect::<PyResult<Vec<_>>>()?
        };
        // Parse dtype_json
        let dtype_json = dtype_json_obj
            .downcast_ref::<PyStr>()
            .ok_or_else(|| vm.new_type_error("dtype_json must be a str".to_owned()))?
            .as_str()
            .to_owned();
        // Parse fields: list of [name_str, PyNdArray]
        let fields_list = fields_obj.try_into_value::<Vec<PyObjectRef>>(vm)?;
        let mut fields: Vec<FieldSpec> = Vec::with_capacity(fields_list.len());
        for item in fields_list {
            let tup = item.try_into_value::<Vec<PyObjectRef>>(vm)?;
            if tup.len() != 2 {
                return Err(vm.new_value_error(
                    "each field must be a [name, ndarray] pair".to_owned(),
                ));
            }
            let name = tup[0]
                .downcast_ref::<PyStr>()
                .ok_or_else(|| vm.new_type_error("field name must be a str".to_owned()))?
                .as_str()
                .to_owned();
            let py_arr = tup[1]
                .downcast_ref::<PyNdArray>()
                .ok_or_else(|| {
                    vm.new_type_error(format!("field '{}': value must be ndarray", name))
                })?;
            // O(1) clone: increments Arc refcount only
            let array_data = py_arr.inner().data().clone();
            if array_data.shape() != shape.as_slice() {
                return Err(vm.new_value_error(format!(
                    "field '{}' has shape {:?}, expected {:?}",
                    name,
                    array_data.shape(),
                    shape
                )));
            }
            fields.push(FieldSpec { name, data: array_data });
        }
        if shape.iter().product::<usize>() > 0 && fields.is_empty() {
            return Err(vm.new_value_error(
                "structured array with non-zero size must have at least one field".to_owned(),
            ));
        }
        let sa = StructArrayData::new(fields, shape);
        Ok(Self::new_from_data(sa, dtype_json)
            .into_ref_with_type(vm, cls)?
            .into())
    }

    #[pygetset]
    fn shape(&self, vm: &VirtualMachine) -> PyObjectRef {
        let inner = self.inner.read().unwrap();
        let shape_vec: Vec<PyObjectRef> = inner
            .shape
            .iter()
            .map(|&d| vm.ctx.new_int(d as i64).into())
            .collect();
        vm.ctx.new_tuple(shape_vec).into()
    }

    #[pygetset]
    fn ndim(&self) -> usize {
        self.inner.read().unwrap().shape.len()
    }

    /// Returns the JSON dtype string — Python side parses with json.loads().
    #[pygetset]
    fn dtype(&self) -> String {
        self.dtype_json.clone()
    }

    #[pymethod]
    fn field_names(&self, vm: &VirtualMachine) -> PyObjectRef {
        let inner = self.inner.read().unwrap();
        let names: Vec<PyObjectRef> = inner
            .field_names()
            .into_iter()
            .map(|n| vm.ctx.new_str(n).into())
            .collect();
        vm.ctx.new_list(names).into()
    }

    #[pymethod(magic)]
    fn len(&self) -> usize {
        self.inner.read().unwrap().len()
    }

    #[pymethod(magic)]
    fn getitem(&self, key: PyObjectRef, vm: &VirtualMachine) -> PyResult<PyObjectRef> {
        let inner = self.inner.read().unwrap();

        // String key → return PyNdArray column
        if let Some(key_str) = key.downcast_ref::<PyStr>() {
            let name = key_str.as_str();
            let col_data = inner.field(name).ok_or_else(|| {
                vm.new_key_error(vm.ctx.new_str(name).into())
            })?;
            let nd = NdArray::from_data(col_data.clone());
            return Ok(PyNdArray::from_core(nd).to_py(vm));
        }

        // List of strings → new PyStructuredArray with subset of fields
        if let Ok(key_list) = key.clone().try_into_value::<Vec<PyObjectRef>>(vm) {
            if !key_list.is_empty() && key_list.iter().all(|k| k.downcast_ref::<PyStr>().is_some()) {
                let names: Vec<&str> = key_list
                    .iter()
                    .map(|k| k.downcast_ref::<PyStr>().unwrap().as_str())
                    .collect();
                for &n in &names {
                    if inner.field(n).is_none() {
                        return Err(vm.new_key_error(vm.ctx.new_str(n).into()));
                    }
                }
                let subset_fields: Vec<FieldSpec> = names
                    .iter()
                    .map(|&n| FieldSpec {
                        name: n.to_owned(),
                        data: inner.field(n).unwrap().clone(),
                    })
                    .collect();
                // Build subset dtype_json via parse_dtype_json_to_vec
                let all_fields = parse_dtype_json_to_vec(&self.dtype_json, vm)?;
                let mut subset_parts: Vec<String> = Vec::new();
                for &n in &names {
                    if let Some((_, dt)) = all_fields.iter().find(|(k, _)| k == n) {
                        subset_parts.push(format!("[\"{}\",\"{}\"]", n, dt));
                    }
                }
                let subset_dtype_json = format!("[{}]", subset_parts.join(","));
                let sub_sa = StructArrayData::new(subset_fields, inner.shape.clone());
                // Use into_pyobject (from PyPayload) — no need for cls reference
                return Ok(Self::new_from_data(sub_sa, subset_dtype_json).into_pyobject(vm));
            }
        }

        // Integer key → return list of scalars in field order.
        // Python StructuredArray.__getitem__ zips with dtype.names to build void.
        if let Ok(idx) = key.clone().try_into_value::<i64>(vm) {
            let row_scalars = inner.get_row(idx as isize).map_err(|e| numpy_err(e, vm))?;
            let values: Vec<PyObjectRef> = row_scalars
                .into_iter()
                .map(|s| scalar_to_py(s, vm))
                .collect();
            return Ok(vm.ctx.new_list(values).into());
        }

        Err(vm.new_type_error(format!(
            "StructuredArray indices must be string, list of strings, or integer, not {}",
            key.class().name()
        )))
    }

    #[pymethod(magic)]
    fn setitem(
        &self,
        key: PyObjectRef,
        value: PyObjectRef,
        vm: &VirtualMachine,
    ) -> PyResult<()> {
        if let Some(key_str) = key.downcast_ref::<PyStr>() {
            let name = key_str.as_str();
            let mut inner = self.inner.write().unwrap();
            // value must be a PyNdArray
            let py_arr = value.downcast_ref::<PyNdArray>().ok_or_else(|| {
                vm.new_type_error("field assignment value must be ndarray".to_owned())
            })?;
            let array_data = py_arr.inner().data().clone();
            inner.set_field(name, array_data).map_err(|e| numpy_err(e, vm))?;
            return Ok(());
        }
        Err(vm.new_type_error("StructuredArray field assignment key must be a string".to_owned()))
    }
}

impl AsMapping for PyStructuredArray {
    fn as_mapping() -> &'static PyMappingMethods {
        use once_cell::sync::Lazy;
        static AS_MAPPING: Lazy<PyMappingMethods> = Lazy::new(|| PyMappingMethods {
            length: atomic_func!(|mapping, _vm| {
                let zelf = PyStructuredArray::mapping_downcast(mapping);
                Ok(zelf.__len__())
            }),
            subscript: atomic_func!(|mapping, needle: &vm::PyObject, vm| {
                let zelf = PyStructuredArray::mapping_downcast(mapping);
                zelf.__getitem__(needle.to_owned(), vm)
            }),
            ass_subscript: atomic_func!(|mapping, needle: &vm::PyObject, value, vm| {
                let zelf = PyStructuredArray::mapping_downcast(mapping);
                let value = value.ok_or_else(|| {
                    vm.new_type_error(
                        "StructuredArray does not support item deletion".to_owned(),
                    )
                })?;
                zelf.__setitem__(needle.to_owned(), value.to_owned(), vm)?;
                Ok(())
            }),
        });
        &AS_MAPPING
    }
}

/// Parse dtype_json into Vec<(name, dtype_str)>. Format: [["x","float64"],["y","int32"]]
/// Hand-rolled parser avoids adding a JSON crate dependency.
fn parse_dtype_json_to_vec(
    dtype_json: &str,
    vm: &VirtualMachine,
) -> PyResult<Vec<(String, String)>> {
    let s = dtype_json.trim();
    if !s.starts_with('[') || !s.ends_with(']') {
        return Err(vm.new_value_error(format!("invalid dtype_json: {}", dtype_json)));
    }
    let inner = &s[1..s.len() - 1];
    if inner.trim().is_empty() {
        return Ok(vec![]);
    }
    let pairs: Vec<&str> = inner.split("],[").collect();
    let mut result = Vec::new();
    for pair in pairs {
        let clean = pair.trim_matches(|c| c == '[' || c == ']');
        let parts: Vec<&str> = clean.splitn(2, ',').collect();
        if parts.len() != 2 {
            return Err(vm.new_value_error(format!("bad pair in dtype_json: {}", pair)));
        }
        let name = parts[0].trim().trim_matches('"').to_owned();
        let dtype_str = parts[1].trim().trim_matches('"').to_owned();
        result.push((name, dtype_str));
    }
    Ok(result)
}
```

- [ ] **Step 2: Add module declaration in `crates/numpy-rust-python/src/lib.rs`**

Add after `pub mod py_array;` (first line of the file):
```rust
pub mod py_struct_array;
```

- [ ] **Step 3: Register `PyStructuredArray` in `lib.rs`**

In `crates/numpy-rust-python/src/lib.rs`, inside the `_numpy_native` module, add after the `flagsobj` registration:

```rust
    use crate::py_struct_array::PyStructuredArray;
    use vm::class::PyClassImpl;

    /// Exposes the class as _native.StructuredArray (PascalCase function name → attribute name).
    #[allow(non_snake_case)]
    #[pyattr]
    fn StructuredArray(vm: &VirtualMachine) -> vm::builtins::PyTypeRef {
        PyStructuredArray::make_class(&vm.ctx)
    }
```

Python calls `_native.StructuredArray(fields, shape, dtype_json)` which invokes `slot_new` via the class constructor protocol. No separate `#[pyfunction]` is needed.

- [ ] **Step 4: Build and check for compile errors**

```bash
cargo build -p numpy-rust-python --release 2>&1 | grep "^error" | head -30
```

Expected: no errors. Fix any compile errors before proceeding.

- [ ] **Step 5: Commit**

```bash
git add crates/numpy-rust-python/src/py_struct_array.rs crates/numpy-rust-python/src/lib.rs
git commit -m "feat(bindings): add PyStructuredArray with field access, row access, setitem"
```

---

## Chunk 2: Python Layers + Tests

### Task 4: `StructuredDtype.descr` and `dtype` copy paths

**Files:**
- Modify: `python/numpy/_core_types.py`

The `StructuredDtype` class (`_core_types.py` ~line 464) already has `self.names`, `self.fields`, `self.itemsize`. Only `descr` is missing.

- [ ] **Step 1: Write the failing test**

In `tests/python/test_structured.py` (create if needed):

```python
"""Tests for structured array support."""
import numpy as np

passed = 0
failed = 0

def check(name, got, expected):
    global passed, failed
    if got == expected:
        passed += 1
    else:
        print(f"FAIL {name}: got {repr(got)}, expected {repr(expected)}")
        failed += 1

# Task 4: dtype.descr
dt = np.dtype([('x', 'float64'), ('y', 'int32')])
check("dtype.names", dt.names, ('x', 'y'))
check("dtype.fields keys", set(dt.fields.keys()), {'x', 'y'})
check("dtype.descr", dt.descr, [('x', 'float64'), ('y', 'int32')])
```

- [ ] **Step 2: Run to verify it fails**

```bash
./target/release/numpy-python tests/python/test_structured.py 2>&1 | tail -5
```

Expected: `FAIL dtype.descr: ...` (AttributeError or wrong value).

- [ ] **Step 3: Add `descr` to `StructuredDtype.__init__`**

In `_core_types.py`, find the end of `StructuredDtype.__init__` (after `self.str = '|V{}'.format(self.itemsize)`). Add:

```python
        # descr: list of (name, typestr) tuples — numpy compat
        self.descr = [(name, str(dt_obj)) for name, dt_obj in self._fields]
```

- [ ] **Step 4: Propagate `descr` in `dtype.__init__`**

In `dtype.__init__`, find the block `if isinstance(tp, list):` (~line 570). After `self.fields = sd.fields`, add:
```python
            self.descr = sd.descr
```

Find the block `elif isinstance(tp, StructuredDtype):` (~line 580). After `self.fields = tp.fields`, add:
```python
            self.descr = tp.descr
```

Find the block `elif isinstance(tp, dtype):` (~line 588). After the existing copies, add:
```python
            if hasattr(tp, 'descr'):
                self.descr = tp.descr
```

- [ ] **Step 5: Run the test to verify it passes**

```bash
./target/release/numpy-python tests/python/test_structured.py 2>&1 | tail -5
```

Expected: `passed: 3, failed: 0`

- [ ] **Step 6: Run full test suite to check no regressions**

```bash
bash tests/python/run_tests.sh target/release/numpy-python 2>&1 | tail -10
```

Expected: `ALL TEST FILES PASSED`

- [ ] **Step 7: Commit**

```bash
git add python/numpy/_core_types.py tests/python/test_structured.py
git commit -m "feat(dtype): add descr attribute to StructuredDtype and dtype"
```

---

### Task 5: `void`, `StructuredArray` wrapper, and `_parse_dtype_json` helper

**Files:**
- Modify: `python/numpy/__init__.py`

- [ ] **Step 1: Add failing tests to `test_structured.py`**

Add at the end of the file (after the Task 4 tests):

```python
# Task 5: void scalar
dt2 = np.dtype([('a', 'int64'), ('b', 'float64')])
row = np.void({'a': 1, 'b': 2.5}, dt2)
check("void getitem", row['a'], 1)
check("void getattr", row.b, 2.5)
check("void repr", row.__repr__(), repr((1, 2.5)))
```

- [ ] **Step 2: Run to verify fails**

```bash
./target/release/numpy-python tests/python/test_structured.py 2>&1 | tail -5
```

Expected: error — `np.void` doesn't exist yet.

- [ ] **Step 3: Add `void` class and helpers to `__init__.py`**

Find a suitable location in `__init__.py` (after the constants block, before or after the `__getattr__` def). Add:

```python
import json as _json

def _parse_dtype_json(json_str):
    """Convert dtype_json string back to StructuredDtype.
    dtype_json format: [["x","float64"],["y","int32"]]
    Returns a StructuredDtype instance.
    """
    from numpy._core_types import StructuredDtype
    pairs = _json.loads(json_str)   # list of [name, dtype_str]
    return StructuredDtype([(name, dt_str) for name, dt_str in pairs])


class void:
    """Scalar returned by arr[i] on a structured array."""

    def __init__(self, data, dtype):
        # Use object.__setattr__ throughout to avoid any __setattr__ override issues
        object.__setattr__(self, '_data', data)   # dict {fieldname: scalar_value}
        object.__setattr__(self, 'dtype', dtype)

    def __getitem__(self, key):
        return object.__getattribute__(self, '_data')[key]

    def __getattr__(self, name):
        # __getattr__ is only called when normal lookup fails
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


class StructuredArray:
    """Python wrapper for _native.StructuredArray (columnar Rust-backed structured array)."""

    def __init__(self, native_arr):
        object.__setattr__(self, '_native_arr', native_arr)
        dt = _parse_dtype_json(native_arr.dtype)
        object.__setattr__(self, 'dtype', dt)

    def __getitem__(self, key):
        native = object.__getattribute__(self, '_native_arr')
        result = native[key]
        dt = object.__getattribute__(self, 'dtype')
        # Integer key → Rust returns a list of scalars in field order
        if isinstance(result, list):
            return void({n: v for n, v in zip(dt.names, result)}, dt)
        # List of strings → Rust returns _native.StructuredArray → wrap
        if hasattr(result, 'field_names'):
            return StructuredArray(result)
        # String key → Rust returns PyNdArray column directly
        return result

    def __setitem__(self, key, val):
        native = object.__getattribute__(self, '_native_arr')
        # Unwrap StructuredArray or ndarray wrappers for Rust
        if isinstance(val, StructuredArray):
            val = object.__getattribute__(val, '_native_arr')
        native[key] = val

    def __len__(self):
        return len(object.__getattribute__(self, '_native_arr'))

    def __iter__(self):
        dt = object.__getattribute__(self, 'dtype')
        native = object.__getattribute__(self, '_native_arr')
        for i in range(len(self)):
            row_dict = native[i]   # dict from Rust __getitem__(int)
            yield void(row_dict, dt)

    @property
    def shape(self):
        return tuple(object.__getattribute__(self, '_native_arr').shape)

    @property
    def ndim(self):
        return len(self.shape)

    @property
    def field_names(self):
        return object.__getattribute__(self, '_native_arr').field_names()

    def __repr__(self):
        dt = object.__getattribute__(self, 'dtype')
        rows = list(self)
        return f"StructuredArray({rows}, dtype={dt})"
```

- [ ] **Step 4: Run tests**

```bash
./target/release/numpy-python tests/python/test_structured.py 2>&1 | tail -5
```

Expected: all void tests pass.

- [ ] **Step 5: Commit**

```bash
git add python/numpy/__init__.py
git commit -m "feat(python): add void scalar, StructuredArray wrapper, _parse_dtype_json"
```

---

### Task 6: Creation routing — `_create_structured_array`, `_create_empty_structured`, `zeros`/`empty`/`full`

**Files:**
- Modify: `python/numpy/_creation.py`
- Modify: `python/numpy/__init__.py` (zeros/empty/full wrappers)

- [ ] **Step 1: Add failing tests to `test_structured.py`**

```python
# Task 6: creation
sdt = np.dtype([('x', 'float64'), ('y', 'int32')])
data = [(1.0, 2), (3.0, 4), (5.0, 6)]
arr = np.array(data, dtype=sdt)
check("sa type", type(arr).__name__, 'StructuredArray')
check("sa shape", arr.shape, (3,))
check("sa ndim", arr.ndim, 1)
x_col = arr['x']
check("x_col[0]", float(x_col[0]), 1.0)
check("x_col[2]", float(x_col[2]), 5.0)
y_col = arr['y']
check("y_col[1]", int(y_col[1]), 4)

# zeros with structured dtype
zarr = np.zeros(3, dtype=sdt)
check("zeros sa type", type(zarr).__name__, 'StructuredArray')
zx = zarr['x']
check("zeros x[0]", float(zx[0]), 0.0)
```

- [ ] **Step 2: Run to verify it fails**

```bash
./target/release/numpy-python tests/python/test_structured.py 2>&1 | grep "FAIL\|Error" | head -10
```

Expected: `FAIL sa type` — currently routes to `_ObjectArray`.

- [ ] **Step 3: Add `_create_structured_array` and `_create_empty_structured` to `_creation.py`**

Add after the `_make_complex_array` function (~line 74):

```python
def _is_structured_dtype(dt_str):
    """Return True if dt_str represents a structured (compound) dtype."""
    from ._core_types import StructuredDtype, dtype as _dtype_cls
    try:
        parsed = _dtype_cls(dt_str) if isinstance(dt_str, str) else dt_str
        return hasattr(parsed, '_structured') or isinstance(parsed, StructuredDtype)
    except Exception:
        return False

def _create_structured_array(data, sdt):
    """Create a StructuredArray from a sequence of tuples and a StructuredDtype.

    Args:
        data: sequence of tuples, e.g. [(1.0, 2), (3.0, 4)]
        sdt: StructuredDtype or dtype wrapping a StructuredDtype
    """
    import json
    from numpy import StructuredArray
    from ._core_types import StructuredDtype
    # Unwrap if dtype wraps a StructuredDtype
    if hasattr(sdt, '_structured'):
        sdt = sdt._structured
    names = sdt.names
    nrows = len(data)
    fields = []
    for i, name in enumerate(names):
        field_dtype_obj, _ = sdt.fields[name]
        col_values = [row[i] for row in data]
        col_arr = array(col_values, dtype=field_dtype_obj)
        fields.append((name, col_arr))
    dtype_json = json.dumps([[nm, str(sdt.fields[nm][0])] for nm in names])
    native_fields = [(name, col._native if hasattr(col, '_native') else col)
                     for name, col in fields]
    native = _native.StructuredArray(native_fields, [nrows], dtype_json)
    return StructuredArray(native)

def _create_empty_structured(nrows, sdt, fill_value=0):
    """Create a zero-filled StructuredArray of shape (nrows,).

    Args:
        nrows: int — number of records
        sdt: StructuredDtype
        fill_value: scalar to fill each column (default 0)
    """
    import json
    from numpy import StructuredArray
    from ._core_types import StructuredDtype
    if hasattr(sdt, '_structured'):
        sdt = sdt._structured
    names = sdt.names
    fields = []
    for name in names:
        field_dtype_obj, _ = sdt.fields[name]
        col_arr = full(nrows, fill_value, dtype=field_dtype_obj)
        fields.append((name, col_arr))
    dtype_json = json.dumps([[nm, str(sdt.fields[nm][0])] for nm in names])
    native_fields = [(name, col._native if hasattr(col, '_native') else col)
                     for name, col in fields]
    native = _native.StructuredArray(native_fields, [nrows], dtype_json)
    return StructuredArray(native)
```

- [ ] **Step 4: Update `_array_core()` to route structured dtypes**

In `_creation.py`, find this block (~line 129):
```python
        if dt == "object" or "," in dt:
            return _ObjectArray(data if isinstance(data, (list, tuple)) else [data], dt)
```

Replace with:
```python
        if dt == "object":
            return _ObjectArray(data if isinstance(data, (list, tuple)) else [data], dt)
        if "," in dt or (hasattr(dtype, '_structured') if not isinstance(dtype, str) else False):
            # Structured dtype: route to columnar Rust-backed StructuredArray
            from ._core_types import dtype as _dtype_cls, StructuredDtype
            parsed = _dtype_cls(dtype) if not isinstance(dtype, _dtype_cls) else dtype
            if hasattr(parsed, '_structured') or isinstance(parsed, StructuredDtype):
                data_seq = data if isinstance(data, (list, tuple)) else [data]
                return _create_structured_array(data_seq, parsed)
            return _ObjectArray(data if isinstance(data, (list, tuple)) else [data], dt)
```

- [ ] **Step 5: Update `zeros`, `empty`, `full` functions to handle structured dtype**

In `_creation.py`, find the `zeros` function. Add a structured dtype check at the top:

```python
def zeros(shape, dtype=None, order='C', *, like=None):
    # Handle structured dtype
    if dtype is not None:
        from ._core_types import dtype as _dtype_cls, StructuredDtype
        parsed = _dtype_cls(dtype) if not isinstance(dtype, _dtype_cls) else dtype
        if hasattr(parsed, '_structured') or isinstance(getattr(parsed, '_structured', None), StructuredDtype):
            sdt = parsed._structured if hasattr(parsed, '_structured') else parsed
            nrows = shape if isinstance(shape, int) else shape[0]
            return _create_empty_structured(nrows, sdt, fill_value=0)
    # ... existing implementation continues unchanged
```

Do the same for `empty` and `full` (use `fill_value=0` for empty, use the actual fill value for `full`).

- [ ] **Step 6: Run the tests**

```bash
./target/release/numpy-python tests/python/test_structured.py 2>&1 | tail -10
```

Expected: all Task 6 tests pass.

- [ ] **Step 7: Run full suite**

```bash
bash tests/python/run_tests.sh target/release/numpy-python 2>&1 | tail -5
```

Expected: `ALL TEST FILES PASSED`

- [ ] **Step 8: Commit**

```bash
git add python/numpy/_creation.py
git commit -m "feat(creation): route structured dtypes to columnar StructuredArray"
```

---

### Task 7: Row access, field assignment, iteration, negative indexing

**Files:**
- Modify: `tests/python/test_structured.py` (add tests, verify all pass)

The Rust layer already handles these from Task 3. This task verifies they work end-to-end.

- [ ] **Step 1: Add tests**

```python
# Task 7: row access, field assignment, iteration
sdt3 = np.dtype([('name_x', 'float64'), ('name_y', 'int32')])
arr3 = np.array([(10.0, 1), (20.0, 2), (30.0, 3)], dtype=sdt3)

# Row access → void
row0 = arr3[0]
check("row0 type", type(row0).__name__, 'void')
check("row0['name_x']", float(row0['name_x']), 10.0)
check("row0.name_y", int(row0.name_y), 1)

# Negative indexing
row_last = arr3[-1]
check("row_last['name_x']", float(row_last['name_x']), 30.0)

# Field assignment
arr3['name_x'] = np.array([100.0, 200.0, 300.0])
check("after assign arr3['name_x'][0]", float(arr3['name_x'][0]), 100.0)

# Multi-field subset
sub = arr3[['name_x', 'name_y']]
check("sub type", type(sub).__name__, 'StructuredArray')
check("sub.dtype.names", sub.dtype.names, ('name_x', 'name_y'))

# Iteration
rows = list(arr3)
check("iter len", len(rows), 3)
check("iter row0 type", type(rows[0]).__name__, 'void')
check("iter row2 name_y", int(rows[2]['name_y']), 3)

# Empty array (n=0)
empty_sa = np.zeros(0, dtype=sdt3)
check("empty shape", empty_sa.shape, (0,))
check("empty len", len(empty_sa), 0)
```

- [ ] **Step 2: Run tests**

```bash
./target/release/numpy-python tests/python/test_structured.py 2>&1 | tail -10
```

Expected: all pass.

- [ ] **Step 3: Commit**

```bash
git add tests/python/test_structured.py
git commit -m "test(structured): add row access, field assignment, iteration, negative index tests"
```

---

### Task 8: `recarray` + `rec` module stub

**Files:**
- Modify: `python/numpy/__init__.py`

- [ ] **Step 1: Add failing tests**

```python
# Task 8: recarray
sdt4 = np.dtype([('px', 'float64'), ('py', 'float64')])
data4 = [(1.0, 2.0), (3.0, 4.0)]
ra = np.recarray((2,), dtype=sdt4)
check("recarray type", type(ra).__name__, 'recarray')
# recarray attribute access
arr_from_list = np.array(data4, dtype=sdt4)
# wrap StructuredArray as recarray via np.rec.array
rec_arr = np.rec.array(data4, dtype=sdt4)
check("rec.array type", type(rec_arr).__name__, 'recarray')
check("rec.array .px[0]", float(rec_arr.px[0]), 1.0)
check("rec.array .py[1]", float(rec_arr.py[1]), 4.0)
```

- [ ] **Step 2: Run to verify fails**

```bash
./target/release/numpy-python tests/python/test_structured.py 2>&1 | grep "FAIL\|AttributeError" | head -5
```

- [ ] **Step 3: Add `recarray` and `rec` to `__init__.py`**

```python
class recarray:
    """Structured array with attribute-style field access (np.recarray)."""

    def __init__(self, shape, dtype):
        """Create an empty (zero-filled) structured array."""
        if isinstance(shape, int):
            shape = (shape,)
        if len(shape) != 1:
            raise ValueError(
                "recarray only supports 1D arrays; got shape {}".format(shape)
            )
        arr = zeros(shape[0], dtype=dtype)   # returns StructuredArray
        object.__setattr__(self, '_arr', arr)
        object.__setattr__(self, 'dtype', arr.dtype)

    @classmethod
    def _from_structured(cls, structured_arr):
        """Wrap an existing StructuredArray as a recarray (no data copy)."""
        obj = object.__new__(cls)
        object.__setattr__(obj, '_arr', structured_arr)
        object.__setattr__(obj, 'dtype', structured_arr.dtype)
        return obj

    def __getattr__(self, name):
        try:
            arr = object.__getattribute__(self, '_arr')
            dt = object.__getattribute__(self, 'dtype')
        except AttributeError:
            raise AttributeError(name)
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


class _RecModule:
    """Stub for numpy.rec submodule."""

    def array(self, data, dtype=None, **kwargs):
        """Create a recarray from data (list of tuples)."""
        arr = array(data, dtype=dtype)   # → StructuredArray
        return recarray._from_structured(arr)


rec = _RecModule()
```

- [ ] **Step 4: Run tests**

```bash
./target/release/numpy-python tests/python/test_structured.py 2>&1 | tail -10
```

Expected: all pass.

- [ ] **Step 5: Run full suite + compat tests**

```bash
bash tests/python/run_tests.sh target/release/numpy-python 2>&1 | tail -5
./target/release/numpy-python tests/numpy_compat/run_compat.py --ci 2>&1 | tail -10
```

Expected: all pass, no new xfails.

- [ ] **Step 6: Commit**

```bash
git add python/numpy/__init__.py tests/python/test_structured.py
git commit -m "feat(python): add recarray and rec.array() with attribute-style field access"
```

---

### Task 9: Final verification and print summary

- [ ] **Step 1: Run all three test suites from the repo root**

```bash
cargo test --release 2>&1 | tail -5
bash tests/python/run_tests.sh target/release/numpy-python 2>&1 | tail -5
./target/release/numpy-python tests/numpy_compat/run_compat.py --ci 2>&1 | tail -10
```

All must pass. If any compat test fails that wasn't failing before, add to `xfail.txt` only if it's a pre-existing limitation (document why).

- [ ] **Step 2: Confirm test counts improved or unchanged**

Compare with baseline:
- Rust: 450+ tests passing
- Vendored: 1261+ passing
- Compat: 1218+ passing

- [ ] **Step 3: Final commit if any loose ends**

```bash
git add -p   # review and stage only intentional changes
git commit -m "test: final verification of structured array support"
```
