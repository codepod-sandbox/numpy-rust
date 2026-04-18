use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::RwLock;

use rustpython_vm as vm;
use vm::atomic_func;
use vm::builtins::{PyList, PySlice, PyStr, PyTuple};
use vm::protocol::{PyIterReturn, PyMappingMethods, PyNumberMethods, PySequenceMethods};
use vm::types::{AsMapping, AsNumber, AsSequence, IterNext, Iterable, Representable, SelfIter};
use vm::{AsObject, Py, PyObjectRef, PyPayload, PyRef, PyResult, TryFromObject, VirtualMachine};

use numpy_rust_core::indexing::{Scalar, SliceArg};
use numpy_rust_core::{DType, NdArray};

/// Python-visible ndarray class wrapping the core NdArray.
/// Uses `RwLock` for interior mutability so `__setitem__` can work
/// through RustPython's `&self` method signatures.
#[vm::pyclass(module = "numpy", name = "ndarray")]
#[derive(Debug, PyPayload)]
pub struct PyNdArray {
    data: RwLock<NdArray>,
    is_fortran: AtomicBool,
    is_aligned: AtomicBool,
    /// Reference to the array this view was created from (None if owner).
    base: RwLock<Option<PyObjectRef>>,
    /// Slice/index prefix used to create this view from its base array.
    view_prefix: RwLock<Option<Vec<SliceArg>>>,
}

impl Clone for PyNdArray {
    fn clone(&self) -> Self {
        Self {
            data: RwLock::new(self.data.read().unwrap().clone()),
            is_fortran: AtomicBool::new(self.is_fortran.load(Ordering::Relaxed)),
            is_aligned: AtomicBool::new(self.is_aligned.load(Ordering::Relaxed)),
            base: RwLock::new(None),
            view_prefix: RwLock::new(None),
        }
    }
}

impl PyNdArray {
    pub fn from_core(data: NdArray) -> Self {
        Self {
            data: RwLock::new(data),
            is_fortran: AtomicBool::new(false),
            is_aligned: AtomicBool::new(true),
            base: RwLock::new(None),
            view_prefix: RwLock::new(None),
        }
    }

    pub fn from_core_fortran(data: NdArray) -> Self {
        Self {
            data: RwLock::new(data),
            is_fortran: AtomicBool::new(true),
            is_aligned: AtomicBool::new(true),
            base: RwLock::new(None),
            view_prefix: RwLock::new(None),
        }
    }

    /// Create a view that references a parent array.
    pub fn from_core_with_base(data: NdArray, parent: PyObjectRef) -> Self {
        Self::from_core_with_base_and_prefix(data, parent, None)
    }

    pub fn from_core_with_base_and_prefix(
        data: NdArray,
        parent: PyObjectRef,
        prefix: Option<Vec<SliceArg>>,
    ) -> Self {
        Self {
            data: RwLock::new(data),
            is_fortran: AtomicBool::new(false),
            is_aligned: AtomicBool::new(true),
            base: RwLock::new(Some(parent)),
            view_prefix: RwLock::new(prefix),
        }
    }

    pub fn inner(&self) -> std::sync::RwLockReadGuard<'_, NdArray> {
        self.data.read().unwrap()
    }

    pub fn replace_inner(&self, data: NdArray) {
        *self.data.write().unwrap() = data;
    }

    pub fn to_py(self, vm: &VirtualMachine) -> PyObjectRef {
        self.into_pyobject(vm)
    }
}

/// Iterator over the first axis of an ndarray.
#[vm::pyclass(module = "numpy", name = "ndarray_iterator")]
#[derive(Debug, PyPayload)]
pub struct PyNdArrayIter {
    array: PyRef<PyNdArray>,
    index: std::sync::atomic::AtomicUsize,
    length: usize,
}

#[vm::pyclass(flags(DISALLOW_INSTANTIATION), with(IterNext, Iterable))]
impl PyNdArrayIter {}

impl SelfIter for PyNdArrayIter {}

impl IterNext for PyNdArrayIter {
    fn next(zelf: &Py<Self>, vm: &VirtualMachine) -> PyResult<PyIterReturn> {
        let idx = zelf
            .index
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        if idx >= zelf.length {
            return Ok(PyIterReturn::StopIteration(None));
        }
        let data = zelf.array.data.read().unwrap();
        if data.ndim() == 1 {
            let s = data.get(&[idx]).map_err(|e| numpy_err(e, vm))?;
            Ok(PyIterReturn::Return(scalar_to_py(s, vm)))
        } else {
            let result = data
                .slice(&[SliceArg::Index(idx as isize)])
                .map_err(|e| numpy_err(e, vm))?;
            Ok(PyIterReturn::Return(
                PyNdArray::from_core(result).into_pyobject(vm),
            ))
        }
    }
}

/// Mutable flat iterator/view over an ndarray.
/// Exposes linear indexing that writes through to the base array.
#[vm::pyclass(module = "numpy", name = "flatiter")]
#[derive(Debug, PyPayload)]
pub struct PyFlatIter {
    array: PyRef<PyNdArray>,
    index: std::sync::atomic::AtomicUsize,
}

#[vm::pyclass(
    flags(DISALLOW_INSTANTIATION),
    with(IterNext, Iterable, AsMapping, Representable)
)]
impl PyFlatIter {
    #[pymethod]
    fn __len__(&self) -> usize {
        self.array.data.read().unwrap().size()
    }

    #[pymethod]
    fn __getitem__(&self, key: PyObjectRef, vm: &VirtualMachine) -> PyResult<PyObjectRef> {
        let data = self.array.data.read().unwrap();
        let total = data.size();
        let idx = parse_flat_index(&key, total, vm)?;
        let coord = linear_to_coord(idx, data.shape());
        let s = data.get(&coord).map_err(|e| numpy_err(e, vm))?;
        Ok(scalar_to_py(s, vm))
    }

    #[pymethod]
    fn __setitem__(
        &self,
        key: PyObjectRef,
        value: PyObjectRef,
        vm: &VirtualMachine,
    ) -> PyResult<()> {
        let mut data = self.array.data.write().unwrap();
        let total = data.size();
        let idx = parse_flat_index(&key, total, vm)?;
        let coord = linear_to_coord(idx, data.shape());
        let scalar = py_obj_to_scalar(&value, vm)?;
        data.set(&coord, scalar).map_err(|e| numpy_err(e, vm))
    }

    #[pymethod]
    fn tolist(&self, vm: &VirtualMachine) -> PyObjectRef {
        let data = self.array.data.read().unwrap();
        let total = data.size();
        let mut out = Vec::with_capacity(total);
        for i in 0..total {
            let coord = linear_to_coord(i, data.shape());
            let s = data.get(&coord).expect("flat index must be valid");
            out.push(scalar_to_py(s, vm));
        }
        vm.ctx.new_list(out).into()
    }
}

impl SelfIter for PyFlatIter {}

impl Representable for PyFlatIter {
    fn repr_str(_zelf: &Py<Self>, _vm: &VirtualMachine) -> PyResult<String> {
        Ok("<numpy.flatiter object>".to_owned())
    }
}

impl IterNext for PyFlatIter {
    fn next(zelf: &Py<Self>, vm: &VirtualMachine) -> PyResult<PyIterReturn> {
        let idx = zelf
            .index
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        let data = zelf.array.data.read().unwrap();
        let total = data.size();
        if idx >= total {
            return Ok(PyIterReturn::StopIteration(None));
        }
        let coord = linear_to_coord(idx, data.shape());
        let s = data.get(&coord).map_err(|e| numpy_err(e, vm))?;
        Ok(PyIterReturn::Return(scalar_to_py(s, vm)))
    }
}

impl AsMapping for PyFlatIter {
    fn as_mapping() -> &'static PyMappingMethods {
        use once_cell::sync::Lazy;
        static AS_MAPPING: Lazy<PyMappingMethods> = Lazy::new(|| PyMappingMethods {
            length: atomic_func!(|mapping, _vm| {
                let zelf = PyFlatIter::mapping_downcast(mapping);
                Ok(zelf.__len__())
            }),
            subscript: atomic_func!(|mapping, needle: &vm::PyObject, vm| {
                let zelf = PyFlatIter::mapping_downcast(mapping);
                zelf.__getitem__(needle.to_owned(), vm)
            }),
            ass_subscript: atomic_func!(|mapping, needle: &vm::PyObject, value, vm| {
                let zelf = PyFlatIter::mapping_downcast(mapping);
                let value = value.ok_or_else(|| {
                    vm.new_type_error("flat iterator does not support item deletion".to_owned())
                })?;
                zelf.__setitem__(needle.to_owned(), value.to_owned(), vm)?;
                Ok(())
            }),
        });
        &AS_MAPPING
    }
}

fn parse_flat_index(key: &PyObjectRef, total: usize, vm: &VirtualMachine) -> PyResult<usize> {
    let idx: i64 = key
        .clone()
        .try_into_value(vm)
        .map_err(|_| vm.new_type_error("flat index must be an integer".to_owned()))?;
    let resolved = if idx < 0 {
        (total as i64)
            .checked_add(idx)
            .ok_or_else(|| vm.new_index_error("index out of bounds".to_owned()))?
    } else {
        idx
    };
    if resolved < 0 || resolved as usize >= total {
        return Err(
            vm.new_index_error(format!("index {} is out of bounds for size {}", idx, total))
        );
    }
    Ok(resolved as usize)
}

fn slice_arg_to_pyobject(arg: &SliceArg, vm: &VirtualMachine) -> PyResult<PyObjectRef> {
    match arg {
        SliceArg::Index(idx) => Ok(vm.ctx.new_int(*idx).into()),
        SliceArg::Range { start, stop, step } => {
            let slice_type: PyObjectRef = vm.ctx.types.slice_type.to_owned().into();
            let start_obj = start
                .map(|v| vm.ctx.new_int(v).into())
                .unwrap_or_else(|| vm.ctx.none());
            let stop_obj = stop
                .map(|v| vm.ctx.new_int(v).into())
                .unwrap_or_else(|| vm.ctx.none());
            let step_obj = vm.ctx.new_int(*step).into();
            slice_type.call(vec![start_obj, stop_obj, step_obj], vm)
        }
        SliceArg::Full => {
            let slice_type: PyObjectRef = vm.ctx.types.slice_type.to_owned().into();
            let none_obj = vm.ctx.none();
            slice_type.call(vec![none_obj.clone(), none_obj.clone(), vm.ctx.none()], vm)
        }
    }
}

fn compose_view_key(
    prefix: &[SliceArg],
    key: &PyObjectRef,
    vm: &VirtualMachine,
) -> PyResult<PyObjectRef> {
    let mut items: Vec<PyObjectRef> = prefix
        .iter()
        .map(|arg| slice_arg_to_pyobject(arg, vm))
        .collect::<PyResult<Vec<_>>>()?;
    if let Some(tuple) = key.downcast_ref::<PyTuple>() {
        items.extend(tuple.as_slice().iter().cloned());
    } else {
        items.push(key.clone());
    }
    Ok(PyTuple::new_ref(items, &vm.ctx).into())
}

fn linear_to_coord(mut idx: usize, shape: &[usize]) -> Vec<usize> {
    if shape.is_empty() {
        return vec![];
    }
    let mut coord = vec![0usize; shape.len()];
    for d in (0..shape.len()).rev() {
        let dim = shape[d];
        coord[d] = idx % dim;
        idx /= dim;
    }
    coord
}

/// Convert a NumpyError to a Python exception.
pub(crate) fn numpy_err(
    e: numpy_rust_core::NumpyError,
    vm: &VirtualMachine,
) -> vm::builtins::PyBaseExceptionRef {
    let msg = e.to_string();
    // Map index-related errors to IndexError
    if msg.contains("index") && msg.contains("out of bounds") {
        return vm.new_index_error(msg);
    }
    if msg.contains("too many indices") {
        return vm.new_index_error(msg);
    }
    vm.new_value_error(msg)
}

/// Convert a Scalar to a Python object.
pub(crate) fn scalar_to_py(s: Scalar, vm: &VirtualMachine) -> PyObjectRef {
    scalar_to_py_typed(s, None, vm)
}

/// Like scalar_to_py but interprets Int64 values as u64 when declared_dtype is UInt64.
pub(crate) fn scalar_to_py_typed(
    s: Scalar,
    declared_dtype: Option<DType>,
    vm: &VirtualMachine,
) -> PyObjectRef {
    match s {
        Scalar::Bool(v) => vm.ctx.new_bool(v).into(),
        Scalar::Int32(v) => vm.ctx.new_int(v).into(),
        Scalar::Int64(v) => {
            // UInt64 is stored as Int64; reinterpret negative values as large positive u64
            if declared_dtype == Some(DType::UInt64) {
                vm.ctx.new_int(v as u64).into()
            } else {
                vm.ctx.new_int(v).into()
            }
        }
        Scalar::Float32(v) => vm.ctx.new_float(v as f64).into(),
        Scalar::Float64(v) => vm.ctx.new_float(v).into(),
        Scalar::Complex64(v) => {
            let tup = vm.ctx.new_tuple(vec![
                vm.ctx.new_float(v.re as f64).into(),
                vm.ctx.new_float(v.im as f64).into(),
            ]);
            tup.into()
        }
        Scalar::Complex128(v) => {
            let tup = vm.ctx.new_tuple(vec![
                vm.ctx.new_float(v.re).into(),
                vm.ctx.new_float(v.im).into(),
            ]);
            tup.into()
        }
        Scalar::Str(v) => vm.ctx.new_str(v).into(),
    }
}

/// Convert a Scalar to a 0-d NdArray.
fn scalar_to_0d_ndarray(s: Scalar) -> NdArray {
    match s {
        Scalar::Float64(v) => NdArray::from_scalar(v),
        Scalar::Float32(v) => NdArray::from_scalar(v as f64),
        Scalar::Int64(v) => NdArray::from_scalar(v as f64),
        Scalar::Int32(v) => NdArray::from_scalar(v as f64),
        Scalar::Bool(v) => NdArray::from_scalar(if v { 1.0 } else { 0.0 }),
        Scalar::Complex64(_) | Scalar::Complex128(_) => NdArray::from_scalar(0.0),
        Scalar::Str(_) => NdArray::from_scalar(0.0),
    }
}

/// Convert a Python object to a generic core Scalar.
pub(crate) fn py_obj_to_scalar(obj: &PyObjectRef, vm: &VirtualMachine) -> PyResult<Scalar> {
    crate::py_creation::object_to_scalar(obj, vm)
}

/// Convert a Python object (int, float, or complex) to f64.
/// For complex numbers, takes the real part.
fn py_obj_to_f64(obj: &PyObjectRef, vm: &VirtualMachine) -> PyResult<f64> {
    if let Ok(f) = obj.clone().try_into_value::<f64>(vm) {
        return Ok(f);
    }
    if let Ok(i) = obj.clone().try_into_value::<i64>(vm) {
        return Ok(i as f64);
    }
    // Complex number: take the real part
    if let Some(c) = obj.downcast_ref::<vm::builtins::PyComplex>() {
        return Ok(c.to_complex().re);
    }
    Err(vm.new_type_error("expected a number".to_owned()))
}

/// Convert an NdArray result to PyObjectRef, returning a scalar for 0-D arrays.
pub fn ndarray_or_scalar(arr: NdArray, vm: &VirtualMachine) -> PyObjectRef {
    if arr.ndim() == 0 {
        let declared = arr.declared_dtype();
        let s = arr.get(&[]).unwrap();
        scalar_to_py_typed(s, declared, vm)
    } else {
        PyNdArray::from_core(arr).into_pyobject(vm)
    }
}

/// Parse a Python dtype string to DType.
pub fn parse_dtype(s: &str, vm: &VirtualMachine) -> PyResult<DType> {
    match s {
        "bool" | "b1" | "?" | "<class 'bool'>" | "bool_" => Ok(DType::Bool),
        // Single-char numpy type codes (also handle byte-order prefixes)
        "b" | "|b" | "<b" | ">b" => Ok(DType::Int8),
        "h" | "|h" | "<h" | ">h" => Ok(DType::Int16),
        "i" | "|i" | "<i" | ">i" => Ok(DType::Int32),
        "l" | "q" | "|l" | "<l" | ">l" | "|q" | "<q" | ">q" => Ok(DType::Int64),
        "B" | "|B" | "<B" | ">B" => Ok(DType::UInt8),
        "H" | "|H" | "<H" | ">H" => Ok(DType::UInt16),
        "I" | "|I" | "<I" | ">I" => Ok(DType::UInt32),
        "L" | "Q" | "|L" | "<L" | ">L" | "|Q" | "<Q" | ">Q" => Ok(DType::UInt64),
        "e" | "<e" | ">e" => Ok(DType::Float16),
        "f" | "<f" | ">f" => Ok(DType::Float32),
        "d" | "<d" | ">d" => Ok(DType::Float64),
        "F" | "<F" | ">F" => Ok(DType::Complex64),
        "D" | "<D" | ">D" => Ok(DType::Complex128),
        // Full name aliases
        "int8" | "i1" => Ok(DType::Int8),
        "int16" | "i2" => Ok(DType::Int16),
        "int32" | "i32" | "i4" => Ok(DType::Int32),
        "int64" | "i64" | "i8" | "int" | "<class 'int'>" | "intp" | "int_" => Ok(DType::Int64),
        "uint8" | "u1" => Ok(DType::UInt8),
        "uint16" | "u2" => Ok(DType::UInt16),
        "uint32" | "u4" => Ok(DType::UInt32),
        "uint64" | "u8" | "uintp" => Ok(DType::UInt64),
        "float16" | "f2" => Ok(DType::Float16),
        "float32" | "f32" | "f4" => Ok(DType::Float32),
        "float64" | "f64" | "f8" | "float" | "<class 'float'>" => Ok(DType::Float64),
        // Compatibility fallback: map temporal dtypes to float64 until native
        // datetime/timedelta storage exists in the Rust core.
        "timedelta64" | "datetime64" | "m8" | "M8" | "<m8" | ">m8" | "<M8" | ">M8" => {
            Ok(DType::Float64)
        }
        "complex64" | "c64" | "c8" => Ok(DType::Complex64),
        "complex128" | "c128" | "c16" | "complex" | "<class 'complex'>" => Ok(DType::Complex128),
        // longdouble/longcomplex: map to float64/complex128
        "longdouble" | "longfloat" | "g" => Ok(DType::Float64),
        "clongdouble" | "clongfloat" | "G" => Ok(DType::Complex128),
        "str" | "U" | "<class 'str'>" | "bytes" | "<class 'bytes'>" | "bytes_" => Ok(DType::Str),
        // object dtype: map to Float64 as fallback (no true object array support)
        "object" | "O" | "<class 'object'>" => Ok(DType::Float64),
        _ if s.starts_with('S') || s.starts_with('U') || s.starts_with("|S") => Ok(DType::Str),
        // Void dtype: map to UInt8 as fallback (raw bytes)
        "void" | "V" | "V0" => Ok(DType::UInt8),
        _ if s.starts_with('V') && s[1..].chars().all(|c| c.is_ascii_digit()) => Ok(DType::UInt8),
        // Strip byte-order prefix (<, >, =, |) and retry
        _ if s.len() >= 2
            && (s.starts_with('<')
                || s.starts_with('>')
                || s.starts_with('=')
                || s.starts_with('|')) =>
        {
            parse_dtype(&s[1..], vm)
        }
        _ => Err(vm.new_type_error(format!("unsupported dtype: {s}"))),
    }
}

/// Extract a shape tuple from a Python object as raw i64 values (allows -1).
fn extract_shape_i64(obj: &PyObjectRef, vm: &VirtualMachine) -> PyResult<Vec<i64>> {
    if let Some(tuple) = obj.downcast_ref::<PyTuple>() {
        let mut shape = Vec::new();
        for item in tuple.as_slice() {
            let n: i64 = item.clone().try_into_value(vm)?;
            shape.push(n);
        }
        Ok(shape)
    } else if let Some(list) = obj.downcast_ref::<PyList>() {
        let mut shape = Vec::new();
        for item in list.borrow_vec().iter() {
            let n: i64 = item.clone().try_into_value(vm)?;
            shape.push(n);
        }
        Ok(shape)
    } else if let Ok(n) = obj.clone().try_into_value::<i64>(vm) {
        Ok(vec![n])
    } else {
        Err(vm.new_type_error("shape must be a tuple, list, or integer".to_owned()))
    }
}

/// Resolve a raw shape (which may contain one -1) given the total number of elements.
fn resolve_shape(raw: &[i64], total: usize, vm: &VirtualMachine) -> PyResult<Vec<usize>> {
    let mut neg_idx: Option<usize> = None;
    let mut product: usize = 1;
    for (i, &dim) in raw.iter().enumerate() {
        if dim == -1 {
            if neg_idx.is_some() {
                return Err(vm.new_value_error("can only specify one unknown dimension".to_owned()));
            }
            neg_idx = Some(i);
        } else if dim < 0 {
            return Err(vm.new_value_error("negative dimensions are not allowed".to_owned()));
        } else {
            product = product
                .checked_mul(dim as usize)
                .ok_or_else(|| vm.new_value_error("shape too large".to_owned()))?;
        }
    }
    if let Some(idx) = neg_idx {
        if product == 0 {
            return Err(vm.new_value_error(
                "cannot reshape array of size 0 into shape with unknown dimension".to_owned(),
            ));
        }
        if !total.is_multiple_of(product) {
            return Err(vm.new_value_error(format!(
                "cannot reshape array of size {} into shape {:?}",
                total, raw
            )));
        }
        let inferred = total / product;
        Ok(raw
            .iter()
            .enumerate()
            .map(|(i, &d)| if i == idx { inferred } else { d as usize })
            .collect())
    } else {
        Ok(raw.iter().map(|&d| d as usize).collect())
    }
}

/// Extract a shape tuple from a Python object (tuple or list of ints).
pub fn extract_shape(obj: &PyObjectRef, vm: &VirtualMachine) -> PyResult<Vec<usize>> {
    let shape = if let Some(tuple) = obj.downcast_ref::<PyTuple>() {
        let mut shape = Vec::new();
        for item in tuple.as_slice() {
            let n: i64 = item.clone().try_into_value(vm)?;
            shape.push(n as usize);
        }
        shape
    } else if let Some(list) = obj.downcast_ref::<PyList>() {
        let mut shape = Vec::new();
        for item in list.borrow_vec().iter() {
            let n: i64 = item.clone().try_into_value(vm)?;
            shape.push(n as usize);
        }
        shape
    } else if let Ok(n) = obj.clone().try_into_value::<i64>(vm) {
        vec![n as usize]
    } else {
        return Err(vm.new_type_error("shape must be a tuple, list, or integer".to_owned()));
    };
    // Validate shape doesn't overflow
    numpy_rust_core::validate_shape(&shape).map_err(|e| vm.new_value_error(e.to_string()))?;
    Ok(shape)
}

#[vm::pyclass(
    flags(BASETYPE),
    with(AsNumber, AsMapping, AsSequence, Representable, Iterable)
)]
impl PyNdArray {
    /// ndarray(shape, dtype=float64) constructor – creates a zero-filled array.
    #[pyslot]
    fn slot_new(
        cls: vm::builtins::PyTypeRef,
        args: vm::function::FuncArgs,
        vm: &VirtualMachine,
    ) -> PyResult {
        // Accept positional shape or keyword shape=
        let shape_obj = if !args.args.is_empty() {
            args.args[0].clone()
        } else if let Some(s) = args.kwargs.get("shape") {
            s.clone()
        } else {
            // Default: scalar (empty shape)
            vm.ctx.new_tuple(vec![]).into()
        };
        let sh = extract_shape(&shape_obj, vm)?;
        let arr = NdArray::zeros(&sh, DType::Float64);
        Ok(Self::from_core(arr).into_ref_with_type(vm, cls)?.into())
    }

    // --- Properties ---

    #[pygetset]
    fn shape(&self, vm: &VirtualMachine) -> PyObjectRef {
        let data = self.data.read().unwrap();
        let shape: Vec<PyObjectRef> = data
            .shape()
            .iter()
            .map(|&s| vm.ctx.new_int(s).into())
            .collect();
        PyTuple::new_ref(shape, &vm.ctx).into()
    }

    #[pygetset]
    fn ndim(&self) -> usize {
        self.data.read().unwrap().ndim()
    }

    #[pygetset]
    fn size(&self) -> usize {
        self.data.read().unwrap().size()
    }

    #[pygetset]
    fn dtype(&self, vm: &VirtualMachine) -> PyResult<PyObjectRef> {
        let dtype_str = self.data.read().unwrap().dtype().to_string();
        // Import numpy.dtype and construct a dtype object
        let numpy_mod = vm.import("numpy", 0)?;
        let dtype_cls = numpy_mod.get_attr("dtype", vm)?;
        let args = vec![vm.ctx.new_str(dtype_str).into()];
        dtype_cls.call(args, vm)
    }

    #[pygetset(name = "T")]
    fn transpose_prop(zelf: PyRef<Self>, _vm: &VirtualMachine) -> PyNdArray {
        let self_obj: PyObjectRef = zelf.as_object().to_owned();
        let result =
            PyNdArray::from_core_with_base(zelf.data.read().unwrap().transpose(), self_obj);
        result
            .is_aligned
            .store(zelf.is_aligned.load(Ordering::Relaxed), Ordering::Relaxed);
        result
    }

    #[pygetset(name = "mT")]
    fn matrix_transpose_prop(zelf: PyRef<Self>, vm: &VirtualMachine) -> PyResult<PyNdArray> {
        let self_obj: PyObjectRef = zelf.as_object().to_owned();
        let inner = zelf.data.read().unwrap();
        let ndim = inner.ndim();
        if ndim < 2 {
            return Err(
                vm.new_value_error("matrix transpose with ndim < 2 is undefined".to_owned())
            );
        }
        inner
            .swapaxes(ndim - 2, ndim - 1)
            .map(|r| PyNdArray::from_core_with_base(r, self_obj))
            .map_err(|e| numpy_err(e, vm))
    }

    #[pymethod]
    fn transpose(
        zelf: PyRef<Self>,
        args: vm::function::FuncArgs,
        vm: &VirtualMachine,
    ) -> PyResult<PyNdArray> {
        let self_obj: PyObjectRef = zelf.as_object().to_owned();
        let inner = zelf.data.read().unwrap();

        // No args or single None arg => simple transpose (reverse axes)
        if args.args.is_empty() {
            return Ok(PyNdArray::from_core_with_base(inner.transpose(), self_obj));
        }

        // Extract axes from args - could be transpose(1, 0, 2) or transpose((1, 0, 2))
        let mut axes: Vec<usize> = Vec::new();
        if args.args.len() == 1 {
            let first = &args.args[0];
            if vm.is_none(first) {
                return Ok(PyNdArray::from_core_with_base(inner.transpose(), self_obj));
            }
            // Try as tuple/list
            if let Ok(tuple) = first.clone().try_into_value::<vm::builtins::PyTupleRef>(vm) {
                for item in tuple.as_slice() {
                    let idx: i64 = item.clone().try_into_value(vm)?;
                    let ndim = inner.ndim();
                    let ax = if idx < 0 {
                        (ndim as i64 + idx) as usize
                    } else {
                        idx as usize
                    };
                    axes.push(ax);
                }
            } else if let Ok(list) = first.clone().try_into_value::<vm::builtins::PyListRef>(vm) {
                for item in list.borrow_vec().iter() {
                    let idx: i64 = item.clone().try_into_value(vm)?;
                    let ndim = inner.ndim();
                    let ax = if idx < 0 {
                        (ndim as i64 + idx) as usize
                    } else {
                        idx as usize
                    };
                    axes.push(ax);
                }
            } else if let Some(arr) = first.downcast_ref::<PyNdArray>() {
                // ndarray of ints as axes
                let arr_flat = arr.inner().flatten();
                for i in 0..arr_flat.size() {
                    let s = arr_flat.get(&[i]).map_err(|e| numpy_err(e, vm))?;
                    let idx: i64 = match s {
                        Scalar::Int32(v) => v as i64,
                        Scalar::Int64(v) => v,
                        Scalar::Float32(v) => v as i64,
                        Scalar::Float64(v) => v as i64,
                        _ => return Err(vm.new_type_error("expected integer axis".to_owned())),
                    };
                    let ndim = inner.ndim();
                    let ax = if idx < 0 {
                        (ndim as i64 + idx) as usize
                    } else {
                        idx as usize
                    };
                    axes.push(ax);
                }
            } else {
                let _idx: i64 = first.clone().try_into_value(vm)?;
                // Single int arg for 1-d: just return view
                return Ok(PyNdArray::from_core_with_base(inner.transpose(), self_obj));
            }
        } else {
            // Multiple positional args: transpose(1, 0, 2)
            for arg in &args.args {
                let idx: i64 = arg.clone().try_into_value(vm)?;
                let ndim = inner.ndim();
                let ax = if idx < 0 {
                    (ndim as i64 + idx) as usize
                } else {
                    idx as usize
                };
                axes.push(ax);
            }
        }

        inner
            .transpose_axes(&axes)
            .map(|r| PyNdArray::from_core_with_base(r, self_obj))
            .map_err(|e| vm.new_value_error(e.to_string()))
    }

    #[pygetset]
    fn real(&self) -> PyNdArray {
        PyNdArray::from_core(self.data.read().unwrap().real())
    }

    #[pygetset(setter)]
    fn set_real(&self, value: PyObjectRef, vm: &VirtualMachine) -> PyResult<()> {
        let val_arr = obj_to_ndarray(&value, vm)?;
        self.data.write().unwrap().set_real(&val_arr);
        Ok(())
    }

    #[pygetset]
    fn imag(&self) -> PyNdArray {
        PyNdArray::from_core(self.data.read().unwrap().imag())
    }

    #[pygetset(setter)]
    fn set_imag(&self, value: PyObjectRef, vm: &VirtualMachine) -> PyResult<()> {
        let val_arr = obj_to_ndarray(&value, vm)?;
        self.data.write().unwrap().set_imag(&val_arr);
        Ok(())
    }

    #[pygetset(name = "__array_interface__")]
    fn array_interface(&self, vm: &VirtualMachine) -> PyResult<PyObjectRef> {
        let data = self.data.read().unwrap();
        let dtype = data.dtype();

        // Build typestr
        let typestr = match dtype {
            DType::Bool => "|b1",
            DType::Int8 => "|i1",
            DType::Int16 => "<i2",
            DType::Int32 => "<i4",
            DType::Int64 => "<i8",
            DType::UInt8 => "|u1",
            DType::UInt16 => "<u2",
            DType::UInt32 => "<u4",
            DType::UInt64 => "<u8",
            DType::Float16 => "<f2",
            DType::Float32 => "<f4",
            DType::Float64 => "<f8",
            DType::Complex64 => "<c8",
            DType::Complex128 => "<c16",
            DType::Str => "|U1",
        };

        // Build shape tuple
        let shape: Vec<PyObjectRef> = data
            .shape()
            .iter()
            .map(|&s| vm.ctx.new_int(s).into())
            .collect();
        let shape_tuple = PyTuple::new_ref(shape, &vm.ctx).into();

        // Build data tuple: (pointer_as_int, read_only_flag)
        let data_tuple = PyTuple::new_ref(
            vec![
                vm.ctx.new_int(data.raw_data_ptr()).into(),
                vm.ctx.new_bool(false).into(),
            ],
            &vm.ctx,
        )
        .into();

        // Build the dict
        let dict = vm.ctx.new_dict();
        dict.set_item("shape", shape_tuple, vm)?;
        dict.set_item("typestr", vm.ctx.new_str(typestr).into(), vm)?;
        dict.set_item("data", data_tuple, vm)?;
        dict.set_item("strides", vm.ctx.none(), vm)?;
        dict.set_item("version", vm.ctx.new_int(3).into(), vm)?;

        Ok(dict.into())
    }

    // --- Methods ---

    #[pymethod]
    fn reshape(
        zelf: PyRef<Self>,
        args: vm::function::FuncArgs,
        vm: &VirtualMachine,
    ) -> PyResult<PyObjectRef> {
        let self_obj: PyObjectRef = zelf.as_object().to_owned();
        // Accept reshape(shape) or reshape(*dims) plus optional order= and copy= kwargs.
        let shape_obj = if args.args.len() == 1 {
            args.args[0].clone()
        } else {
            // Multiple positional args: treat as individual dimensions
            let dims: Vec<i64> = args
                .args
                .iter()
                .map(|a| a.clone().try_into_value::<i64>(vm))
                .collect::<PyResult<Vec<_>>>()?;
            let list: Vec<PyObjectRef> =
                dims.into_iter().map(|d| vm.ctx.new_int(d).into()).collect();
            vm.ctx.new_tuple(list).into()
        };

        // Parse order= kwarg (default 'C')
        let order_str = args
            .kwargs
            .get("order")
            .and_then(|o| o.clone().try_into_value::<String>(vm).ok())
            .unwrap_or_else(|| "C".to_string());

        // Parse copy= kwarg: None (default), True, or False
        let copy_val = args.kwargs.get("copy").cloned();
        let copy_is_true = copy_val
            .as_ref()
            .map(|v| v.clone().try_into_value::<bool>(vm).unwrap_or(false))
            .unwrap_or(false);
        let copy_is_false = copy_val
            .as_ref()
            .map(|v| {
                // copy=False means the Python object is False (not None)
                !v.is(&vm.ctx.none()) && !v.clone().try_into_value::<bool>(vm).unwrap_or(true)
            })
            .unwrap_or(false);

        // Determine whether a reshape with the requested order would be a view.
        // C-contiguous + order C -> view; F-contiguous + order F -> view; else copy.
        let is_f = zelf.is_fortran.load(Ordering::Relaxed);
        let would_be_view = match order_str.as_str() {
            "F" => is_f,
            "A" => true, // 'A' uses source order -> always view-compatible
            _ => !is_f,  // 'C' or anything else
        };

        // copy=False with incompatible order -> error
        if copy_is_false && !would_be_view {
            return Err(
                vm.new_value_error("Unable to avoid creating a copy while reshaping.".to_string())
            );
        }

        let raw = extract_shape_i64(&shape_obj, vm)?;
        let total = zelf.data.read().unwrap().size();
        let sh = resolve_shape(&raw, total, vm)?;

        // Perform the actual reshape
        let mut result_data = zelf
            .data
            .read()
            .unwrap()
            .reshape(&sh)
            .map_err(|e| numpy_err(e, vm))?;

        // Force deep copy when: copy=True explicitly, or order change requires copy
        if copy_is_true || !would_be_view {
            result_data = result_data.copy();
        }

        let result = PyNdArray {
            data: RwLock::new(result_data),
            is_fortran: AtomicBool::new(order_str == "F" || (order_str == "A" && is_f)),
            is_aligned: AtomicBool::new(zelf.is_aligned.load(Ordering::Relaxed)),
            base: RwLock::new(if would_be_view && !copy_is_true {
                Some(self_obj.clone())
            } else {
                None
            }),
            view_prefix: RwLock::new(None),
        };

        let self_type = self_obj.class().to_owned();
        let ndarray_type = Self::class(&vm.ctx);
        if !self_type.is(ndarray_type.as_ref()) && self_type.fast_issubclass(ndarray_type.as_ref())
        {
            let obj_ref: PyObjectRef = result.into_ref_with_type(vm, self_type)?.into();
            if let Ok(finalize) = obj_ref.get_attr("__array_finalize__", vm) {
                let _ = finalize.call((self_obj,), vm);
            }
            return Ok(obj_ref);
        }

        Ok(result.into_pyobject(vm))
    }

    #[pymethod]
    fn flatten(zelf: PyRef<Self>, vm: &VirtualMachine) -> PyResult<PyObjectRef> {
        let self_obj: PyObjectRef = zelf.as_object().to_owned();
        let result = PyNdArray::from_core(zelf.data.read().unwrap().flatten());
        let self_type = self_obj.class().to_owned();
        let ndarray_type = Self::class(&vm.ctx);
        if !self_type.is(ndarray_type.as_ref()) && self_type.fast_issubclass(ndarray_type.as_ref())
        {
            let obj_ref: PyObjectRef = result.into_ref_with_type(vm, self_type)?.into();
            if let Ok(finalize) = obj_ref.get_attr("__array_finalize__", vm) {
                let _ = finalize.call((self_obj,), vm);
            }
            return Ok(obj_ref);
        }
        Ok(result.into_pyobject(vm))
    }

    #[pymethod]
    fn ravel(zelf: PyRef<Self>, vm: &VirtualMachine) -> PyResult<PyObjectRef> {
        let self_obj: PyObjectRef = zelf.as_object().to_owned();
        let result =
            PyNdArray::from_core_with_base(zelf.data.read().unwrap().ravel(), self_obj.clone());
        let self_type = self_obj.class().to_owned();
        let ndarray_type = Self::class(&vm.ctx);
        if !self_type.is(ndarray_type.as_ref()) && self_type.fast_issubclass(ndarray_type.as_ref())
        {
            let obj_ref: PyObjectRef = result.into_ref_with_type(vm, self_type)?.into();
            if let Ok(finalize) = obj_ref.get_attr("__array_finalize__", vm) {
                let _ = finalize.call((self_obj,), vm);
            }
            return Ok(obj_ref);
        }
        Ok(result.into_pyobject(vm))
    }

    #[pymethod]
    fn expand_dims(&self, axis: usize, vm: &VirtualMachine) -> PyResult<PyNdArray> {
        self.data
            .read()
            .unwrap()
            .expand_dims(axis)
            .map(PyNdArray::from_core)
            .map_err(|e| numpy_err(e, vm))
    }

    #[pymethod]
    fn squeeze(
        &self,
        axis: vm::function::OptionalArg<PyObjectRef>,
        vm: &VirtualMachine,
    ) -> PyResult<PyNdArray> {
        let ax = parse_optional_axis(axis, vm)?;
        self.data
            .read()
            .unwrap()
            .squeeze(ax)
            .map(PyNdArray::from_core)
            .map_err(|e| numpy_err(e, vm))
    }

    #[pymethod]
    fn copy(&self) -> PyNdArray {
        PyNdArray::from_core(self.data.read().unwrap().copy())
    }

    #[pymethod]
    fn astype(&self, args: vm::function::FuncArgs, vm: &VirtualMachine) -> PyResult<PyObjectRef> {
        // Accept dtype as first positional arg or as 'dtype' keyword arg
        let dtype_obj = args
            .args
            .first()
            .or_else(|| args.kwargs.get("dtype"))
            .ok_or_else(|| vm.new_type_error("astype() requires a dtype argument".to_owned()))?;
        // Accept either a string or any object convertible via str()
        let dtype_str: PyRef<PyStr> = match dtype_obj.clone().try_into_value::<PyRef<PyStr>>(vm) {
            Ok(s) => s,
            Err(_) => {
                // Convert to string using Python's str() builtin
                dtype_obj.str(vm)?
            }
        };

        // Detect structured dtype strings (comma-separated like 'i4,f8')
        // and delegate to Python's numpy._astype_structured(self, dtype_arg)
        if dtype_str.as_str().contains(',') || dtype_obj.downcast_ref::<PyList>().is_some() {
            let numpy_mod = vm.import("numpy", 0)?;
            let helper = numpy_mod.get_attr("_astype_structured", vm)?;
            let self_obj: PyObjectRef =
                PyNdArray::from_core(self.data.read().unwrap().clone()).into_pyobject(vm);
            let result = helper.call((self_obj, dtype_obj.clone()), vm)?;
            return Ok(result);
        }

        let dtype_name = dtype_str.as_str();
        if dtype_name.starts_with("datetime64[")
            || dtype_name.starts_with("timedelta64[")
            || dtype_name.starts_with("M8[")
            || dtype_name.starts_with("m8[")
        {
            let numpy_mod = vm.import("numpy", 0)?;
            let array_fn = numpy_mod.get_attr("array", vm)?;
            let self_obj: PyObjectRef =
                PyNdArray::from_core(self.data.read().unwrap().clone()).into_pyobject(vm);
            let tolist = self_obj.get_attr("tolist", vm)?;
            let data = tolist.call((), vm)?;
            let kwargs: vm::function::KwArgs =
                std::iter::once(("dtype".to_owned(), dtype_obj.clone())).collect();
            return array_fn.call(vm::function::FuncArgs::new(vec![data], kwargs), vm);
        }

        let dt = parse_dtype(dtype_name, vm)?;

        // Extract optional 'casting' keyword argument
        let casting_str = args
            .kwargs
            .get("casting")
            .map(|v| -> PyResult<String> {
                let s: PyRef<PyStr> = v
                    .clone()
                    .try_into_value(vm)
                    .map_err(|_| vm.new_type_error("casting must be a string".to_owned()))?;
                Ok(s.as_str().to_owned())
            })
            .transpose()?
            .unwrap_or_else(|| "unsafe".to_owned());

        let inner = self.data.read().unwrap();
        let from_dt = inner.dtype();
        if casting_str != "unsafe" && from_dt != dt {
            let allowed = match casting_str.as_str() {
                "no" | "equiv" => false,
                "safe" => from_dt.promote(dt) == dt,
                "same_kind" => {
                    from_dt.promote(dt) == dt
                        || (from_dt.is_float() && dt.is_float())
                        || (from_dt.is_integer() && dt.is_integer())
                        || (from_dt.is_complex() && dt.is_complex())
                }
                _ => true,
            };
            if !allowed {
                return Err(vm.new_type_error(format!(
                    "Cannot cast array data from {} to {} according to the rule '{}'",
                    from_dt, dt, casting_str
                )));
            }
        }
        Ok(PyNdArray::from_core(inner.astype(dt)).into_pyobject(vm))
    }

    /// Returns a view of the array sharing the same underlying buffer.
    /// Clone is O(1) via ArcArray reference counting.
    /// When passed a Python type (subclass of ndarray), returns an instance of that type.
    #[pymethod]
    fn view(&self, args: vm::function::FuncArgs, vm: &VirtualMachine) -> PyResult<PyObjectRef> {
        // Accept view(dtype) positionally or view(dtype=dtype) as keyword
        let dtype_opt: Option<PyObjectRef> = if let Some(pos) = args.args.first() {
            Some(pos.clone())
        } else {
            args.kwargs.get("dtype").cloned()
        };
        let data = self.data.read().unwrap();
        let is_f = self.is_fortran.load(Ordering::Relaxed);
        if let Some(dtype_obj) = dtype_opt {
            // If it's a Python type/class (e.g. MyNDArray subclass), distinguish
            // ndarray subclasses from scalar dtype classes like np.int8.
            if format!("{}", dtype_obj.class().name()) == "type" {
                let py_type: vm::builtins::PyTypeRef =
                    unsafe { dtype_obj.clone().downcast_unchecked() };
                let ndarray_type = Self::class(&vm.ctx);
                if py_type.fast_issubclass(ndarray_type.as_ref()) {
                    let shape_tuple = vm.ctx.new_tuple(
                        data.shape()
                            .iter()
                            .map(|&d| vm.ctx.new_int(d as i64).into())
                            .collect(),
                    );
                    let slot_new = ndarray_type
                        .slots
                        .new
                        .load()
                        .expect("ndarray should define tp_new");
                    let obj_ref = slot_new(
                        py_type.clone(),
                        vm::function::FuncArgs::new(
                            vec![shape_tuple.into()],
                            vm::function::KwArgs::default(),
                        ),
                        vm,
                    )?;
                    let arr_ref = PyRef::<PyNdArray>::try_from_object(vm, obj_ref.clone())
                        .map_err(|_| {
                            vm.new_type_error(format!(
                                "'{}' is not a subtype of 'ndarray'",
                                py_type.name()
                            ))
                        })?;
                    {
                        let mut target = arr_ref.data.write().unwrap();
                        *target = data.clone();
                    }
                    arr_ref.is_fortran.store(is_f, Ordering::Relaxed);
                    arr_ref
                        .is_aligned
                        .store(self.is_aligned.load(Ordering::Relaxed), Ordering::Relaxed);
                    *arr_ref.base.write().unwrap() = None;
                    *arr_ref.view_prefix.write().unwrap() = None;
                    if let Ok(finalize) = obj_ref.get_attr("__array_finalize__", vm) {
                        let self_obj = PyNdArray::from_core(data.clone()).into_pyobject(vm);
                        let _ = finalize.call((self_obj,), vm);
                    }
                    return Ok(obj_ref);
                }

                // Non-ndarray Python classes may still represent scalar dtype classes.
                if let Ok(scalar_name) = dtype_obj
                    .get_attr("_scalar_name", vm)
                    .and_then(|v| v.try_into_value::<String>(vm))
                {
                    match parse_dtype(scalar_name.as_str(), vm) {
                        Ok(target_dt) => {
                            let result = data
                                .view_as_dtype(target_dt)
                                .map_err(|e| vm.new_value_error(e.to_string()))?;
                            return Ok(PyNdArray::from_core(result).into_pyobject(vm));
                        }
                        Err(_) => return Ok(PyNdArray::from_core(data.clone()).into_pyobject(vm)),
                    }
                }
                if let Ok(type_name) = dtype_obj
                    .get_attr("__name__", vm)
                    .and_then(|v| v.try_into_value::<String>(vm))
                {
                    if let Ok(target_dt) = parse_dtype(type_name.as_str(), vm) {
                        let result = data
                            .view_as_dtype(target_dt)
                            .map_err(|e| vm.new_value_error(e.to_string()))?;
                        return Ok(PyNdArray::from_core(result).into_pyobject(vm));
                    }
                }
                return Ok(PyNdArray::from_core(data.clone()).into_pyobject(vm));
            }
            // Try to parse as dtype string
            let dtype_str: String = match dtype_obj.try_into_value::<String>(vm) {
                Ok(s) => s,
                Err(_) => return Ok(PyNdArray::from_core(data.clone()).into_pyobject(vm)),
            };
            match parse_dtype(dtype_str.as_str(), vm) {
                Ok(target_dt) => {
                    let result = data
                        .view_as_dtype(target_dt)
                        .map_err(|e| vm.new_value_error(e.to_string()))?;
                    Ok(PyNdArray::from_core(result).into_pyobject(vm))
                }
                Err(_) => Ok(PyNdArray::from_core(data.clone()).into_pyobject(vm)),
            }
        } else {
            Ok(PyNdArray::from_core(data.clone()).into_pyobject(vm))
        }
    }

    #[pymethod]
    fn sum(&self, args: vm::function::FuncArgs, vm: &VirtualMachine) -> PyResult<PyObjectRef> {
        let (axis_arg, kd) = extract_axis_keepdims(&args, vm);
        let inner = self.data.read().unwrap();
        let ax = parse_axis_arg(axis_arg, inner.ndim(), vm)?;
        match ax {
            AxisArg::None => inner.sum(None, kd),
            AxisArg::Single(a) => inner.sum(Some(a), kd),
            AxisArg::Multi(axes) => reduce_multi_axis(&inner, &axes, |arr, a| arr.sum(a, false)),
        }
        .map(|arr| ndarray_or_scalar(arr, vm))
        .map_err(|e| numpy_err(e, vm))
    }

    #[pymethod]
    fn mean(&self, args: vm::function::FuncArgs, vm: &VirtualMachine) -> PyResult<PyObjectRef> {
        let (axis_arg, kd) = extract_axis_keepdims(&args, vm);
        let inner = self.data.read().unwrap();
        let ax = parse_axis_arg(axis_arg, inner.ndim(), vm)?;
        match ax {
            AxisArg::None => inner.mean(None, kd),
            AxisArg::Single(a) => inner.mean(Some(a), kd),
            AxisArg::Multi(axes) => reduce_multi_axis(&inner, &axes, |arr, a| arr.mean(a, false)),
        }
        .map(|arr| ndarray_or_scalar(arr, vm))
        .map_err(|e| numpy_err(e, vm))
    }

    #[pymethod]
    fn min(&self, args: vm::function::FuncArgs, vm: &VirtualMachine) -> PyResult<PyObjectRef> {
        let (axis_arg, kd) = extract_axis_keepdims(&args, vm);
        let inner = self.data.read().unwrap();
        let ax = parse_axis_arg(axis_arg, inner.ndim(), vm)?;
        match ax {
            AxisArg::None => inner.min(None, kd),
            AxisArg::Single(a) => inner.min(Some(a), kd),
            AxisArg::Multi(axes) => reduce_multi_axis(&inner, &axes, |arr, a| arr.min(a, false)),
        }
        .map(|arr| ndarray_or_scalar(arr, vm))
        .map_err(|e| numpy_err(e, vm))
    }

    #[pymethod]
    fn max(&self, args: vm::function::FuncArgs, vm: &VirtualMachine) -> PyResult<PyObjectRef> {
        let (axis_arg, kd) = extract_axis_keepdims(&args, vm);
        let inner = self.data.read().unwrap();
        let ax = parse_axis_arg(axis_arg, inner.ndim(), vm)?;
        match ax {
            AxisArg::None => inner.max(None, kd),
            AxisArg::Single(a) => inner.max(Some(a), kd),
            AxisArg::Multi(axes) => reduce_multi_axis(&inner, &axes, |arr, a| arr.max(a, false)),
        }
        .map(|arr| ndarray_or_scalar(arr, vm))
        .map_err(|e| numpy_err(e, vm))
    }

    #[pymethod]
    fn std(&self, args: vm::function::FuncArgs, vm: &VirtualMachine) -> PyResult<PyObjectRef> {
        let (axis_arg, kd) = extract_axis_keepdims_at(&args, vm, 2);
        let inner = self.data.read().unwrap();
        let dd = args
            .args
            .get(1)
            .cloned()
            .or_else(|| args.kwargs.get("ddof").cloned())
            .and_then(|v| v.try_into_value::<usize>(vm).ok())
            .unwrap_or(0);
        let ax = parse_axis_arg(axis_arg, inner.ndim(), vm)?;
        match ax {
            AxisArg::None => inner.std(None, dd, kd),
            AxisArg::Single(a) => inner.std(Some(a), dd, kd),
            AxisArg::Multi(axes) => std_multi_axis(&inner, &axes, dd),
        }
        .map(|arr| ndarray_or_scalar(arr, vm))
        .map_err(|e| numpy_err(e, vm))
    }

    #[pymethod]
    fn var(&self, args: vm::function::FuncArgs, vm: &VirtualMachine) -> PyResult<PyObjectRef> {
        let (axis_arg, kd) = extract_axis_keepdims_at(&args, vm, 2);
        let inner = self.data.read().unwrap();
        let dd = args
            .args
            .get(1)
            .cloned()
            .or_else(|| args.kwargs.get("ddof").cloned())
            .and_then(|v| v.try_into_value::<usize>(vm).ok())
            .unwrap_or(0);
        let ax = parse_axis_arg(axis_arg, inner.ndim(), vm)?;
        match ax {
            AxisArg::None => inner.var(None, dd, kd),
            AxisArg::Single(a) => inner.var(Some(a), dd, kd),
            AxisArg::Multi(axes) => var_multi_axis(&inner, &axes, dd),
        }
        .map(|arr| ndarray_or_scalar(arr, vm))
        .map_err(|e| numpy_err(e, vm))
    }

    #[pymethod]
    fn argmin(
        &self,
        axis: vm::function::OptionalArg<PyObjectRef>,
        vm: &VirtualMachine,
    ) -> PyResult<PyObjectRef> {
        let ax = parse_optional_axis(axis, vm)?;
        self.data
            .read()
            .unwrap()
            .argmin(ax)
            .map(|arr| ndarray_or_scalar(arr, vm))
            .map_err(|e| numpy_err(e, vm))
    }

    #[pymethod]
    fn argmax(
        &self,
        axis: vm::function::OptionalArg<PyObjectRef>,
        vm: &VirtualMachine,
    ) -> PyResult<PyObjectRef> {
        let ax = parse_optional_axis(axis, vm)?;
        self.data
            .read()
            .unwrap()
            .argmax(ax)
            .map(|arr| ndarray_or_scalar(arr, vm))
            .map_err(|e| numpy_err(e, vm))
    }

    #[pymethod]
    fn all(&self) -> bool {
        self.data.read().unwrap().all()
    }

    #[pymethod]
    fn any(&self) -> bool {
        self.data.read().unwrap().any()
    }

    // --- Unary math ---

    #[pymethod]
    fn abs(&self) -> PyNdArray {
        PyNdArray::from_core(self.data.read().unwrap().abs())
    }

    #[pymethod]
    fn sqrt(&self) -> PyNdArray {
        PyNdArray::from_core(self.data.read().unwrap().sqrt())
    }

    #[pymethod]
    fn exp(&self) -> PyNdArray {
        PyNdArray::from_core(self.data.read().unwrap().exp())
    }

    #[pymethod]
    fn log(&self) -> PyNdArray {
        PyNdArray::from_core(self.data.read().unwrap().log())
    }

    #[pymethod]
    fn sin(&self) -> PyNdArray {
        PyNdArray::from_core(self.data.read().unwrap().sin())
    }

    #[pymethod]
    fn cos(&self) -> PyNdArray {
        PyNdArray::from_core(self.data.read().unwrap().cos())
    }

    #[pymethod]
    fn tan(&self) -> PyNdArray {
        PyNdArray::from_core(self.data.read().unwrap().tan())
    }

    #[pymethod]
    fn floor(&self, vm: &VirtualMachine) -> PyResult<PyNdArray> {
        let result = self
            .data
            .read()
            .unwrap()
            .floor()
            .map_err(|e| numpy_err(e, vm))?;
        Ok(PyNdArray::from_core(result))
    }

    #[pymethod]
    fn ceil(&self, vm: &VirtualMachine) -> PyResult<PyNdArray> {
        let result = self
            .data
            .read()
            .unwrap()
            .ceil()
            .map_err(|e| numpy_err(e, vm))?;
        Ok(PyNdArray::from_core(result))
    }

    #[pymethod]
    fn round(&self, vm: &VirtualMachine) -> PyResult<PyNdArray> {
        let result = self
            .data
            .read()
            .unwrap()
            .round()
            .map_err(|e| numpy_err(e, vm))?;
        Ok(PyNdArray::from_core(result))
    }

    #[pymethod]
    fn log10(&self) -> PyNdArray {
        PyNdArray::from_core(self.data.read().unwrap().log10())
    }

    #[pymethod]
    fn log2(&self) -> PyNdArray {
        PyNdArray::from_core(self.data.read().unwrap().log2())
    }

    #[pymethod]
    fn sinh(&self) -> PyNdArray {
        PyNdArray::from_core(self.data.read().unwrap().sinh())
    }

    #[pymethod]
    fn cosh(&self) -> PyNdArray {
        PyNdArray::from_core(self.data.read().unwrap().cosh())
    }

    #[pymethod]
    fn tanh(&self) -> PyNdArray {
        PyNdArray::from_core(self.data.read().unwrap().tanh())
    }

    #[pymethod]
    fn arcsin(&self) -> PyNdArray {
        PyNdArray::from_core(self.data.read().unwrap().arcsin())
    }

    #[pymethod]
    fn arccos(&self) -> PyNdArray {
        PyNdArray::from_core(self.data.read().unwrap().arccos())
    }

    #[pymethod]
    fn arctan(&self) -> PyNdArray {
        PyNdArray::from_core(self.data.read().unwrap().arctan())
    }

    #[pymethod]
    fn log1p(&self, vm: &VirtualMachine) -> PyResult<PyNdArray> {
        self.data
            .read()
            .unwrap()
            .log1p()
            .map(PyNdArray::from_core)
            .map_err(|e| vm.new_type_error(e.to_string()))
    }

    #[pymethod]
    fn expm1(&self, vm: &VirtualMachine) -> PyResult<PyNdArray> {
        self.data
            .read()
            .unwrap()
            .expm1()
            .map(PyNdArray::from_core)
            .map_err(|e| vm.new_type_error(e.to_string()))
    }

    #[pymethod]
    fn deg2rad(&self, vm: &VirtualMachine) -> PyResult<PyNdArray> {
        self.data
            .read()
            .unwrap()
            .deg2rad()
            .map(PyNdArray::from_core)
            .map_err(|e| vm.new_type_error(e.to_string()))
    }

    #[pymethod]
    fn rad2deg(&self, vm: &VirtualMachine) -> PyResult<PyNdArray> {
        self.data
            .read()
            .unwrap()
            .rad2deg()
            .map(PyNdArray::from_core)
            .map_err(|e| vm.new_type_error(e.to_string()))
    }

    #[pymethod]
    fn sign(&self) -> PyNdArray {
        PyNdArray::from_core(self.data.read().unwrap().sign())
    }

    #[pymethod]
    fn conj(&self) -> PyNdArray {
        PyNdArray::from_core(self.data.read().unwrap().conj())
    }

    #[pymethod]
    fn conjugate(&self) -> PyNdArray {
        PyNdArray::from_core(self.data.read().unwrap().conj())
    }

    #[pymethod]
    fn angle(&self) -> PyNdArray {
        PyNdArray::from_core(self.data.read().unwrap().angle())
    }

    // --- Element-wise checks ---

    #[pymethod]
    fn isnan(&self) -> PyNdArray {
        PyNdArray::from_core(self.data.read().unwrap().isnan())
    }

    #[pymethod]
    fn isinf(&self) -> PyNdArray {
        PyNdArray::from_core(self.data.read().unwrap().isinf())
    }

    #[pymethod]
    fn isfinite(&self) -> PyNdArray {
        PyNdArray::from_core(self.data.read().unwrap().isfinite())
    }

    #[pymethod]
    fn around(&self, decimals: vm::function::OptionalArg<i32>) -> PyNdArray {
        let d = decimals.unwrap_or(0);
        PyNdArray::from_core(self.data.read().unwrap().around(d))
    }

    // --- ndarray methods mirroring top-level functions ---

    #[pymethod]
    fn dot(&self, other: PyRef<PyNdArray>, vm: &VirtualMachine) -> PyResult<PyObjectRef> {
        let a = self.data.read().unwrap();
        let b = other.data.read().unwrap();
        numpy_rust_core::utility::dot(&a, &b)
            .map(|arr| ndarray_or_scalar(arr, vm))
            .map_err(|e| numpy_err(e, vm))
    }

    #[pymethod]
    fn swapaxes(
        zelf: PyRef<Self>,
        axis1: usize,
        axis2: usize,
        vm: &VirtualMachine,
    ) -> PyResult<PyNdArray> {
        let self_obj: PyObjectRef = zelf.as_object().to_owned();
        zelf.data
            .read()
            .unwrap()
            .swapaxes(axis1, axis2)
            .map(|r| PyNdArray::from_core_with_base(r, self_obj))
            .map_err(|e| numpy_err(e, vm))
    }

    #[pymethod]
    fn take(&self, args: vm::function::FuncArgs, vm: &VirtualMachine) -> PyResult<PyNdArray> {
        if args.args.is_empty() {
            return Err(vm.new_type_error("take() requires at least 1 argument".to_owned()));
        }
        let indices_obj = &args.args[0];
        let axis_obj = if args.args.len() > 1 {
            vm::function::OptionalArg::Present(args.args[1].clone())
        } else {
            match args.kwargs.get("axis") {
                Some(v) => vm::function::OptionalArg::Present(v.clone()),
                None => vm::function::OptionalArg::Missing,
            }
        };
        let ax = parse_optional_axis(axis_obj, vm)?;
        let idx_array = if let Ok(arr) = indices_obj.clone().downcast::<PyNdArray>() {
            arr
        } else {
            let numpy_mod = vm.import("numpy", 0)?;
            let asarray_fn = numpy_mod.get_attr("asarray", vm)?;
            let idx_arr_obj = asarray_fn.call(vec![indices_obj.clone()], vm)?;
            idx_arr_obj
                .downcast::<PyNdArray>()
                .map_err(|_| vm.new_type_error("indices must be array-like".to_owned()))?
        };
        let inner = self.data.read().unwrap();
        let idx_arr = idx_array.data.read().unwrap();
        let dim_size = match ax {
            Some(a) => inner.shape()[a],
            None => inner.size(),
        };
        let idx_vec = extract_int_indices(&idx_arr, dim_size, vm)?;
        inner
            .take(&idx_vec, ax)
            .map(PyNdArray::from_core)
            .map_err(|e| numpy_err(e, vm))
    }

    #[pymethod]
    fn repeat(
        &self,
        repeats: usize,
        axis: vm::function::OptionalArg<PyObjectRef>,
        vm: &VirtualMachine,
    ) -> PyResult<PyNdArray> {
        let ax = parse_optional_axis(axis, vm)?;
        let inner = self.data.read().unwrap();
        numpy_rust_core::manipulation::repeat(&inner, repeats, ax)
            .map(PyNdArray::from_core)
            .map_err(|e| numpy_err(e, vm))
    }

    #[pymethod]
    fn diagonal(
        &self,
        offset: vm::function::OptionalArg<i64>,
        vm: &VirtualMachine,
    ) -> PyResult<PyNdArray> {
        let off = offset.unwrap_or(0);
        let inner = self.data.read().unwrap();
        numpy_rust_core::utility::diagonal(&inner, off)
            .map(PyNdArray::from_core)
            .map_err(|e| numpy_err(e, vm))
    }

    #[pymethod]
    fn trace(
        &self,
        offset: vm::function::OptionalArg<i64>,
        vm: &VirtualMachine,
    ) -> PyResult<PyObjectRef> {
        let off = offset.unwrap_or(0);
        let inner = self.data.read().unwrap();
        let diag = numpy_rust_core::utility::diagonal(&inner, off).map_err(|e| numpy_err(e, vm))?;
        diag.sum(None, false)
            .map(|arr| ndarray_or_scalar(arr, vm))
            .map_err(|e| numpy_err(e, vm))
    }

    // --- Scalar conversion ---

    #[pymethod]
    fn float(&self, vm: &VirtualMachine) -> PyResult<f64> {
        let data = self.data.read().unwrap();
        if data.size() != 1 {
            return Err(vm.new_type_error(
                "only size-1 arrays can be converted to Python scalars".to_owned(),
            ));
        }
        let s = data
            .get(&vec![0; data.ndim()])
            .map_err(|e| numpy_err(e, vm))?;
        Ok(match s {
            Scalar::Bool(v) => {
                if v {
                    1.0
                } else {
                    0.0
                }
            }
            Scalar::Int32(v) => v as f64,
            Scalar::Int64(v) => v as f64,
            Scalar::Float32(v) => v as f64,
            Scalar::Float64(v) => v,
            Scalar::Complex64(v) => v.re as f64,
            Scalar::Complex128(v) => v.re,
            Scalar::Str(v) => v.parse::<f64>().map_err(|_| {
                vm.new_value_error(format!("could not convert string to float: '{v}'"))
            })?,
        })
    }

    #[pymethod]
    fn int(&self, vm: &VirtualMachine) -> PyResult<i64> {
        let data = self.data.read().unwrap();
        if data.size() != 1 {
            return Err(vm.new_type_error(
                "only size-1 arrays can be converted to Python scalars".to_owned(),
            ));
        }
        let s = data
            .get(&vec![0; data.ndim()])
            .map_err(|e| numpy_err(e, vm))?;
        Ok(match s {
            Scalar::Bool(v) => {
                if v {
                    1
                } else {
                    0
                }
            }
            Scalar::Int32(v) => v as i64,
            Scalar::Int64(v) => v,
            Scalar::Float32(v) => v as i64,
            Scalar::Float64(v) => v as i64,
            Scalar::Complex64(v) => v.re as i64,
            Scalar::Complex128(v) => v.re as i64,
            Scalar::Str(v) => v.parse::<i64>().map_err(|_| {
                vm.new_value_error(format!("could not convert string to int: '{v}'"))
            })?,
        })
    }

    #[pymethod]
    fn bool(&self, vm: &VirtualMachine) -> PyResult<bool> {
        let data = self.data.read().unwrap();
        if data.size() != 1 {
            return Err(vm.new_value_error(
                "The truth value of an array with more than one element is ambiguous".to_owned(),
            ));
        }
        let s = data
            .get(&vec![0; data.ndim()])
            .map_err(|e| numpy_err(e, vm))?;
        Ok(match s {
            Scalar::Bool(v) => v,
            Scalar::Int32(v) => v != 0,
            Scalar::Int64(v) => v != 0,
            Scalar::Float32(v) => v != 0.0,
            Scalar::Float64(v) => v != 0.0,
            Scalar::Complex64(v) => v.re != 0.0 || v.im != 0.0,
            Scalar::Complex128(v) => v.re != 0.0 || v.im != 0.0,
            Scalar::Str(v) => !v.is_empty(),
        })
    }

    // --- Sort / Argsort ---

    #[pymethod]
    fn sort(
        &self,
        axis: vm::function::OptionalArg<PyObjectRef>,
        vm: &VirtualMachine,
    ) -> PyResult<PyNdArray> {
        let ax = parse_optional_axis(axis, vm)?;
        self.data
            .read()
            .unwrap()
            .sort(ax)
            .map(PyNdArray::from_core)
            .map_err(|e| numpy_err(e, vm))
    }

    #[pymethod]
    fn argsort(
        &self,
        axis: vm::function::OptionalArg<PyObjectRef>,
        vm: &VirtualMachine,
    ) -> PyResult<PyObjectRef> {
        let ax = parse_optional_axis(axis, vm)?;
        self.data
            .read()
            .unwrap()
            .argsort(ax)
            .map(|arr| ndarray_or_scalar(arr, vm))
            .map_err(|e| numpy_err(e, vm))
    }

    // --- Cumulative operations ---

    #[pymethod]
    fn cumsum(
        &self,
        axis: vm::function::OptionalArg<PyObjectRef>,
        vm: &VirtualMachine,
    ) -> PyResult<PyNdArray> {
        let axis = parse_optional_axis(axis, vm)?;
        self.data
            .read()
            .unwrap()
            .cumsum(axis)
            .map(PyNdArray::from_core)
            .map_err(|e| vm.new_value_error(e.to_string()))
    }

    #[pymethod]
    fn cumprod(
        &self,
        axis: vm::function::OptionalArg<PyObjectRef>,
        vm: &VirtualMachine,
    ) -> PyResult<PyNdArray> {
        let axis = parse_optional_axis(axis, vm)?;
        self.data
            .read()
            .unwrap()
            .cumprod(axis)
            .map(PyNdArray::from_core)
            .map_err(|e| vm.new_value_error(e.to_string()))
    }

    // --- Product reduction ---

    #[pymethod]
    fn prod(&self, args: vm::function::FuncArgs, vm: &VirtualMachine) -> PyResult<PyObjectRef> {
        let (axis_arg, kd) = extract_axis_keepdims(&args, vm);
        let inner = self.data.read().unwrap();
        let ax = parse_axis_arg(axis_arg, inner.ndim(), vm)?;
        match ax {
            AxisArg::None => inner.prod(None, kd),
            AxisArg::Single(a) => inner.prod(Some(a), kd),
            AxisArg::Multi(axes) => reduce_multi_axis(&inner, &axes, |arr, a| arr.prod(a, false)),
        }
        .map(|arr| ndarray_or_scalar(arr, vm))
        .map_err(|e| vm.new_value_error(e.to_string()))
    }

    // --- Operators ---

    // Arithmetic operators are implemented via AsNumber below.
    // Reverse ops (radd, rsub, etc.) are handled by the number protocol.

    // Comparison operators are implemented via the Comparable trait below.

    // --- Indexing ---

    #[pymethod]
    fn __getitem__(
        zelf: PyRef<Self>,
        key: PyObjectRef,
        vm: &VirtualMachine,
    ) -> PyResult<PyObjectRef> {
        let parent_obj: PyObjectRef = zelf.as_object().to_owned();
        zelf.getitem_impl(key, parent_obj, vm)
    }

    /// Internal implementation of `__getitem__`.  `parent_obj` is the `PyObjectRef` of
    /// the array being indexed, stored as the `base` on returned view arrays.
    fn getitem_impl(
        &self,
        key: PyObjectRef,
        parent_obj: PyObjectRef,
        vm: &VirtualMachine,
    ) -> PyResult<PyObjectRef> {
        let data = self.data.read().unwrap();
        let build_array_result =
            |result: NdArray, prefix: Option<Vec<SliceArg>>| -> PyResult<PyObjectRef> {
                let payload =
                    PyNdArray::from_core_with_base_and_prefix(result, parent_obj.clone(), prefix);
                let parent_type = parent_obj.class().to_owned();
                let ndarray_type = Self::class(&vm.ctx);
                if !parent_type.is(ndarray_type.as_ref())
                    && parent_type.fast_issubclass(ndarray_type.as_ref())
                {
                    let obj_ref: PyObjectRef = payload.into_ref_with_type(vm, parent_type)?.into();
                    if let Ok(finalize) = obj_ref.get_attr("__array_finalize__", vm) {
                        let _ = finalize.call((parent_obj.clone(),), vm);
                    }
                    return Ok(obj_ref);
                }
                Ok(payload.to_py(vm))
            };

        // Integer index -> scalar or sub-array
        if let Ok(idx) = key.clone().try_into_value::<i64>(vm) {
            let resolved = if idx < 0 {
                (data.shape()[0] as i64 + idx) as usize
            } else {
                idx as usize
            };

            if data.ndim() == 1 {
                let s = data.get(&[resolved]).map_err(|e| numpy_err(e, vm))?;
                return Ok(scalar_to_py_typed(s, data.declared_dtype(), vm));
            } else {
                let result = data
                    .slice(&[SliceArg::Index(idx as isize)])
                    .map_err(|e| numpy_err(e, vm))?;
                return build_array_result(result, Some(vec![SliceArg::Index(idx as isize)]));
            }
        }

        // Single slice -> e.g. a[0:3], a[::2], a[:]
        if let Some(slice) = key.downcast_ref::<PySlice>() {
            let arg = py_slice_to_slice_arg(slice, vm)?;
            let result = data
                .slice(std::slice::from_ref(&arg))
                .map_err(|e| numpy_err(e, vm))?;
            return build_array_result(result, Some(vec![arg]));
        }

        // Tuple index -> multi-dimensional indexing (integers and/or slices)
        if let Some(tuple) = key.downcast_ref::<PyTuple>() {
            let items = tuple.as_slice();

            // Expand Ellipsis and handle None (newaxis) in index tuples
            let has_ellipsis = items.iter().any(|item| item.is(&vm.ctx.ellipsis));
            let is_newaxis_item =
                |item: &PyObjectRef| -> bool { vm.is_none(item) && !item.is(&vm.ctx.ellipsis) };
            let expanded_items: Vec<PyObjectRef>;
            let items = if has_ellipsis {
                let ndim = data.ndim();
                // None/newaxis don't consume array dimensions
                let indexing_count = items
                    .iter()
                    .filter(|item| !item.is(&vm.ctx.ellipsis) && !is_newaxis_item(item))
                    .count();
                let expand_count = ndim.saturating_sub(indexing_count);
                let slice_type: PyObjectRef = vm.ctx.types.slice_type.to_owned().into();
                let none_val: PyObjectRef = vm.ctx.none();
                expanded_items = items
                    .iter()
                    .flat_map(|item| {
                        if item.is(&vm.ctx.ellipsis) {
                            (0..expand_count)
                                .map(|_| {
                                    slice_type
                                        .call(vec![none_val.clone()], vm)
                                        .unwrap_or_else(|_| none_val.clone())
                                })
                                .collect::<Vec<_>>()
                        } else {
                            vec![item.clone()]
                        }
                    })
                    .collect();
                &expanded_items
            } else {
                items
            };

            // Separate None (newaxis) entries from real indexing items
            let has_any_newaxis = items.iter().any(is_newaxis_item);
            let newaxis_positions: Vec<usize>;
            let real_items_vec: Vec<PyObjectRef>;
            let items = if has_any_newaxis {
                let mut positions = Vec::new();
                let mut real = Vec::new();
                let mut out_pos = 0usize;
                for item in items.iter() {
                    if is_newaxis_item(item) {
                        positions.push(out_pos);
                        out_pos += 1;
                    } else {
                        if item.downcast_ref::<PySlice>().is_some() {
                            out_pos += 1;
                        }
                        real.push(item.clone());
                    }
                }
                newaxis_positions = positions;
                real_items_vec = real;
                &real_items_vec
            } else {
                newaxis_positions = Vec::new();
                items
            };

            // Check if any element is an ndarray or list (fancy indexing) or boolean ndarray
            let has_ndarray = items
                .iter()
                .any(|item| item.downcast_ref::<PyNdArray>().is_some());
            let has_list = items
                .iter()
                .any(|item| item.downcast_ref::<PyList>().is_some());
            let has_slice = items
                .iter()
                .any(|item| item.downcast_ref::<PySlice>().is_some());

            if has_ndarray || has_list {
                if has_list {
                    // Convert any PyList items to PyNdArray for fancy indexing
                    let converted: Vec<PyObjectRef> = items
                        .iter()
                        .map(|item| {
                            if let Some(list) = item.downcast_ref::<PyList>() {
                                let list_items = list.borrow_vec();
                                // Detect boolean list → create bool ndarray
                                let is_bool_list = !list_items.is_empty()
                                    && list_items
                                        .iter()
                                        .all(|x| x.class().is(vm.ctx.types.bool_type));
                                if is_bool_list {
                                    let bools: Vec<bool> = list_items
                                        .iter()
                                        .map(|x| x.clone().try_into_value::<bool>(vm))
                                        .collect::<PyResult<Vec<_>>>()?;
                                    let arr_data = NdArray::from_vec(bools);
                                    Ok(PyNdArray::from_core(arr_data).to_py(vm))
                                } else {
                                    let mut indices = Vec::with_capacity(list_items.len());
                                    for x in list_items.iter() {
                                        let i: i64 = x.clone().try_into_value(vm)?;
                                        indices.push(i);
                                    }
                                    let arr_data = NdArray::from_vec(indices);
                                    Ok(PyNdArray::from_core(arr_data).to_py(vm))
                                }
                            } else {
                                Ok(item.clone())
                            }
                        })
                        .collect::<PyResult<Vec<_>>>()?;
                    return multi_dim_fancy_getitem(&data, &converted, vm);
                }
                return multi_dim_fancy_getitem(&data, items, vm);
            }

            if has_slice {
                let args: Vec<SliceArg> = items
                    .iter()
                    .map(|item| py_obj_to_slice_arg(item, vm))
                    .collect::<PyResult<Vec<_>>>()?;
                let mut result = data.slice(&args).map_err(|e| numpy_err(e, vm))?;
                if !newaxis_positions.is_empty() {
                    let mut shape = result.shape().to_vec();
                    for &pos in newaxis_positions.iter() {
                        let p = if pos <= shape.len() { pos } else { shape.len() };
                        shape.insert(p, 1);
                    }
                    if let Ok(reshaped) = result.reshape(&shape) {
                        result = reshaped;
                    }
                    return build_array_result(result, Some(args.clone()));
                }
                // Return view if ndim > 0, scalar otherwise
                if result.ndim() == 0 {
                    return Ok(ndarray_or_scalar(result, vm));
                }
                return build_array_result(result, Some(args.clone()));
            }

            // All integers (with possible newaxis insertions)
            let mut indices = Vec::new();
            for item in items {
                let i: i64 = item.clone().try_into_value(vm)?;
                indices.push(i as isize);
            }
            if indices.len() == data.ndim() {
                // Empty tuple () on a 0-d array → return the 0-d array itself
                // (preserves .dtype, matches NumPy semantics for scalar preservation)
                if indices.is_empty() {
                    let result = data.clone();
                    return Ok(PyNdArray::from_core(result).to_py(vm));
                }
                let shape = data.shape();
                let usize_indices: Vec<usize> = indices
                    .iter()
                    .enumerate()
                    .map(|(axis, &i)| {
                        if i < 0 {
                            (shape[axis] as isize + i) as usize
                        } else {
                            i as usize
                        }
                    })
                    .collect();
                let s = data.get(&usize_indices).map_err(|e| numpy_err(e, vm))?;
                // If Ellipsis was present, return 0-d array instead of scalar
                if has_ellipsis {
                    let result = scalar_to_0d_ndarray(s);
                    return Ok(PyNdArray::from_core(result).to_py(vm));
                }
                return Ok(scalar_to_py_typed(s, data.declared_dtype(), vm));
            }
            let args: Vec<SliceArg> = indices.iter().map(|&i| SliceArg::Index(i)).collect();
            let result = data.slice(&args).map_err(|e| numpy_err(e, vm))?;
            return build_array_result(result, Some(args));
        }

        // NdArray index: boolean mask or integer fancy indexing
        if let Some(arr) = key.downcast_ref::<PyNdArray>() {
            let arr_data = arr.data.read().unwrap();
            if arr_data.dtype() == DType::Bool {
                let result = data.mask_select(&arr_data).map_err(|e| numpy_err(e, vm))?;
                return Ok(PyNdArray::from_core(result).to_py(vm));
            } else if arr_data.dtype().is_integer() || arr_data.dtype().is_float() {
                let idx_shape = arr_data.shape().to_vec();
                let indices = extract_int_indices(&arr_data, data.shape()[0], vm)?;
                let result = data
                    .index_select(0, &indices)
                    .map_err(|e| numpy_err(e, vm))?;
                // Preserve the shape of the index array in the result.
                // For a 1D source array: result shape = idx_shape.
                // For a nD source array indexed on axis 0: shape = (*idx_shape, *data.shape()[1..]).
                if idx_shape.len() > 1 || (idx_shape.len() == 1 && data.ndim() > 1) {
                    let mut new_shape = idx_shape.clone();
                    if data.ndim() > 1 {
                        new_shape.extend_from_slice(&data.shape()[1..]);
                    }
                    if let Ok(reshaped) = result.reshape(&new_shape) {
                        return Ok(PyNdArray::from_core(reshaped).to_py(vm));
                    }
                }
                return Ok(PyNdArray::from_core(result).to_py(vm));
            }
        }

        // List of integers -> fancy indexing: a[[0, 2, 4]]
        if let Some(list) = key.downcast_ref::<PyList>() {
            let items = list.borrow_vec();
            let dim_size = data.shape()[0];
            let mut indices = Vec::with_capacity(items.len());
            for item in items.iter() {
                let i: i64 = item.clone().try_into_value(vm)?;
                let resolved = if i < 0 {
                    (dim_size as i64 + i) as usize
                } else {
                    i as usize
                };
                indices.push(resolved);
            }
            let result = data
                .index_select(0, &indices)
                .map_err(|e| numpy_err(e, vm))?;
            return Ok(PyNdArray::from_core(result).to_py(vm));
        }

        // Ellipsis -> return a view of the entire array
        if key.is(&vm.ctx.ellipsis) {
            let result = data.clone();
            return Ok(PyNdArray::from_core_with_base(result, parent_obj).to_py(vm));
        }

        Err(vm.new_type_error("unsupported index type".to_owned()))
    }

    // --- Comparison (slot-based, returns ndarray like NumPy) ---

    #[pyslot]
    fn slot_richcompare(
        zelf: &vm::PyObject,
        other: &vm::PyObject,
        op: vm::types::PyComparisonOp,
        vm: &VirtualMachine,
    ) -> PyResult<vm::function::Either<PyObjectRef, vm::function::PyComparisonValue>> {
        let zelf = zelf
            .downcast_ref::<PyNdArray>()
            .ok_or_else(|| vm.new_type_error("expected ndarray".to_owned()))?;
        // NEP50: plain Python scalars (int, float) adopt the array's dtype (weak typing).
        let zelf_dtype = zelf.data.read().unwrap().dtype();
        // For comparisons, Python int scalars must NOT be truncated to the array dtype.
        // Use Int64 for Bool/unsigned integer targets so values are compared numerically.
        // e.g., array(True) < 2 → 1 < 2 = True (not True < True = False)
        // e.g., uint8_array >= -1 → 0 >= -1 = True (not 0 >= 255 = False)
        let nep50_target = if zelf_dtype == DType::Bool
            || zelf_dtype == DType::UInt8
            || zelf_dtype == DType::UInt16
            || zelf_dtype == DType::UInt32
            || zelf_dtype == DType::UInt64
        {
            DType::Int64
        } else {
            zelf_dtype
        };
        let other = match crate::py_creation::object_to_ndarray_weak(other, nep50_target, vm) {
            Ok(arr) => arr,
            Err(_) => {
                return Ok(vm::function::Either::B(
                    vm::function::PyComparisonValue::NotImplemented,
                ))
            }
        };
        // Ordering comparisons (lt/le/gt/ge) with complex arrays are not supported.
        let is_ordering_op = matches!(
            op,
            vm::types::PyComparisonOp::Lt
                | vm::types::PyComparisonOp::Le
                | vm::types::PyComparisonOp::Gt
                | vm::types::PyComparisonOp::Ge
        );
        if is_ordering_op {
            let other_dtype = other.dtype();
            let zelf_dtype_check = zelf.data.read().unwrap().dtype();
            if zelf_dtype_check.is_complex() || other_dtype.is_complex() {
                return Err(vm.new_type_error(
                    "type error: '<', '<=', '>', '>=' not supported for complex arrays".to_owned(),
                ));
            }
        }
        let data = zelf.data.read().unwrap();
        let result = match op {
            vm::types::PyComparisonOp::Eq => data.eq(&other),
            vm::types::PyComparisonOp::Ne => data.ne(&other),
            vm::types::PyComparisonOp::Lt => data.lt(&other),
            vm::types::PyComparisonOp::Le => data.le(&other),
            vm::types::PyComparisonOp::Gt => data.gt(&other),
            vm::types::PyComparisonOp::Ge => data.ge(&other),
        };
        match result {
            Ok(arr) => Ok(vm::function::Either::A(
                PyNdArray::from_core(arr).into_pyobject(vm),
            )),
            Err(e) => Err(numpy_err(e, vm)),
        }
    }

    // --- New methods: tolist, item ---

    #[pymethod]
    fn tolist(&self, vm: &VirtualMachine) -> PyResult<PyObjectRef> {
        let data = self.data.read().unwrap();
        Ok(ndarray_to_pylist(&data, vm))
    }

    #[pymethod(name = "__array_wrap__")]
    fn array_wrap(
        &self,
        args: vm::function::FuncArgs,
        vm: &VirtualMachine,
    ) -> PyResult<PyObjectRef> {
        let arr_obj = args.args.first().ok_or_else(|| {
            vm.new_type_error("__array_wrap__ requires at least 1 argument".to_owned())
        })?;
        let return_scalar = if args.args.len() >= 3 {
            let third = &args.args[2];
            if vm.is_none(third) {
                false
            } else {
                third.clone().try_into_value::<bool>(vm).unwrap_or(false)
            }
        } else {
            false
        };
        if let Ok(arr) = arr_obj.clone().downcast::<PyNdArray>() {
            let data = arr.data.read().unwrap();
            if return_scalar && data.ndim() == 0 && data.size() == 1 {
                let s = data.get(&[]).map_err(|e| numpy_err(e, vm))?;
                let dname = data.dtype().to_string();
                drop(data);
                let py_val = scalar_to_py(s, vm);
                if let Ok(numpy_mod) = vm.import("numpy", 0) {
                    let numpy_obj: PyObjectRef = numpy_mod;
                    let attr_name: &str = match dname.as_str() {
                        "int8" => "int8",
                        "int16" => "int16",
                        "int32" => "int32",
                        "int64" => "int64",
                        "uint8" => "uint8",
                        "uint16" => "uint16",
                        "uint32" => "uint32",
                        "uint64" => "uint64",
                        "float16" => "float16",
                        "float32" => "float32",
                        "float64" => "float64",
                        "bool" => "bool_",
                        "complex64" => "complex64",
                        "complex128" => "complex128",
                        _ => "float64",
                    };
                    if let Ok(dtype_cls) = numpy_obj.get_attr(attr_name, vm) {
                        if let Ok(result) = dtype_cls.call(vec![py_val.clone()], vm) {
                            return Ok(result);
                        }
                    }
                }
                return Ok(py_val);
            }
        }
        Ok(arr_obj.clone())
    }

    #[pymethod]
    fn item(&self, args: vm::function::FuncArgs, vm: &VirtualMachine) -> PyResult<PyObjectRef> {
        let data = self.data.read().unwrap();
        if args.args.is_empty() {
            if data.size() != 1 {
                return Err(vm.new_value_error(
                    "can only convert an array of size 1 to a Python scalar".to_owned(),
                ));
            }
            let s = data
                .get(&vec![0; data.ndim()])
                .map_err(|e| numpy_err(e, vm))?;
            return Ok(scalar_to_py_typed(s, data.declared_dtype(), vm));
        }
        let mut indices: Vec<usize> = Vec::new();
        if args.args.len() == 1 {
            let arg = &args.args[0];
            if let Some(tuple) = arg.downcast_ref::<PyTuple>() {
                for elem in tuple.as_slice() {
                    let idx: i64 = elem.clone().try_into_value(vm)?;
                    let shape = data.shape();
                    let dim_idx = indices.len();
                    if dim_idx >= data.ndim() {
                        return Err(vm.new_index_error("too many indices".to_owned()));
                    }
                    let dim = shape[dim_idx] as i64;
                    let r = if idx < 0 { dim + idx } else { idx };
                    if r < 0 || r >= dim {
                        return Err(vm.new_index_error(format!(
                            "index {} is out of bounds for axis {} with size {}",
                            idx, dim_idx, dim
                        )));
                    }
                    indices.push(r as usize);
                }
            } else {
                let flat_idx: i64 = arg.clone().try_into_value(vm)?;
                let total = data.size() as i64;
                let r = if flat_idx < 0 {
                    total + flat_idx
                } else {
                    flat_idx
                };
                if r < 0 || r >= total {
                    return Err(vm.new_index_error(format!(
                        "index {} is out of bounds for size {}",
                        flat_idx, total
                    )));
                }
                let mut rem = r as usize;
                let shape = data.shape();
                indices = vec![0; data.ndim()];
                for d in (0..data.ndim()).rev() {
                    indices[d] = rem % shape[d];
                    rem /= shape[d];
                }
            }
        } else {
            let shape = data.shape();
            for (d, arg) in args.args.iter().enumerate() {
                if d >= data.ndim() {
                    return Err(vm.new_index_error("too many indices".to_owned()));
                }
                let idx: i64 = arg.clone().try_into_value(vm)?;
                let dim = shape[d] as i64;
                let r = if idx < 0 { dim + idx } else { idx };
                if r < 0 || r >= dim {
                    return Err(vm.new_index_error(format!(
                        "index {} is out of bounds for axis {} with size {}",
                        idx, d, dim
                    )));
                }
                indices.push(r as usize);
            }
        }
        let s = data.get(&indices).map_err(|e| numpy_err(e, vm))?;
        Ok(scalar_to_py_typed(s, data.declared_dtype(), vm))
    }

    /// as_integer_ratio() for 0-d float arrays.
    #[pymethod]
    fn as_integer_ratio(&self, vm: &VirtualMachine) -> PyResult<PyObjectRef> {
        let data = self.data.read().unwrap();
        if data.ndim() != 0 {
            return Err(
                vm.new_type_error("as_integer_ratio() only supported for 0-d arrays".to_owned())
            );
        }
        let s = data.get(&[]).map_err(|e| numpy_err(e, vm))?;
        let f = match s {
            Scalar::Float32(v) => v as f64,
            Scalar::Float64(v) => v,
            _ => {
                return Err(
                    vm.new_type_error("as_integer_ratio() requires a float dtype".to_owned())
                );
            }
        };
        if f.is_infinite() {
            return Err(
                vm.new_overflow_error("cannot convert Infinity to integer ratio".to_owned())
            );
        }
        if f.is_nan() {
            return Err(vm.new_value_error("cannot convert NaN to integer ratio".to_owned()));
        }
        let py_float: PyObjectRef = vm.ctx.new_float(f).into();
        let method = py_float.get_attr("as_integer_ratio", vm)?;
        method.call((), vm)
    }

    // --- clip / fill / nonzero methods ---

    #[pymethod]
    fn clip(&self, args: vm::function::FuncArgs, vm: &VirtualMachine) -> PyResult<PyNdArray> {
        // Parse casting kwarg; None means default ("same_kind")
        let casting_rule: String = if let Some(casting_obj) = args.kwargs.get("casting") {
            if vm.is_none(casting_obj) {
                "same_kind".to_string()
            } else {
                let s = casting_obj.str(vm)?.as_str().to_string();
                let valid = ["no", "equiv", "safe", "same_kind", "unsafe"];
                if !valid.contains(&s.as_str()) {
                    return Err(vm.new_value_error(
                        "casting must be one of 'no', 'equiv', 'safe', 'same_kind', or 'unsafe'"
                            .to_string(),
                    ));
                }
                s
            }
        } else {
            "same_kind".to_string()
        };
        // Accept clip(a_min, a_max) or clip(min=, max=, out=)
        let a_min_obj = if !args.args.is_empty() {
            Some(args.args[0].clone())
        } else {
            args.kwargs.get("min").or(args.kwargs.get("a_min")).cloned()
        };
        let a_max_obj = if args.args.len() > 1 {
            Some(args.args[1].clone())
        } else {
            args.kwargs.get("max").or(args.kwargs.get("a_max")).cloned()
        };
        // Check if input is integer and bounds are explicitly float-typed.
        // Used to enforce same_kind casting when writing to integer out array.
        let self_is_integer = self.data.read().unwrap().dtype().is_integer();
        let bounds_are_float = {
            let is_py_float = |obj: &PyObjectRef| -> bool {
                // _NumpyFloatScalar subclasses float, so downcast_ref::<PyFloat> catches it
                obj.downcast_ref::<vm::builtins::PyFloat>().is_some()
                    || obj
                        .downcast_ref::<PyNdArray>()
                        .map(|nd| nd.data.read().unwrap().dtype().is_float())
                        .unwrap_or(false)
            };
            a_min_obj
                .as_ref()
                .filter(|o| !vm.is_none(o))
                .map(is_py_float)
                .unwrap_or(false)
                || a_max_obj
                    .as_ref()
                    .filter(|o| !vm.is_none(o))
                    .map(is_py_float)
                    .unwrap_or(false)
        };

        // Helper to extract complex from Python object (tuple or complex number)
        fn py_obj_to_complex(
            obj: &PyObjectRef,
            vm: &VirtualMachine,
        ) -> Option<num_complex::Complex<f64>> {
            // Try as tuple (re, im) - complex values from tolist() are tuples
            if let Some(tup) = obj.downcast_ref::<rustpython_vm::builtins::PyTuple>() {
                let elems = tup.as_slice();
                if elems.len() >= 2 {
                    let re = py_obj_to_f64(&elems[0], vm).unwrap_or(0.0);
                    let im = py_obj_to_f64(&elems[1], vm).unwrap_or(0.0);
                    return Some(num_complex::Complex::new(re, im));
                }
            }
            // Try as f64 (real number → complex with im=0)
            if let Ok(v) = py_obj_to_f64(obj, vm) {
                return Some(num_complex::Complex::new(v, 0.0));
            }
            None
        }

        // Check if array is complex dtype - use complex clip path
        let is_complex = self.data.read().unwrap().dtype().is_complex();
        if is_complex {
            let cmin = match &a_min_obj {
                None => None,
                Some(obj) if vm.is_none(obj) => None,
                Some(obj) => py_obj_to_complex(obj, vm),
            };
            let cmax = match &a_max_obj {
                None => None,
                Some(obj) if vm.is_none(obj) => None,
                Some(obj) => py_obj_to_complex(obj, vm),
            };
            let result = PyNdArray::from_core(self.data.read().unwrap().clip_complex(cmin, cmax));
            if let Some(out_obj) = args.kwargs.get("out") {
                if let Ok(out_arr) = out_obj.clone().downcast::<PyNdArray>() {
                    let core_data = result.data.read().unwrap().clone();
                    {
                        let mut out_data = out_arr.data.write().unwrap();
                        *out_data = core_data;
                    }
                    return Ok((*out_arr).clone());
                }
            }
            return Ok(result);
        }

        let min_val = match a_min_obj {
            None => None,
            Some(ref obj) if vm.is_none(obj) => None,
            Some(ref obj) => match py_obj_to_f64(obj, vm) {
                Ok(v) => Some(v),
                Err(_) => {
                    // Array-valued clip: delegate to Python np.clip
                    let numpy_mod = vm.import("numpy", 0)?;
                    let clip_fn = numpy_mod.get_attr("clip", vm)?;
                    let self_obj: PyObjectRef = self.clone().into_pyobject(vm);
                    let mut call_args = vec![self_obj, a_min_obj.unwrap().clone()];
                    if let Some(ref max_obj) = a_max_obj {
                        call_args.push(max_obj.clone());
                    }
                    let result_obj = clip_fn.call(call_args, vm)?;
                    let result_arr: PyRef<PyNdArray> = result_obj.try_into_value(vm)?;
                    let out_result = (*result_arr).clone();
                    // Handle out=
                    if let Some(out_obj) = args.kwargs.get("out") {
                        if let Ok(out_arr) = out_obj.clone().downcast::<PyNdArray>() {
                            let core_data = out_result.data.read().unwrap().clone();
                            {
                                let mut out_data = out_arr.data.write().unwrap();
                                *out_data = core_data;
                            }
                            return Ok((*out_arr).clone());
                        }
                    }
                    return Ok(out_result);
                }
            },
        };
        let max_val = match a_max_obj {
            None => None,
            Some(ref obj) if vm.is_none(obj) => None,
            Some(ref obj) => Some(py_obj_to_f64(obj, vm)?),
        };
        if min_val.is_some_and(|v| v.is_nan()) || max_val.is_some_and(|v| v.is_nan()) {
            let inner = self.data.read().unwrap();
            let shape = inner.shape().to_vec();
            let dtype = inner.dtype();
            drop(inner);
            let result = PyNdArray::from_core(numpy_rust_core::full(&shape, f64::NAN, dtype));
            if let Some(out_obj) = args.kwargs.get("out") {
                if let Ok(out_arr) = out_obj.clone().downcast::<PyNdArray>() {
                    let core_data = result.data.read().unwrap().clone();
                    {
                        let mut out_data = out_arr.data.write().unwrap();
                        *out_data = core_data;
                    }
                    return Ok((*out_arr).clone());
                }
            }
            return Ok(result);
        }
        let result = PyNdArray::from_core(self.data.read().unwrap().clip(min_val, max_val));

        // If 'out' is provided, check casting compatibility then copy data into it
        if let Some(out_obj) = args.kwargs.get("out") {
            if let Ok(out_arr) = out_obj.clone().downcast::<PyNdArray>() {
                // Enforce same_kind casting: int input + float bounds → float result,
                // which cannot be stored in an integer out without "unsafe" casting.
                if casting_rule != "unsafe" && self_is_integer && bounds_are_float {
                    let out_dt = out_arr.data.read().unwrap().dtype();
                    if out_dt.is_integer() {
                        return Err(vm.new_type_error(format!(
                            "Cannot cast ufunc 'clip' output from dtype('float64') to dtype('{}') with casting rule '{}'",
                            out_dt, casting_rule
                        )));
                    }
                }
                let core_data = result.data.read().unwrap().clone();
                drop(result);
                {
                    let mut out_data = out_arr.data.write().unwrap();
                    *out_data = core_data;
                }
                let cloned: PyNdArray = (*out_arr).clone();
                return Ok(cloned);
            }
        }
        Ok(result)
    }

    #[pymethod]
    fn fill(&self, value: PyObjectRef, vm: &VirtualMachine) -> PyResult<()> {
        let inner = self.data.read().unwrap();
        let shape = inner.shape().to_vec();
        let dtype = inner.dtype();
        drop(inner);
        let val = py_obj_to_f64(&value, vm)?;
        *self.data.write().unwrap() = numpy_rust_core::full(&shape, val, dtype);
        Ok(())
    }

    #[pymethod]
    fn nonzero(&self, vm: &VirtualMachine) -> PyResult<PyObjectRef> {
        let inner = self.data.read().unwrap();
        if inner.ndim() == 0 {
            return Err(vm.new_value_error(
                "Calling nonzero on 0d arrays is not allowed. Use np.atleast_1d(a).nonzero() instead.".to_owned()
            ));
        }
        let result = numpy_rust_core::nonzero(&inner);
        let py_arrays: Vec<PyObjectRef> = result
            .into_iter()
            .map(|arr| PyNdArray::from_core(arr).into_pyobject(vm))
            .collect();
        Ok(PyTuple::new_ref(py_arrays, &vm.ctx).into())
    }

    // --- nbytes / strides / itemsize properties ---

    #[pygetset]
    fn itemsize(&self) -> usize {
        self.data.read().unwrap().dtype().itemsize()
    }

    #[pygetset]
    fn nbytes(&self) -> usize {
        let inner = self.data.read().unwrap();
        inner.size() * inner.dtype().itemsize()
    }

    #[pygetset]
    fn strides(&self, vm: &VirtualMachine) -> PyObjectRef {
        let inner = self.data.read().unwrap();
        let itemsize = inner.dtype().itemsize();
        // Return actual strides (element strides * itemsize = byte strides)
        let elem_strides = inner.data().strides();
        let py_strides: Vec<PyObjectRef> = elem_strides
            .iter()
            .map(|&s| vm.ctx.new_int(s * itemsize as isize).into())
            .collect();
        PyTuple::new_ref(py_strides, &vm.ctx).into()
    }

    // --- String representations ---

    #[pymethod]
    fn __repr__(&self) -> String {
        let data = self.data.read().unwrap();
        format_array(&data, true)
    }

    #[pymethod]
    fn __str__(&self) -> String {
        let data = self.data.read().unwrap();
        format_array(&data, false)
    }

    #[pymethod]
    fn __len__(&self) -> usize {
        let data = self.data.read().unwrap();
        if data.ndim() == 0 {
            0
        } else {
            data.shape()[0]
        }
    }

    // --- Tier 27 Group A methods ---

    #[pymethod]
    fn ptp(&self, args: vm::function::FuncArgs, vm: &VirtualMachine) -> PyResult<PyObjectRef> {
        let axis_obj = args
            .args
            .first()
            .cloned()
            .or_else(|| args.kwargs.get("axis").cloned());
        let axis_arg = if let Some(obj) = axis_obj {
            vm::function::OptionalArg::Present(obj)
        } else {
            vm::function::OptionalArg::Missing
        };
        let kd = args
            .kwargs
            .get("keepdims")
            .and_then(|v| v.clone().try_into_value::<bool>(vm).ok())
            .unwrap_or(false);
        let ax = parse_optional_axis(axis_arg, vm)?;
        let inner = self.data.read().unwrap();
        let max_val = inner.max(ax, kd).map_err(|e| numpy_err(e, vm))?;
        let min_val = inner.min(ax, kd).map_err(|e| numpy_err(e, vm))?;
        let result = (&max_val - &min_val).map_err(|e| numpy_err(e, vm))?;
        Ok(ndarray_or_scalar(result, vm))
    }

    #[pymethod]
    fn tobytes(&self, vm: &VirtualMachine) -> PyResult<PyObjectRef> {
        let inner = self.data.read().unwrap();
        let flat = inner.flatten();
        let bytes = match flat.data() {
            numpy_rust_core::ArrayData::Float64(arr) => {
                let mut bytes = Vec::with_capacity(arr.len() * 8);
                for &val in arr.iter() {
                    bytes.extend_from_slice(&val.to_le_bytes());
                }
                bytes
            }
            numpy_rust_core::ArrayData::Int64(arr) => {
                let mut bytes = Vec::with_capacity(arr.len() * 8);
                for &val in arr.iter() {
                    bytes.extend_from_slice(&val.to_le_bytes());
                }
                bytes
            }
            numpy_rust_core::ArrayData::Int32(arr) => {
                let mut bytes = Vec::with_capacity(arr.len() * 4);
                for &val in arr.iter() {
                    bytes.extend_from_slice(&val.to_le_bytes());
                }
                bytes
            }
            numpy_rust_core::ArrayData::Float32(arr) => {
                let mut bytes = Vec::with_capacity(arr.len() * 4);
                for &val in arr.iter() {
                    bytes.extend_from_slice(&val.to_le_bytes());
                }
                bytes
            }
            numpy_rust_core::ArrayData::Bool(arr) => {
                arr.iter().map(|&b| if b { 1u8 } else { 0u8 }).collect()
            }
            _ => return Err(vm.new_type_error("tobytes not supported for this dtype".to_string())),
        };
        Ok(vm.ctx.new_bytes(bytes).into())
    }

    #[pymethod]
    fn tostring(&self, vm: &VirtualMachine) -> PyResult<PyObjectRef> {
        self.tobytes(vm)
    }

    #[pymethod]
    fn compress(
        &self,
        condition: PyRef<PyNdArray>,
        axis: vm::function::OptionalArg<PyObjectRef>,
        vm: &VirtualMachine,
    ) -> PyResult<PyNdArray> {
        let ax = parse_optional_axis(axis, vm)?;
        let inner = self.data.read().unwrap();
        let cond = condition.data.read().unwrap();
        inner
            .compress(&cond, ax)
            .map(PyNdArray::from_core)
            .map_err(|e| numpy_err(e, vm))
    }

    #[pymethod]
    fn searchsorted(
        &self,
        v: PyRef<PyNdArray>,
        side: vm::function::OptionalArg<PyRef<PyStr>>,
        vm: &VirtualMachine,
    ) -> PyResult<PyNdArray> {
        let inner = self.data.read().unwrap();
        let values = v.data.read().unwrap();
        let side_str = side
            .into_option()
            .map(|s| s.as_str().to_string())
            .unwrap_or_else(|| "left".to_string());
        inner
            .searchsorted(&values, &side_str)
            .map(PyNdArray::from_core)
            .map_err(|e| numpy_err(e, vm))
    }

    #[pymethod]
    fn partition(&self, _kth: usize, vm: &VirtualMachine) -> PyResult<PyNdArray> {
        // Full sort satisfies the partition contract
        let inner = self.data.read().unwrap();
        inner
            .sort(None)
            .map(PyNdArray::from_core)
            .map_err(|e| numpy_err(e, vm))
    }

    #[pygetset]
    fn flat(zelf: PyRef<Self>, vm: &VirtualMachine) -> PyObjectRef {
        PyFlatIter {
            array: zelf,
            index: std::sync::atomic::AtomicUsize::new(0),
        }
        .into_pyobject(vm)
    }

    // --- Tier 35 Group A methods ---

    #[pymethod]
    fn put(&self, args: vm::function::FuncArgs, vm: &VirtualMachine) -> PyResult<()> {
        if args.args.len() < 2 {
            return Err(vm.new_type_error("put() requires at least 2 arguments".to_owned()));
        }
        let numpy_mod = vm.import("numpy", 0)?;
        let asarray_fn = numpy_mod.get_attr("asarray", vm)?;
        let idx_arr_obj = asarray_fn.call(vec![args.args[0].clone()], vm)?;
        let indices = idx_arr_obj
            .downcast::<PyNdArray>()
            .map_err(|_| vm.new_type_error("indices must be array-like".to_owned()))?;
        let val_arr_obj = asarray_fn.call(vec![args.args[1].clone()], vm)?;
        let values = val_arr_obj
            .downcast::<PyNdArray>()
            .map_err(|_| vm.new_type_error("values must be array-like".to_owned()))?;
        let idx_data = indices.data.read().unwrap();
        let val_data = values.data.read().unwrap();

        let idx_flat = idx_data.flatten();
        let val_flat = val_data.flatten();
        let n_idx = idx_flat.size();
        let n_val = val_flat.size();

        drop(idx_data);
        drop(val_data);

        let mut write_guard = self.data.write().unwrap();
        let total = write_guard.size();
        let shape = write_guard.shape().to_vec();

        // Flatten to work with flat indices
        let mut flat = write_guard.flatten();

        for j in 0..n_idx {
            let idx_s = idx_flat.get(&[j]).map_err(|e| numpy_err(e, vm))?;
            let idx_val: i64 = match idx_s {
                Scalar::Int64(v) => v,
                Scalar::Int32(v) => v as i64,
                Scalar::Float64(v) => v as i64,
                Scalar::Float32(v) => v as i64,
                _ => return Err(vm.new_type_error("indices must be integer".to_owned())),
            };
            let resolved = if idx_val < 0 {
                (total as i64 + idx_val) as usize
            } else {
                idx_val as usize
            };
            if resolved >= total {
                return Err(vm.new_index_error(format!(
                    "index {} is out of bounds for axis with size {}",
                    idx_val, total
                )));
            }
            let val_s = val_flat.get(&[j % n_val]).map_err(|e| numpy_err(e, vm))?;
            flat.set(&[resolved], val_s).map_err(|e| numpy_err(e, vm))?;
        }
        *write_guard = flat.reshape(&shape).unwrap_or(flat);
        Ok(())
    }

    #[pymethod]
    fn choose(&self, args: vm::function::FuncArgs, vm: &VirtualMachine) -> PyResult<PyNdArray> {
        let choices_obj = args
            .args
            .first()
            .ok_or_else(|| vm.new_type_error("choose() requires a choices argument".to_owned()))?;

        // Accept list, tuple, or any iterable
        let items: Vec<PyObjectRef> = if let Some(list) = choices_obj.downcast_ref::<PyList>() {
            list.borrow_vec().to_vec()
        } else if let Some(tuple) = choices_obj.downcast_ref::<PyTuple>() {
            tuple.as_slice().to_vec()
        } else {
            return Err(vm.new_type_error("choose requires a sequence of arrays".to_owned()));
        };

        // Convert items to PyNdArray, wrapping scalars as needed
        // and broadcasting to match self's shape
        let self_shape = self.data.read().unwrap().shape().to_vec();
        let _self_dtype = self.data.read().unwrap().dtype();
        let py_arrays: Vec<PyRef<PyNdArray>> = items
            .iter()
            .map(|item| {
                match item.clone().try_into_value::<PyRef<PyNdArray>>(vm) {
                    Ok(arr) => Ok(arr),
                    Err(_) => {
                        // Try to convert scalar to a full array matching self's shape.
                        // Use Float64 for the choice (not the selector's dtype) to
                        // avoid truncating float values to int.
                        let val = py_obj_to_f64(item, vm)?;
                        let arr = PyNdArray::from_core(numpy_rust_core::full(
                            &self_shape,
                            val,
                            DType::Float64,
                        ));
                        let py_obj = arr.into_pyobject(vm);
                        py_obj.try_into_value::<PyRef<PyNdArray>>(vm)
                    }
                }
            })
            .collect::<PyResult<Vec<_>>>()?;
        let result = {
            let borrowed: Vec<std::sync::RwLockReadGuard<'_, NdArray>> =
                py_arrays.iter().map(|c| c.inner()).collect();
            let refs: Vec<&NdArray> = borrowed.iter().map(|r| &**r).collect();
            let inner = self.data.read().unwrap();
            numpy_rust_core::choose(&inner, &refs)
                .map(PyNdArray::from_core)
                .map_err(|e| vm.new_value_error(e.to_string()))?
        };

        // Handle out= kwarg
        if let Some(out_obj) = args.kwargs.get("out") {
            if let Ok(out_arr) = out_obj.clone().downcast::<PyNdArray>() {
                let core_data = result.data.read().unwrap().clone();
                drop(result);
                {
                    let mut out_data = out_arr.data.write().unwrap();
                    *out_data = core_data;
                }
                let cloned: PyNdArray = (*out_arr).clone();
                return Ok(cloned);
            }
        }
        Ok(result)
    }

    #[pygetset]
    fn flags(&self, vm: &VirtualMachine) -> PyResult<PyObjectRef> {
        let data = self.data.read().unwrap();
        let marked_fortran = self.is_fortran.load(Ordering::Relaxed);
        let ndim = data.ndim();
        let (is_c, is_f) = if marked_fortran && ndim > 1 {
            // Array was marked Fortran-order at creation
            (false, true)
        } else {
            (data.data().is_c_contiguous(), data.data().is_f_contiguous())
        };
        drop(data);
        let is_aligned = self.is_aligned.load(Ordering::Relaxed);
        let mut map = HashMap::new();
        map.insert("C_CONTIGUOUS".into(), is_c);
        map.insert("F_CONTIGUOUS".into(), is_f);
        let owns_data = self.base.read().unwrap().is_none();
        map.insert("OWNDATA".into(), owns_data);
        map.insert("WRITEABLE".into(), true);
        map.insert("ALIGNED".into(), is_aligned);
        map.insert("WRITEBACKIFCOPY".into(), false);
        // Short aliases
        map.insert("C".into(), is_c);
        map.insert("F".into(), is_f);
        map.insert("O".into(), owns_data);
        map.insert("W".into(), true);
        map.insert("A".into(), is_aligned);
        // Additional aliases
        map.insert("CONTIGUOUS".into(), is_c);
        map.insert("FORTRAN".into(), is_f);
        map.insert("UPDATEIFCOPY".into(), false);
        map.insert("FNC".into(), is_f);
        map.insert("FORC".into(), is_aligned);
        map.insert("BEHAVED".into(), is_aligned);
        map.insert("CARRAY".into(), is_c && is_aligned);
        map.insert("FARRAY".into(), is_f && is_aligned);
        // Lowercase aliases are handled by #[pygetset] on PyFlagsObj
        let obj = PyFlagsObj::new(map).into_ref(&vm.ctx);
        Ok(obj.into())
    }

    #[pymethod]
    fn _mark_fortran(&self) {
        self.is_fortran.store(true, Ordering::Relaxed);
    }

    #[pymethod]
    fn _mark_c_contiguous(&self) {
        self.is_fortran.store(false, Ordering::Relaxed);
    }

    #[pymethod]
    fn _set_base(&self, parent: PyObjectRef) {
        *self.base.write().unwrap() = Some(parent);
    }

    #[pymethod]
    fn _set_unaligned(&self) {
        self.is_aligned.store(false, Ordering::Relaxed);
    }

    /// Check if this array shares its underlying buffer with another ndarray.
    #[pymethod]
    fn _shares_memory_with(&self, other: PyRef<PyNdArray>) -> bool {
        let a = self.data.read().unwrap();
        let b = other.data.read().unwrap();
        a.shares_memory_with(&b)
    }

    #[pygetset]
    fn base(&self, vm: &VirtualMachine) -> PyObjectRef {
        match self.base.read().unwrap().as_ref() {
            Some(b) => b.clone(),
            None => vm.ctx.none(),
        }
    }

    #[pygetset]
    fn ctypes(&self, vm: &VirtualMachine) -> PyObjectRef {
        vm.ctx.none() // Not supported in RustPython
    }

    #[pymethod]
    fn resize(&self, new_shape: PyObjectRef, vm: &VirtualMachine) -> PyResult<PyNdArray> {
        let shape = extract_shape(&new_shape, vm)?;
        let mut total = 1usize;
        for &s in &shape {
            total *= s;
        }
        let inner = self.data.read().unwrap();
        let dtype = inner.dtype();
        if total == 0 {
            return Ok(PyNdArray::from_core(NdArray::zeros(&shape, dtype)));
        }
        let flat = inner.flatten();
        let n = flat.size();
        if n == 0 {
            return Ok(PyNdArray::from_core(NdArray::zeros(&shape, dtype)));
        }
        drop(inner);
        // Build repeated data to fill new_shape
        let mut result = NdArray::zeros(&[total], dtype);
        for i in 0..total {
            let s = flat.get(&[i % n]).map_err(|e| numpy_err(e, vm))?;
            result.set(&[i], s).map_err(|e| numpy_err(e, vm))?;
        }
        result
            .reshape(&shape)
            .map(PyNdArray::from_core)
            .map_err(|e| numpy_err(e, vm))
    }

    #[pymethod]
    fn dump(&self, _file: PyRef<PyStr>, vm: &VirtualMachine) -> PyResult<()> {
        Err(vm
            .new_not_implemented_error("dump() not supported in sandboxed environment".to_owned()))
    }

    #[pymethod]
    fn dumps(&self, vm: &VirtualMachine) -> PyResult<PyObjectRef> {
        // Return a string representation of the data
        let inner = self.data.read().unwrap();
        let s = format!("{:?}", inner.data());
        Ok(vm.ctx.new_str(s).into())
    }

    #[pymethod]
    fn byteswap(&self, _inplace: vm::function::OptionalArg<bool>) -> PyNdArray {
        // No-op: return clone (we don't deal with byte order)
        PyNdArray::from_core(self.data.read().unwrap().clone())
    }

    #[pymethod]
    #[allow(unused_variables)]
    fn setflags(&self, args: vm::function::FuncArgs) {
        // No-op: flags are fixed in our implementation
        // Accepts any combination of write=, align=, uic= kwargs
    }

    #[pymethod]
    fn getfield(&self, _dtype: PyObjectRef, vm: &VirtualMachine) -> PyResult<PyNdArray> {
        Err(vm.new_not_implemented_error("getfield() not supported".to_owned()))
    }

    #[pymethod]
    fn setfield(
        &self,
        _val: PyObjectRef,
        _dtype: PyObjectRef,
        vm: &VirtualMachine,
    ) -> PyResult<()> {
        Err(vm.new_not_implemented_error("setfield() not supported".to_owned()))
    }
}

/// Try to get an NdArray from a PyObject, auto-wrapping scalars (int/float/bool/str).
pub fn obj_to_ndarray(obj: &vm::PyObject, vm: &VirtualMachine) -> PyResult<NdArray> {
    thread_local! {
        static OBJ_TO_NDARRAY_DEPTH: std::cell::Cell<u32> = const { std::cell::Cell::new(0) };
    }
    let depth = OBJ_TO_NDARRAY_DEPTH.with(|d| {
        let cur = d.get();
        d.set(cur + 1);
        cur
    });
    // Guard: reset depth on exit
    struct DepthGuard;
    impl Drop for DepthGuard {
        fn drop(&mut self) {
            OBJ_TO_NDARRAY_DEPTH.with(|d| d.set(d.get().saturating_sub(1)));
        }
    }
    let _guard = DepthGuard;
    if depth > 50 {
        return Err(vm.new_recursion_error(
            "maximum recursion depth exceeded in array conversion".to_owned(),
        ));
    }
    crate::py_creation::object_to_ndarray(obj, vm)
}

/// Check numpy errstate for invalid operations (NaN results from subtraction etc).
/// Handles 'raise' and 'call' modes.
fn check_invalid_errstate(result_obj: &PyObjectRef, vm: &VirtualMachine) -> PyResult<()> {
    if let Ok(numpy_mod) = vm.import("numpy", 0) {
        if let Ok(err_state) = numpy_mod.get_attr("_err_state", vm) {
            if let Ok(dict) = err_state.downcast::<vm::builtins::PyDict>() {
                if let Some(val) = dict.get_item_opt("invalid", vm)? {
                    if let Ok(s) = val.try_into_value::<String>(vm) {
                        if let Some(arr) = result_obj.downcast_ref::<PyNdArray>() {
                            let data = arr.data.read().unwrap();
                            let has_nan = data.has_nan();
                            if has_nan {
                                if s == "raise" {
                                    return Err(vm.new_exception_msg(
                                        vm.ctx.exceptions.floating_point_error.to_owned(),
                                        "invalid value encountered in subtract".to_owned(),
                                    ));
                                } else if s == "call" {
                                    // Call the error callback
                                    if let Ok(geterrcall) = numpy_mod.get_attr("geterrcall", vm) {
                                        if let Ok(callback) = geterrcall.call((), vm) {
                                            if !vm.is_none(&callback) {
                                                let msg = vm.ctx.new_str(
                                                    "invalid value encountered".to_owned(),
                                                );
                                                let flag = vm.ctx.new_int(8);
                                                let _ = callback.call((msg, flag), vm);
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    Ok(())
}

/// Check numpy errstate for division warnings/errors.
/// If any element is inf/nan due to division by zero, and errstate is 'raise', raise FloatingPointError.
fn check_division_errstate(result_obj: &PyObjectRef, vm: &VirtualMachine) -> PyResult<()> {
    // Get numpy._err_state['divide']
    if let Ok(numpy_mod) = vm.import("numpy", 0) {
        if let Ok(err_state) = numpy_mod.get_attr("_err_state", vm) {
            if let Ok(dict) = err_state.downcast::<vm::builtins::PyDict>() {
                if let Some(val) = dict.get_item_opt("divide", vm)? {
                    if let Ok(s) = val.try_into_value::<String>(vm) {
                        if s == "raise" {
                            if let Some(arr) = result_obj.downcast_ref::<PyNdArray>() {
                                let data = arr.data.read().unwrap();
                                if data.has_inf() || data.has_nan() {
                                    return Err(vm.new_exception_msg(
                                        vm.ctx.exceptions.floating_point_error.to_owned(),
                                        "divide by zero encountered in divide".to_owned(),
                                    ));
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    Ok(())
}

// --- AsNumber implementation for operator dispatch ---

fn number_bin_op(
    a: &vm::PyObject,
    b: &vm::PyObject,
    op: fn(&NdArray, &NdArray) -> numpy_rust_core::Result<NdArray>,
    vm: &VirtualMachine,
) -> PyResult {
    let a_is_arr = crate::py_creation::is_array_like_object(a, vm);
    let b_is_arr = crate::py_creation::is_array_like_object(b, vm);

    let (a_arr, b_arr) = if a_is_arr && !b_is_arr {
        let a_arr = obj_to_ndarray(a, vm)?;
        let weak_target = if a_arr.dtype() == DType::Bool {
            DType::Int8
        } else {
            a_arr.dtype()
        };
        match crate::py_creation::object_to_ndarray_weak(b, weak_target, vm) {
            Ok(b_arr) => (a_arr, b_arr),
            Err(_) => return Ok(vm.ctx.not_implemented()),
        }
    } else if b_is_arr && !a_is_arr {
        let b_arr = obj_to_ndarray(b, vm)?;
        let weak_target = if b_arr.dtype() == DType::Bool {
            DType::Int8
        } else {
            b_arr.dtype()
        };
        match crate::py_creation::object_to_ndarray_weak(a, weak_target, vm) {
            Ok(a_arr) => (a_arr, b_arr),
            Err(_) => return Ok(vm.ctx.not_implemented()),
        }
    } else {
        let a_arr = match obj_to_ndarray(a, vm) {
            Ok(arr) => arr,
            Err(_) => return Ok(vm.ctx.not_implemented()),
        };
        let b_arr = match obj_to_ndarray(b, vm) {
            Ok(arr) => arr,
            Err(_) => return Ok(vm.ctx.not_implemented()),
        };
        (a_arr, b_arr)
    };
    // Catch panics from ndarray operations (e.g. shape overflow) and convert to Python errors
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| op(&a_arr, &b_arr)));
    match result {
        Ok(Ok(r)) => Ok(PyNdArray::from_core(r).into_pyobject(vm)),
        Ok(Err(e)) => Err(vm.new_value_error(e.to_string())),
        Err(panic_info) => {
            let msg = if let Some(s) = panic_info.downcast_ref::<String>() {
                s.clone()
            } else if let Some(s) = panic_info.downcast_ref::<&str>() {
                s.to_string()
            } else {
                "internal error in array operation".to_string()
            };
            Err(vm.new_runtime_error(msg))
        }
    }
}

fn number_neg(num: vm::protocol::PyNumber, vm: &VirtualMachine) -> PyResult {
    let a = num
        .downcast_ref::<PyNdArray>()
        .ok_or_else(|| vm.new_type_error("expected ndarray".to_owned()))?;
    Ok(PyNdArray::from_core(a.data.read().unwrap().neg()).into_pyobject(vm))
}

fn number_invert(num: vm::protocol::PyNumber, vm: &VirtualMachine) -> PyResult {
    let a = num
        .downcast_ref::<PyNdArray>()
        .ok_or_else(|| vm.new_type_error("expected ndarray".to_owned()))?;
    let result = a
        .data
        .read()
        .unwrap()
        .bitwise_not()
        .map_err(|e| numpy_err(e, vm))?;
    Ok(PyNdArray::from_core(result).into_pyobject(vm))
}

fn number_absolute(num: vm::protocol::PyNumber, vm: &VirtualMachine) -> PyResult {
    let a = num
        .downcast_ref::<PyNdArray>()
        .ok_or_else(|| vm.new_type_error("expected ndarray".to_owned()))?;
    Ok(PyNdArray::from_core(a.data.read().unwrap().abs()).into_pyobject(vm))
}

fn number_boolean(num: vm::protocol::PyNumber, vm: &VirtualMachine) -> PyResult<bool> {
    let a = num
        .downcast_ref::<PyNdArray>()
        .ok_or_else(|| vm.new_type_error("expected ndarray".to_owned()))?;
    let data = a.data.read().unwrap();
    if data.size() != 1 {
        return Err(vm.new_value_error(
            "The truth value of an array with more than one element is ambiguous".to_owned(),
        ));
    }
    let s = data
        .get(&vec![0; data.ndim()])
        .map_err(|e| numpy_err(e, vm))?;
    Ok(match s {
        Scalar::Bool(v) => v,
        Scalar::Int32(v) => v != 0,
        Scalar::Int64(v) => v != 0,
        Scalar::Float32(v) => v != 0.0,
        Scalar::Float64(v) => v != 0.0,
        Scalar::Complex64(v) => v.re != 0.0 || v.im != 0.0,
        Scalar::Complex128(v) => v.re != 0.0 || v.im != 0.0,
        Scalar::Str(v) => !v.is_empty(),
    })
}

fn number_float(num: vm::protocol::PyNumber, vm: &VirtualMachine) -> PyResult {
    let a = num
        .downcast_ref::<PyNdArray>()
        .ok_or_else(|| vm.new_type_error("expected ndarray".to_owned()))?;
    let data = a.data.read().unwrap();
    if data.size() != 1 {
        return Err(
            vm.new_type_error("only size-1 arrays can be converted to Python scalars".to_owned())
        );
    }
    let s = data
        .get(&vec![0; data.ndim()])
        .map_err(|e| numpy_err(e, vm))?;
    let v = match s {
        Scalar::Bool(v) => {
            if v {
                1.0
            } else {
                0.0
            }
        }
        Scalar::Int32(v) => v as f64,
        Scalar::Int64(v) => v as f64,
        Scalar::Float32(v) => v as f64,
        Scalar::Float64(v) => v,
        Scalar::Complex64(v) => v.re as f64,
        Scalar::Complex128(v) => v.re,
        Scalar::Str(v) => v
            .parse::<f64>()
            .map_err(|_| vm.new_value_error(format!("could not convert string to float: '{v}'")))?,
    };
    Ok(vm.ctx.new_float(v).into())
}

fn number_int(num: vm::protocol::PyNumber, vm: &VirtualMachine) -> PyResult {
    let a = num
        .downcast_ref::<PyNdArray>()
        .ok_or_else(|| vm.new_type_error("expected ndarray".to_owned()))?;
    let data = a.data.read().unwrap();
    if data.size() != 1 {
        return Err(
            vm.new_type_error("only size-1 arrays can be converted to Python scalars".to_owned())
        );
    }
    let s = data
        .get(&vec![0; data.ndim()])
        .map_err(|e| numpy_err(e, vm))?;
    let v = match s {
        Scalar::Bool(v) => {
            if v {
                1i64
            } else {
                0
            }
        }
        Scalar::Int32(v) => v as i64,
        Scalar::Int64(v) => v,
        Scalar::Float32(v) => v as i64,
        Scalar::Float64(v) => v as i64,
        Scalar::Complex64(v) => v.re as i64,
        Scalar::Complex128(v) => v.re as i64,
        Scalar::Str(v) => v
            .parse::<i64>()
            .map_err(|_| vm.new_value_error(format!("could not convert string to int: '{v}'")))?,
    };
    Ok(vm.ctx.new_int(v).into())
}

fn number_inplace_bin_op(
    a: &vm::PyObject,
    b: &vm::PyObject,
    op: fn(&NdArray, &NdArray) -> numpy_rust_core::Result<NdArray>,
    vm: &VirtualMachine,
) -> PyResult {
    let a_py = a
        .downcast_ref::<PyNdArray>()
        .ok_or_else(|| vm.new_type_error("expected ndarray".to_owned()))?;
    let b_arr = obj_to_ndarray(b, vm)?;
    let a_inner = a_py.data.read().unwrap().clone();
    let result = op(&a_inner, &b_arr).map_err(|e| vm.new_value_error(e.to_string()))?;
    *a_py.data.write().unwrap() = result;
    Ok(a.to_owned())
}

impl PyNdArray {
    const AS_NUMBER: PyNumberMethods = PyNumberMethods {
        add: Some(|a, b, vm| {
            let result = number_bin_op(a, b, |x, y| x + y, vm)?;
            check_invalid_errstate(&result, vm)?;
            Ok(result)
        }),
        subtract: Some(|a, b, vm| {
            let result = number_bin_op(a, b, |x, y| x - y, vm)?;
            check_invalid_errstate(&result, vm)?;
            Ok(result)
        }),
        multiply: Some(|a, b, vm| number_bin_op(a, b, |x, y| x * y, vm)),
        true_divide: Some(|a, b, vm| {
            // NumPy true_divide always returns float (integers are promoted to float64)
            let result = number_bin_op(
                a,
                b,
                |x, y| {
                    let xf = if x.dtype().is_integer() || x.dtype() == DType::Bool {
                        x.astype(DType::Float64)
                    } else {
                        x.clone()
                    };
                    let yf = if y.dtype().is_integer() || y.dtype() == DType::Bool {
                        y.astype(DType::Float64)
                    } else {
                        y.clone()
                    };
                    &xf / &yf
                },
                vm,
            )?;
            // Check numpy errstate for divide-by-zero
            check_division_errstate(&result, vm)?;
            Ok(result)
        }),
        floor_divide: Some(|a, b, vm| {
            // floordiv is not supported for complex arrays; raise TypeError like NumPy.
            if let Ok(a_arr) = obj_to_ndarray(a, vm) {
                if a_arr.dtype().is_complex() {
                    return Err(vm.new_type_error(
                        "ufunc 'floor_divide' not supported for complex types".to_owned(),
                    ));
                }
            }
            if let Ok(b_arr) = obj_to_ndarray(b, vm) {
                if b_arr.dtype().is_complex() {
                    return Err(vm.new_type_error(
                        "ufunc 'floor_divide' not supported for complex types".to_owned(),
                    ));
                }
            }
            let result = number_bin_op(a, b, |x, y| x.floor_div(y), vm)?;
            check_division_errstate(&result, vm)?;
            check_invalid_errstate(&result, vm)?;
            Ok(result)
        }),
        remainder: Some(|a, b, vm| {
            // remainder is not supported for complex arrays; raise TypeError like NumPy.
            if let Ok(a_arr) = obj_to_ndarray(a, vm) {
                if a_arr.dtype().is_complex() {
                    return Err(vm.new_type_error(
                        "ufunc 'remainder' not supported for complex types".to_owned(),
                    ));
                }
            }
            if let Ok(b_arr) = obj_to_ndarray(b, vm) {
                if b_arr.dtype().is_complex() {
                    return Err(vm.new_type_error(
                        "ufunc 'remainder' not supported for complex types".to_owned(),
                    ));
                }
            }
            number_bin_op(a, b, |x, y| x.remainder(y), vm)
        }),
        power: Some(|a, b, _modulo, vm| {
            let a_arr = obj_to_ndarray(a, vm)?;
            let b_arr = obj_to_ndarray(b, vm)?;
            a_arr
                .pow(&b_arr)
                .map(|r| PyNdArray::from_core(r).into_pyobject(vm))
                .map_err(|e| vm.new_value_error(e.to_string()))
        }),
        negative: Some(number_neg),
        positive: Some(|num, vm| {
            let a = num
                .downcast_ref::<PyNdArray>()
                .ok_or_else(|| vm.new_type_error("expected ndarray".to_owned()))?;
            let data = a.data.read().unwrap();
            Ok(PyNdArray::from_core(data.clone()).into_pyobject(vm))
        }),
        int: Some(number_int),
        float: Some(number_float),
        and: Some(|a, b, vm| number_bin_op(a, b, |x, y| x.bitwise_and(y), vm)),
        or: Some(|a, b, vm| number_bin_op(a, b, |x, y| x.bitwise_or(y), vm)),
        invert: Some(number_invert),
        matrix_multiply: Some(|a, b, vm| {
            let a_arr = obj_to_ndarray(a, vm)?;
            let b_arr = obj_to_ndarray(b, vm)?;
            numpy_rust_core::dot(&a_arr, &b_arr)
                .map(|r| ndarray_or_scalar(r, vm))
                .map_err(|e| vm.new_value_error(e.to_string()))
        }),
        inplace_add: Some(|a, b, vm| number_inplace_bin_op(a, b, |x, y| x + y, vm)),
        inplace_subtract: Some(|a, b, vm| number_inplace_bin_op(a, b, |x, y| x - y, vm)),
        inplace_multiply: Some(|a, b, vm| number_inplace_bin_op(a, b, |x, y| x * y, vm)),
        inplace_true_divide: Some(|a, b, vm| number_inplace_bin_op(a, b, |x, y| x / y, vm)),
        inplace_floor_divide: Some(|a, b, vm| {
            number_inplace_bin_op(a, b, |x, y| x.floor_div(y), vm)
        }),
        inplace_remainder: Some(|a, b, vm| number_inplace_bin_op(a, b, |x, y| x.remainder(y), vm)),
        inplace_power: Some(|a, b, _modulo, vm| {
            let a_py = a
                .downcast_ref::<PyNdArray>()
                .ok_or_else(|| vm.new_type_error("expected ndarray".to_owned()))?;
            let b_arr = obj_to_ndarray(b, vm)?;
            let a_inner = a_py.data.read().unwrap().clone();
            let result = a_inner
                .pow(&b_arr)
                .map_err(|e| vm.new_value_error(e.to_string()))?;
            *a_py.data.write().unwrap() = result;
            Ok(a.to_owned())
        }),
        inplace_matrix_multiply: Some(|a, b, vm| {
            let a_py = a
                .downcast_ref::<PyNdArray>()
                .ok_or_else(|| vm.new_type_error("expected ndarray".to_owned()))?;
            let b_arr = obj_to_ndarray(b, vm)?;
            let a_inner = a_py.data.read().unwrap().clone();
            let result = numpy_rust_core::dot(&a_inner, &b_arr)
                .map_err(|e| vm.new_value_error(e.to_string()))?;
            *a_py.data.write().unwrap() = result;
            Ok(a.to_owned())
        }),
        xor: Some(|a, b, vm| number_bin_op(a, b, |x, y| x.bitwise_xor(y), vm)),
        lshift: Some(|a, b, vm| number_bin_op(a, b, |x, y| x.left_shift(y), vm)),
        rshift: Some(|a, b, vm| number_bin_op(a, b, |x, y| x.right_shift(y), vm)),
        inplace_and: Some(|a, b, vm| number_inplace_bin_op(a, b, |x, y| x.bitwise_and(y), vm)),
        inplace_or: Some(|a, b, vm| number_inplace_bin_op(a, b, |x, y| x.bitwise_or(y), vm)),
        inplace_xor: Some(|a, b, vm| number_inplace_bin_op(a, b, |x, y| x.bitwise_xor(y), vm)),
        inplace_lshift: Some(|a, b, vm| number_inplace_bin_op(a, b, |x, y| x.left_shift(y), vm)),
        inplace_rshift: Some(|a, b, vm| number_inplace_bin_op(a, b, |x, y| x.right_shift(y), vm)),
        absolute: Some(number_absolute),
        boolean: Some(number_boolean),
        index: Some(number_int),
        ..PyNumberMethods::NOT_IMPLEMENTED
    };
}

impl AsNumber for PyNdArray {
    fn as_number() -> &'static PyNumberMethods {
        static AS_NUMBER: PyNumberMethods = PyNdArray::AS_NUMBER;
        &AS_NUMBER
    }
}

impl AsMapping for PyNdArray {
    fn as_mapping() -> &'static PyMappingMethods {
        use once_cell::sync::Lazy;
        static AS_MAPPING: Lazy<PyMappingMethods> = Lazy::new(|| PyMappingMethods {
            length: atomic_func!(|mapping, _vm| {
                let zelf = PyNdArray::mapping_downcast(mapping);
                let data = zelf.data.read().unwrap();
                Ok(if data.ndim() == 0 { 0 } else { data.shape()[0] })
            }),
            subscript: atomic_func!(|mapping, needle: &vm::PyObject, vm| {
                let zelf = PyNdArray::mapping_downcast(mapping);
                let parent_obj = zelf.as_object().to_owned();
                zelf.getitem_impl(needle.to_owned(), parent_obj, vm)
            }),
            ass_subscript: atomic_func!(|mapping, needle: &vm::PyObject, value, vm| {
                let zelf = PyNdArray::mapping_downcast(mapping);
                match value {
                    Some(value) => setitem_impl(zelf, needle.to_owned(), value.to_owned(), vm),
                    None => {
                        Err(vm.new_type_error("ndarray does not support item deletion".to_owned()))
                    }
                }
            }),
        });
        &AS_MAPPING
    }
}

impl Representable for PyNdArray {
    fn repr_str(zelf: &vm::Py<Self>, _vm: &VirtualMachine) -> PyResult<String> {
        let data = zelf.data.read().unwrap();
        Ok(format_array(&data, true))
    }
}

impl AsSequence for PyNdArray {
    fn as_sequence() -> &'static PySequenceMethods {
        use once_cell::sync::Lazy;
        static AS_SEQUENCE: Lazy<PySequenceMethods> = Lazy::new(|| PySequenceMethods {
            length: atomic_func!(|seq, _vm| {
                let zelf = PyNdArray::sequence_downcast(seq);
                let data = zelf.data.read().unwrap();
                Ok(if data.ndim() == 0 { 0 } else { data.shape()[0] })
            }),
            ..PySequenceMethods::NOT_IMPLEMENTED
        });
        &AS_SEQUENCE
    }
}

impl Iterable for PyNdArray {
    fn iter(zelf: PyRef<Self>, vm: &VirtualMachine) -> PyResult {
        let data = zelf.data.read().unwrap();
        let length = if data.ndim() == 0 {
            return Err(vm.new_type_error("iteration over a 0-d array".to_owned()));
        } else {
            data.shape()[0]
        };
        drop(data);
        Ok(PyNdArrayIter {
            array: zelf,
            index: std::sync::atomic::AtomicUsize::new(0),
            length,
        }
        .into_pyobject(vm))
    }
}

/// Extract integer indices from an NdArray, resolving negative indices.
fn extract_int_indices(
    idx_arr: &NdArray,
    dim_size: usize,
    vm: &VirtualMachine,
) -> PyResult<Vec<usize>> {
    let flat = idx_arr.flatten();
    let mut out = Vec::with_capacity(flat.size());
    for i in 0..flat.size() {
        let s = flat.get(&[i]).map_err(|e| numpy_err(e, vm))?;
        let v: i64 = match s {
            Scalar::Int32(v) => v as i64,
            Scalar::Int64(v) => v,
            Scalar::Float32(v) => v as i64,
            Scalar::Float64(v) => v as i64,
            Scalar::Bool(v) => v as i64,
            _ => return Err(vm.new_type_error("expected integer index".to_owned())),
        };
        let resolved = if v < 0 {
            (dim_size as i64 + v) as usize
        } else {
            v as usize
        };
        if resolved >= dim_size {
            return Err(vm.new_index_error(format!(
                "index {v} is out of bounds for axis with size {dim_size}"
            )));
        }
        out.push(resolved);
    }
    Ok(out)
}

/// Multi-dimensional fancy indexing: a[arr1, arr2] or a[arr1, 3] or a[:, arr1].
/// Each item in the tuple can be an ndarray (integer indices), a scalar int, or a slice.
/// For array indices, they are broadcast together and used for point indexing.
fn multi_dim_fancy_getitem(
    data: &NdArray,
    items: &[PyObjectRef],
    vm: &VirtualMachine,
) -> PyResult<PyObjectRef> {
    let shape = data.shape();
    let ndim = data.ndim();

    // Classify each index item
    enum IdxKind {
        Array(Vec<usize>), // resolved indices for this axis
        Scalar(usize),     // single index (reduces axis)
        Slice(SliceArg),   // slice on this axis
    }

    let mut kinds = Vec::with_capacity(items.len());
    let mut array_len: Option<usize> = None;
    let mut array_shape: Option<Vec<usize>> = None; // original shape of index arrays
    let mut data_axis: usize = 0; // track which data axis we're consuming

    for item in items.iter() {
        let dim_size = if data_axis < ndim {
            shape[data_axis]
        } else {
            1
        };

        if let Some(arr) = item.downcast_ref::<PyNdArray>() {
            let arr_data = arr.data.read().unwrap();
            if arr_data.dtype() == DType::Bool {
                let mask_shape = arr_data.shape().to_vec();
                if mask_shape.len() > 1 {
                    // Multi-dim boolean mask: expand to nonzero indices (one per mask dim)
                    let mask_ndim = mask_shape.len();
                    let mut coords: Vec<Vec<usize>> = (0..mask_ndim).map(|_| Vec::new()).collect();
                    // Flatten the mask first so we can iterate with a single index
                    let flat_mask = arr_data.flatten();
                    for flat_idx in 0..flat_mask.size() {
                        let s = flat_mask.get(&[flat_idx]).map_err(|e| numpy_err(e, vm))?;
                        if let Scalar::Bool(true) = s {
                            let mut remaining = flat_idx;
                            for d in (0..mask_ndim).rev() {
                                coords[d].push(remaining % mask_shape[d]);
                                remaining /= mask_shape[d];
                            }
                        }
                    }
                    let num_true = coords[0].len();
                    if let Some(al) = array_len {
                        if al != num_true {
                            return Err(vm.new_value_error(
                                "shape mismatch: indexing arrays could not be broadcast together"
                                    .to_owned(),
                            ));
                        }
                    }
                    array_len = Some(num_true);
                    for coord_set in coords {
                        kinds.push(IdxKind::Array(coord_set));
                    }
                    data_axis += mask_ndim;
                } else {
                    // 1-D boolean mask on this axis → convert to integer indices
                    let mut indices = Vec::new();
                    for i in 0..arr_data.size() {
                        let s = arr_data.get(&[i]).map_err(|e| numpy_err(e, vm))?;
                        if let Scalar::Bool(true) = s {
                            indices.push(i);
                        }
                    }
                    if let Some(al) = array_len {
                        if al != indices.len() {
                            return Err(vm.new_value_error(
                                "shape mismatch: indexing arrays could not be broadcast together"
                                    .to_owned(),
                            ));
                        }
                    }
                    array_len = Some(indices.len());
                    kinds.push(IdxKind::Array(indices));
                    data_axis += 1;
                }
            } else {
                // Capture original shape for reshaping result
                if array_shape.is_none() {
                    array_shape = Some(arr_data.shape().to_vec());
                }
                let indices = extract_int_indices(&arr_data, dim_size, vm)?;
                if let Some(al) = array_len {
                    if al != indices.len() {
                        return Err(vm.new_value_error(
                            "shape mismatch: indexing arrays could not be broadcast together"
                                .to_owned(),
                        ));
                    }
                }
                array_len = Some(indices.len());
                kinds.push(IdxKind::Array(indices));
                data_axis += 1;
            }
        } else if let Some(slice) = item.downcast_ref::<PySlice>() {
            let arg = py_slice_to_slice_arg(slice, vm)?;
            kinds.push(IdxKind::Slice(arg));
            data_axis += 1;
        } else if let Ok(i) = item.clone().try_into_value::<i64>(vm) {
            let resolved = if i < 0 {
                (dim_size as i64 + i) as usize
            } else {
                i as usize
            };
            kinds.push(IdxKind::Scalar(resolved));
            data_axis += 1;
        } else {
            return Err(vm.new_type_error("unsupported index type in tuple".to_owned()));
        }
    }

    let n = array_len.unwrap_or(0);
    if n == 0 && !kinds.iter().any(|k| matches!(k, IdxKind::Array(_))) {
        // No array indices — fallback to slice/integer logic
        let args: Vec<SliceArg> = kinds
            .into_iter()
            .map(|k| match k {
                IdxKind::Scalar(i) => SliceArg::Index(i as isize),
                IdxKind::Slice(s) => s,
                IdxKind::Array(_) => unreachable!(),
            })
            .collect();
        let result = data.slice(&args).map_err(|e| numpy_err(e, vm))?;
        return Ok(ndarray_or_scalar(result, vm));
    } else if n == 0 {
        // Empty array index: return empty array with appropriate shape
        let mut out_shape = Vec::new();
        // Number of consumed axes = number of Array/Scalar kinds
        let _consumed_axes = kinds
            .iter()
            .filter(|k| !matches!(k, IdxKind::Slice(_)))
            .count();
        // Remaining axes from slices
        for k in &kinds {
            if let IdxKind::Slice(_) = k {
                // These axes are preserved
                out_shape.push(0usize); // placeholder
            }
        }
        // Just return an empty array with 0 elements
        let empty = NdArray::zeros(&[0], data.dtype());
        return Ok(PyNdArray::from_core(empty).to_py(vm));
    }

    // Check if we have only array + scalar indices (no slices) — "point indexing"
    let has_slice = kinds.iter().any(|k| matches!(k, IdxKind::Slice(_)));

    if !has_slice {
        // Pure point indexing: result shape = (n,) + remaining dims
        let num_indexed_dims = kinds.len();

        if num_indexed_dims >= ndim {
            // Full coordinate → collect scalars into a flat 1-d array
            let mut values = Vec::with_capacity(n);
            for j in 0..n {
                let mut coord = Vec::with_capacity(ndim);
                for kind in &kinds {
                    match kind {
                        IdxKind::Array(indices) => coord.push(indices[j]),
                        IdxKind::Scalar(i) => coord.push(*i),
                        _ => unreachable!(),
                    }
                }
                let s = data.get(&coord).map_err(|e| numpy_err(e, vm))?;
                values.push(match s {
                    Scalar::Float64(v) => v,
                    Scalar::Float32(v) => v as f64,
                    Scalar::Int32(v) => v as f64,
                    Scalar::Int64(v) => v as f64,
                    Scalar::Bool(v) => {
                        if v {
                            1.0
                        } else {
                            0.0
                        }
                    }
                    _ => 0.0,
                });
            }
            let result = NdArray::from_vec(values);
            let result = result.astype(data.dtype());
            // Reshape to match the original index array shape if multi-dimensional
            if let Some(ref ashape) = array_shape {
                if ashape.len() > 1 {
                    let result = result.reshape(ashape).map_err(|e| numpy_err(e, vm))?;
                    return Ok(PyNdArray::from_core(result).to_py(vm));
                }
            }
            return Ok(PyNdArray::from_core(result).to_py(vm));
        } else {
            // Partial coordinate → collect sub-arrays and stack
            let mut sub_results = Vec::with_capacity(n);
            for j in 0..n {
                let mut slice_args: Vec<SliceArg> = Vec::with_capacity(ndim);
                for kind in &kinds {
                    match kind {
                        IdxKind::Array(indices) => {
                            slice_args.push(SliceArg::Index(indices[j] as isize));
                        }
                        IdxKind::Scalar(i) => {
                            slice_args.push(SliceArg::Index(*i as isize));
                        }
                        _ => unreachable!(),
                    }
                }
                let sub = data.slice(&slice_args).map_err(|e| numpy_err(e, vm))?;
                sub_results.push(sub);
            }
            if sub_results.is_empty() {
                return Ok(PyNdArray::from_core(NdArray::zeros(&[0], data.dtype())).to_py(vm));
            }
            let refs: Vec<&NdArray> = sub_results.iter().collect();
            let result = numpy_rust_core::concatenate(&refs, 0).map_err(|e| numpy_err(e, vm))?;
            let sub_shape = sub_results[0].shape();
            if sub_shape.len() > 1 || (sub_shape.len() == 1 && sub_shape[0] > 1) {
                let mut new_shape = vec![n];
                new_shape.extend_from_slice(sub_shape);
                let result = result.reshape(&new_shape).map_err(|e| numpy_err(e, vm))?;
                return Ok(PyNdArray::from_core(result).to_py(vm));
            }
            return Ok(PyNdArray::from_core(result).to_py(vm));
        }
    }

    // Mixed slice + array indexing: handle by iterating array indices
    // and slicing for each, then stacking
    // Count leading non-array dims to determine where n dimension goes
    let first_array_pos = kinds
        .iter()
        .position(|k| matches!(k, IdxKind::Array(_)))
        .unwrap_or(0);
    let leading_slice_dims = kinds[..first_array_pos]
        .iter()
        .filter(|k| matches!(k, IdxKind::Slice(_)))
        .count();

    let mut results = Vec::with_capacity(n);
    for j in 0..n {
        let mut slice_args = Vec::with_capacity(items.len());
        for kind in &kinds {
            match kind {
                IdxKind::Array(indices) => {
                    slice_args.push(SliceArg::Index(indices[j] as isize));
                }
                IdxKind::Scalar(i) => {
                    slice_args.push(SliceArg::Index(*i as isize));
                }
                IdxKind::Slice(s) => {
                    slice_args.push(s.clone());
                }
            }
        }
        let sub = data.slice(&slice_args).map_err(|e| numpy_err(e, vm))?;
        results.push(sub);
    }
    if results.is_empty() {
        return Ok(PyNdArray::from_core(NdArray::zeros(&[0], data.dtype())).to_py(vm));
    }
    let refs: Vec<&NdArray> = results.iter().collect();
    let result = numpy_rust_core::concatenate(&refs, 0).map_err(|e| numpy_err(e, vm))?;
    // Reshape to (n, *sub_shape), then move n to correct position
    let sub_shape = results[0].shape();
    if sub_shape.len() > 1 || (sub_shape.len() == 1 && sub_shape[0] > 1) {
        let mut new_shape = vec![n];
        new_shape.extend_from_slice(sub_shape);
        let mut result = result.reshape(&new_shape).map_err(|e| numpy_err(e, vm))?;
        // Move n dimension from position 0 to position leading_slice_dims
        if leading_slice_dims > 0 {
            for i in 0..leading_slice_dims {
                result = result.swapaxes(i, i + 1).map_err(|e| numpy_err(e, vm))?;
            }
        }
        return Ok(PyNdArray::from_core(result).to_py(vm));
    }
    Ok(PyNdArray::from_core(result).to_py(vm))
}

/// Multi-dimensional fancy setitem: a[arr1, arr2] = values
fn multi_dim_fancy_setitem(
    zelf: &PyNdArray,
    items: &[PyObjectRef],
    value: &PyObjectRef,
    vm: &VirtualMachine,
) -> PyResult<()> {
    let data = zelf.data.read().unwrap();
    let shape = data.shape().to_vec();
    let ndim = data.ndim();
    let dtype = data.dtype();
    drop(data);

    // Parse each index item
    let mut array_indices: Vec<Option<Vec<usize>>> = Vec::new();
    let mut scalar_indices: Vec<Option<usize>> = Vec::new();
    let mut array_len: Option<usize> = None;

    for (axis, item) in items.iter().enumerate() {
        let dim_size = if axis < ndim { shape[axis] } else { 1 };
        if let Some(arr) = item.downcast_ref::<PyNdArray>() {
            let arr_data = arr.data.read().unwrap();
            let indices = extract_int_indices(&arr_data, dim_size, vm)?;
            if let Some(al) = array_len {
                if al != indices.len() {
                    return Err(vm.new_value_error(
                        "shape mismatch: indexing arrays could not be broadcast together"
                            .to_owned(),
                    ));
                }
            }
            array_len = Some(indices.len());
            array_indices.push(Some(indices));
            scalar_indices.push(None);
        } else if let Ok(i) = item.clone().try_into_value::<i64>(vm) {
            let resolved = if i < 0 {
                (dim_size as i64 + i) as usize
            } else {
                i as usize
            };
            array_indices.push(None);
            scalar_indices.push(Some(resolved));
        } else {
            return Err(vm.new_type_error("unsupported index type in fancy setitem".to_owned()));
        }
    }

    let n = array_len.unwrap_or(1);
    let value_arr = obj_to_ndarray(value, vm)?;
    let value_arr = if value_arr.dtype() != dtype {
        value_arr.astype(dtype)
    } else {
        value_arr
    };

    let mut write_data = zelf.data.write().unwrap();
    for j in 0..n {
        let mut coord = Vec::with_capacity(ndim);
        for k in 0..items.len() {
            if let Some(ref indices) = array_indices[k] {
                coord.push(indices[j]);
            } else if let Some(idx) = scalar_indices[k] {
                coord.push(idx);
            }
        }
        // Get value for this position
        let val = if value_arr.size() == 1 {
            value_arr
                .get(&vec![0; value_arr.ndim()])
                .map_err(|e| numpy_err(e, vm))?
        } else if value_arr.ndim() == 1 {
            let idx = j % value_arr.size();
            value_arr.get(&[idx]).map_err(|e| numpy_err(e, vm))?
        } else {
            let idx = j % value_arr.size();
            let flat_coord = linear_to_coord(idx, value_arr.shape());
            value_arr.get(&flat_coord).map_err(|e| numpy_err(e, vm))?
        };
        write_data.set(&coord, val).map_err(|e| numpy_err(e, vm))?;
    }
    Ok(())
}

/// Extract an integer value from a Python object for use as a slice bound.
/// Handles PyInt and 0-d ndarray (via nb_index).
fn slice_bound_to_i64(obj: &vm::PyObjectRef, vm: &VirtualMachine) -> PyResult<i64> {
    // Try direct int first (most common)
    if let Ok(v) = obj.clone().try_into_value::<i64>(vm) {
        return Ok(v);
    }
    // Try nb_index (handles 0-d ndarray with numeric dtype)
    if let Some(arr) = obj.downcast_ref::<PyNdArray>() {
        let data = arr.data.read().unwrap();
        if data.size() == 1 {
            let s = data
                .get(&vec![0usize; data.ndim()])
                .map_err(|e| numpy_err(e, vm))?;
            let v: i64 = match s {
                Scalar::Bool(b) => {
                    if b {
                        1
                    } else {
                        0
                    }
                }
                Scalar::Int32(i) => i as i64,
                Scalar::Int64(i) => i,
                Scalar::Float32(f) => f as i64,
                Scalar::Float64(f) => f as i64,
                Scalar::Complex64(c) => c.re as i64,
                Scalar::Complex128(c) => c.re as i64,
                Scalar::Str(_) => {
                    return Err(vm.new_type_error("string cannot be used as slice index".to_owned()))
                }
            };
            return Ok(v);
        }
    }
    Err(vm.new_type_error(format!(
        "'{}' object cannot be interpreted as an integer",
        obj.class().name()
    )))
}

/// Convert a RustPython `PySlice` to a core `SliceArg`.
pub(crate) fn py_slice_to_slice_arg(slice: &PySlice, vm: &VirtualMachine) -> PyResult<SliceArg> {
    let start = match &slice.start {
        Some(obj) if !vm.is_none(obj) => {
            let v = slice_bound_to_i64(obj, vm)?;
            Some(v as isize)
        }
        _ => None,
    };
    let stop = if vm.is_none(&slice.stop) {
        None
    } else {
        let v = slice_bound_to_i64(&slice.stop, vm)?;
        Some(v as isize)
    };
    let step = match &slice.step {
        Some(obj) if !vm.is_none(obj) => {
            let v = slice_bound_to_i64(obj, vm)?;
            v as isize
        }
        _ => 1,
    };
    Ok(SliceArg::Range { start, stop, step })
}

/// Convert a Python object to a `SliceArg` (either a slice or an integer index).
fn py_obj_to_slice_arg(obj: &PyObjectRef, vm: &VirtualMachine) -> PyResult<SliceArg> {
    if let Some(slice) = obj.downcast_ref::<PySlice>() {
        return py_slice_to_slice_arg(slice, vm);
    }
    if let Ok(i) = obj.clone().try_into_value::<i64>(vm) {
        return Ok(SliceArg::Index(i as isize));
    }
    Err(vm.new_type_error("index must be an integer or slice".to_owned()))
}

/// Extract axis and keepdims from FuncArgs (supports both positional and keyword).
/// `keepdims_pos` is the positional index for keepdims (typically 1 for sum/min/max, 2 for std/var).
fn extract_axis_keepdims(
    args: &vm::function::FuncArgs,
    vm: &VirtualMachine,
) -> (vm::function::OptionalArg<PyObjectRef>, bool) {
    extract_axis_keepdims_at(args, vm, 1)
}

fn extract_axis_keepdims_at(
    args: &vm::function::FuncArgs,
    vm: &VirtualMachine,
    keepdims_pos: usize,
) -> (vm::function::OptionalArg<PyObjectRef>, bool) {
    let axis_obj = args
        .args
        .first()
        .cloned()
        .or_else(|| args.kwargs.get("axis").cloned());
    let kd = args
        .kwargs
        .get("keepdims")
        .cloned()
        .or_else(|| args.args.get(keepdims_pos).cloned())
        .and_then(|v| v.try_into_value::<bool>(vm).ok())
        .unwrap_or(false);
    let axis_arg = if let Some(obj) = axis_obj {
        vm::function::OptionalArg::Present(obj)
    } else {
        vm::function::OptionalArg::Missing
    };
    (axis_arg, kd)
}

/// Parse an optional axis argument (None means reduce all).
pub fn parse_optional_axis(
    arg: vm::function::OptionalArg<PyObjectRef>,
    vm: &VirtualMachine,
) -> PyResult<Option<usize>> {
    match arg.into_option() {
        None => Ok(None),
        Some(obj) => {
            if vm.is_none(&obj) {
                Ok(None)
            } else {
                let axis: i64 = obj.try_into_value(vm)?;
                Ok(Some(axis as usize))
            }
        }
    }
}

/// Parsed axis argument that can be None, a single axis, or multiple axes (tuple).
pub enum AxisArg {
    /// No axis specified – reduce over all elements.
    None,
    /// A single axis.
    Single(usize),
    /// Multiple axes (from a tuple).
    Multi(Vec<usize>),
}

/// Parse an optional axis argument that may be None, int, or tuple-of-ints.
pub fn parse_axis_arg(
    arg: vm::function::OptionalArg<PyObjectRef>,
    ndim: usize,
    vm: &VirtualMachine,
) -> PyResult<AxisArg> {
    match arg.into_option() {
        Option::None => Ok(AxisArg::None),
        Some(obj) => {
            if vm.is_none(&obj) {
                return Ok(AxisArg::None);
            }
            // Try as tuple first
            if let Some(tuple) = obj.downcast_ref::<PyTuple>() {
                let mut axes = Vec::new();
                for item in tuple.as_slice() {
                    let i: i64 = item.clone().try_into_value(vm)?;
                    let ax = if i < 0 {
                        (ndim as i64 + i) as usize
                    } else {
                        i as usize
                    };
                    axes.push(ax);
                }
                return Ok(AxisArg::Multi(axes));
            }
            // Try as int
            let i: i64 = obj.try_into_value(vm)?;
            let ax = if i < 0 {
                (ndim as i64 + i) as usize
            } else {
                i as usize
            };
            Ok(AxisArg::Single(ax))
        }
    }
}

/// Reduce along multiple axes sequentially (highest axis first to keep indices valid).
fn reduce_multi_axis(
    arr: &NdArray,
    axes: &[usize],
    reduce_fn: impl Fn(&NdArray, Option<usize>) -> numpy_rust_core::error::Result<NdArray>,
) -> numpy_rust_core::error::Result<NdArray> {
    let mut sorted_axes: Vec<usize> = axes.to_vec();
    sorted_axes.sort();
    sorted_axes.dedup();
    // Reduce from highest axis first so lower indices stay valid
    let mut result = arr.clone();
    for &ax in sorted_axes.iter().rev() {
        result = reduce_fn(&result, Some(ax))?;
    }
    Ok(result)
}

/// Compute variance over multiple axes correctly using the formula:
///   var = mean(x^2, axes) - mean(x, axes)^2
/// Then adjust for ddof: var * N / (N - ddof)
fn var_multi_axis(
    arr: &NdArray,
    axes: &[usize],
    ddof: usize,
) -> numpy_rust_core::error::Result<NdArray> {
    // Compute total number of elements in the reduction axes
    let shape = arr.shape();
    let mut n: usize = 1;
    for &ax in axes {
        n *= shape[ax];
    }

    // mean(x, axes)
    let mean_x = reduce_multi_axis(arr, axes, |a, ax| a.mean(ax, false))?;
    // mean(x^2, axes)
    let x_sq = (arr * arr)?;
    let mean_x_sq = reduce_multi_axis(&x_sq, axes, |a, ax| a.mean(ax, false))?;
    // var = mean(x^2) - mean(x)^2
    let mean_x_sq_val = (&mean_x * &mean_x)?;
    let var = (&mean_x_sq - &mean_x_sq_val)?;

    if ddof > 0 && n > ddof {
        // Adjust: var * N / (N - ddof)
        let adj = n as f64 / (n - ddof) as f64;
        let adj_arr = NdArray::from_scalar(adj);
        let result = (&var * &adj_arr)?;
        Ok(result)
    } else {
        Ok(var)
    }
}

/// Compute std over multiple axes = sqrt(var_multi_axis).
fn std_multi_axis(
    arr: &NdArray,
    axes: &[usize],
    ddof: usize,
) -> numpy_rust_core::error::Result<NdArray> {
    let v = var_multi_axis(arr, axes, ddof)?;
    Ok(v.sqrt())
}

// --- __setitem__ implementation ---

fn setitem_impl(
    zelf: &PyNdArray,
    key: PyObjectRef,
    value: PyObjectRef,
    vm: &VirtualMachine,
) -> PyResult<()> {
    let base_obj = zelf.base.read().unwrap().clone();
    let view_prefix = zelf.view_prefix.read().unwrap().clone();
    if let (Some(base_obj), Some(prefix)) = (base_obj, view_prefix) {
        if let Some(base_array) = base_obj.downcast_ref::<PyNdArray>() {
            let composed = compose_view_key(&prefix, &key, vm)?;
            return setitem_impl(base_array, composed, value, vm);
        }
    }

    // Integer key → set single element or row
    if let Ok(idx) = key.clone().try_into_value::<i64>(vm) {
        let data = zelf.data.read().unwrap();
        let shape = data.shape().to_vec();
        let ndim = data.ndim();
        drop(data);

        let resolved = if idx < 0 {
            (shape[0] as i64 + idx) as usize
        } else {
            idx as usize
        };

        if ndim == 1 {
            // Scalar assignment: a[i] = value
            let scalar = py_obj_to_scalar(&value, vm)?;
            zelf.data
                .write()
                .unwrap()
                .set(&[resolved], scalar)
                .map_err(|e| numpy_err(e, vm))?;
        } else {
            // Row assignment: a[i] = array_value
            let value_arr = obj_to_ndarray(&value, vm)?;
            zelf.data
                .write()
                .unwrap()
                .set_slice(&[SliceArg::Index(idx as isize)], &value_arr)
                .map_err(|e| numpy_err(e, vm))?;
        }
        return Ok(());
    }

    // Slice key → a[start:stop:step] = values
    if let Some(slice) = key.downcast_ref::<PySlice>() {
        let arg = py_slice_to_slice_arg(slice, vm)?;
        let value_arr = obj_to_ndarray(&value, vm)?;
        zelf.data
            .write()
            .unwrap()
            .set_slice(&[arg], &value_arr)
            .map_err(|e| numpy_err(e, vm))?;
        return Ok(());
    }

    // Tuple key → multi-dimensional assignment
    if let Some(tuple) = key.downcast_ref::<PyTuple>() {
        let items = tuple.as_slice();

        // Check for ndarray indices (multi-dim fancy setitem)
        let has_ndarray = items
            .iter()
            .any(|item| item.downcast_ref::<PyNdArray>().is_some());
        if has_ndarray {
            return multi_dim_fancy_setitem(zelf, items, &value, vm);
        }

        let has_slice = items
            .iter()
            .any(|item| item.downcast_ref::<PySlice>().is_some());

        if has_slice {
            let args: Vec<SliceArg> = items
                .iter()
                .map(|item| py_obj_to_slice_arg(item, vm))
                .collect::<PyResult<Vec<_>>>()?;
            let value_arr = obj_to_ndarray(&value, vm)?;
            zelf.data
                .write()
                .unwrap()
                .set_slice(&args, &value_arr)
                .map_err(|e| numpy_err(e, vm))?;
            return Ok(());
        }

        // All integers → scalar assignment at multi-dim index
        let mut indices = Vec::new();
        for item in items {
            let i: i64 = item.clone().try_into_value(vm)?;
            indices.push(i as isize);
        }
        let data = zelf.data.read().unwrap();
        let shape = data.shape().to_vec();
        drop(data);

        let usize_indices: Vec<usize> = indices
            .iter()
            .enumerate()
            .map(|(axis, &i)| {
                if i < 0 {
                    (shape[axis] as isize + i) as usize
                } else {
                    i as usize
                }
            })
            .collect();
        let scalar = py_obj_to_scalar(&value, vm)?;
        zelf.data
            .write()
            .unwrap()
            .set(&usize_indices, scalar)
            .map_err(|e| numpy_err(e, vm))?;
        return Ok(());
    }

    // NdArray key: boolean mask or integer fancy indexing
    if let Some(arr) = key.downcast_ref::<PyNdArray>() {
        let arr_data = arr.data.read().unwrap();
        if arr_data.dtype() == DType::Bool {
            let value_arr = obj_to_ndarray(&value, vm)?;
            // Cast value to target dtype if needed
            let target_dt = zelf.data.read().unwrap().dtype();
            let value_arr = if value_arr.dtype() != target_dt {
                value_arr.astype(target_dt)
            } else {
                value_arr
            };
            let mask_owned = arr_data.clone();
            drop(arr_data);
            zelf.data
                .write()
                .unwrap()
                .mask_set(&mask_owned, &value_arr)
                .map_err(|e| numpy_err(e, vm))?;
            return Ok(());
        } else if arr_data.dtype().is_integer() || arr_data.dtype().is_float() {
            let dim_size = zelf.data.read().unwrap().shape()[0];
            let indices = extract_int_indices(&arr_data, dim_size, vm)?;
            drop(arr_data);
            let value_arr = obj_to_ndarray(&value, vm)?;
            zelf.data
                .write()
                .unwrap()
                .index_set(0, &indices, &value_arr)
                .map_err(|e| numpy_err(e, vm))?;
            return Ok(());
        }
    }

    // List of integers -> fancy indexing assignment: a[[0, 2]] = values
    if let Some(list) = key.downcast_ref::<PyList>() {
        let items = list.borrow_vec();
        let dim_size = zelf.data.read().unwrap().shape()[0];
        let mut indices = Vec::with_capacity(items.len());
        for item in items.iter() {
            let i: i64 = item.clone().try_into_value(vm)?;
            let resolved = if i < 0 {
                (dim_size as i64 + i) as usize
            } else {
                i as usize
            };
            indices.push(resolved);
        }
        let value_arr = obj_to_ndarray(&value, vm)?;
        zelf.data
            .write()
            .unwrap()
            .index_set(0, &indices, &value_arr)
            .map_err(|e| numpy_err(e, vm))?;
        return Ok(());
    }

    // Ellipsis key → a[...] = value (set all elements)
    if key.is(&vm.ctx.ellipsis) {
        let value_arr = obj_to_ndarray(&value, vm)?;
        // Use set_slice with full slice for each dimension
        let ndim = zelf.data.read().unwrap().ndim();
        let args: Vec<SliceArg> = (0..ndim).map(|_| SliceArg::Full).collect();
        zelf.data
            .write()
            .unwrap()
            .set_slice(&args, &value_arr)
            .map_err(|e| numpy_err(e, vm))?;
        return Ok(());
    }

    Err(vm.new_type_error("unsupported index type for assignment".to_owned()))
}

// --- tolist helper ---

fn ndarray_to_pylist(data: &NdArray, vm: &VirtualMachine) -> PyObjectRef {
    let ndim = data.ndim();
    let shape = data.shape();

    if ndim == 0 {
        // 0-D: return plain scalar
        let s = data.get(&[]).unwrap();
        return scalar_to_py(s, vm);
    }

    if ndim == 1 {
        // 1-D: build flat list
        let items: Vec<PyObjectRef> = (0..shape[0])
            .map(|i| {
                let s = data.get(&[i]).unwrap();
                scalar_to_py(s, vm)
            })
            .collect();
        return PyList::from(items).into_ref(&vm.ctx).into();
    }

    // N-D: recursively slice along first axis
    let items: Vec<PyObjectRef> = (0..shape[0])
        .map(|i| {
            let sub = data.slice(&[SliceArg::Index(i as isize)]).unwrap();
            ndarray_to_pylist(&sub, vm)
        })
        .collect();
    PyList::from(items).into_ref(&vm.ctx).into()
}

// --- repr/str formatting ---

/// Format a scalar value for repr output.
fn format_scalar(s: &Scalar) -> String {
    match s {
        Scalar::Bool(v) => {
            if *v {
                " True".to_owned()
            } else {
                "False".to_owned()
            }
        }
        Scalar::Int32(v) => v.to_string(),
        Scalar::Int64(v) => v.to_string(),
        Scalar::Float32(v) => format_float(*v as f64),
        Scalar::Float64(v) => format_float(*v),
        Scalar::Complex64(v) => {
            if v.im >= 0.0 {
                format!("({}+{}j)", v.re, v.im)
            } else {
                format!("({}{}j)", v.re, v.im)
            }
        }
        Scalar::Complex128(v) => {
            if v.im >= 0.0 {
                format!("({}+{}j)", v.re, v.im)
            } else {
                format!("({}{}j)", v.re, v.im)
            }
        }
        Scalar::Str(v) => format!("'{v}'"),
    }
}

/// Format a float like NumPy: use "." suffix for integer-valued floats.
fn format_float(v: f64) -> String {
    if v.is_nan() {
        "nan".to_owned()
    } else if v.is_infinite() {
        if v > 0.0 {
            "inf".to_owned()
        } else {
            "-inf".to_owned()
        }
    } else if v == v.trunc() && v.abs() < 1e16 {
        // Integer-valued float: show as "5." not "5.0"
        format!("{v:.1}")
    } else {
        // Use default float formatting, trim trailing zeros
        let s = format!("{v}");
        s
    }
}

/// Max elements to show per axis edge before truncating.
const EDGE_ITEMS: usize = 3;
/// Threshold for total elements to trigger truncation.
const THRESHOLD: usize = 1000;

fn format_array(data: &NdArray, with_array_prefix: bool) -> String {
    let ndim = data.ndim();

    if ndim == 0 {
        let s = data.get(&[]).unwrap();
        let val = format_scalar(&s);
        return if with_array_prefix {
            format!("array({val})")
        } else {
            val
        };
    }

    let body = format_array_inner(data, 0);

    if with_array_prefix {
        format!("array({body})")
    } else {
        body
    }
}

fn format_array_inner(data: &NdArray, depth: usize) -> String {
    let ndim = data.ndim();
    let shape = data.shape();

    if ndim == 1 {
        // 1-D array
        let n = shape[0];
        let truncate = data.size() > THRESHOLD && n > EDGE_ITEMS * 2;

        let mut parts = Vec::new();
        if truncate {
            for i in 0..EDGE_ITEMS {
                let s = data.get(&[i]).unwrap();
                parts.push(format_scalar(&s));
            }
            parts.push("...".to_owned());
            for i in (n - EDGE_ITEMS)..n {
                let s = data.get(&[i]).unwrap();
                parts.push(format_scalar(&s));
            }
        } else {
            for i in 0..n {
                let s = data.get(&[i]).unwrap();
                parts.push(format_scalar(&s));
            }
        }
        return format!("[{}]", parts.join(", "));
    }

    // N-D: recurse along first axis
    let n = shape[0];
    let total_size = data.size();
    let truncate = total_size > THRESHOLD && n > EDGE_ITEMS * 2;

    let indent = " ".repeat(depth + 1);
    let mut parts = Vec::new();

    let indices: Vec<usize> = if truncate {
        let mut v: Vec<usize> = (0..EDGE_ITEMS).collect();
        v.push(usize::MAX); // sentinel for "..."
        v.extend((n - EDGE_ITEMS)..n);
        v
    } else {
        (0..n).collect()
    };

    for &i in &indices {
        if i == usize::MAX {
            parts.push(format!("{indent}..."));
            continue;
        }
        let sub = data.slice(&[SliceArg::Index(i as isize)]).unwrap();
        let sub_str = format_array_inner(&sub, depth + 1);
        parts.push(format!("{indent}{sub_str}"));
    }

    let sep = if ndim == 2 { ",\n" } else { ",\n\n" };
    format!("[{}]", parts.join(sep))
}

// ---------------------------------------------------------------------------
// flagsobj – dict-like + attribute-access flags object
// ---------------------------------------------------------------------------
use std::collections::HashMap;

#[vm::pyclass(module = "numpy", name = "flagsobj")]
#[derive(Debug, PyPayload)]
pub struct PyFlagsObj {
    map: HashMap<String, bool>,
}

#[vm::pyclass(with(AsMapping, Representable))]
impl PyFlagsObj {
    pub fn new(map: HashMap<String, bool>) -> Self {
        Self { map }
    }

    fn get_flag(&self, key: &str) -> bool {
        self.map.get(key).copied().unwrap_or(false)
    }

    #[pymethod]
    fn __getitem__(&self, key: PyObjectRef, vm: &VirtualMachine) -> PyResult<bool> {
        let k: String = key.try_into_value(vm)?;
        self.map
            .get(&k)
            .copied()
            .ok_or_else(|| vm.new_key_error(vm.ctx.new_str(k).into()))
    }

    #[pyslot]
    fn slot_richcompare(
        zelf: &vm::PyObject,
        other: &vm::PyObject,
        op: vm::types::PyComparisonOp,
        vm: &VirtualMachine,
    ) -> PyResult<vm::function::Either<PyObjectRef, vm::function::PyComparisonValue>> {
        let zelf = zelf
            .downcast_ref::<PyFlagsObj>()
            .ok_or_else(|| vm.new_type_error("expected flagsobj".to_owned()))?;
        if let Some(other_flags) = other.downcast_ref::<PyFlagsObj>() {
            let keys = [
                "C_CONTIGUOUS",
                "F_CONTIGUOUS",
                "OWNDATA",
                "WRITEABLE",
                "ALIGNED",
                "WRITEBACKIFCOPY",
            ];
            let equal = keys
                .iter()
                .all(|k| zelf.map.get(*k) == other_flags.map.get(*k));
            let result = match op {
                vm::types::PyComparisonOp::Eq => equal,
                vm::types::PyComparisonOp::Ne => !equal,
                _ => {
                    return Ok(vm::function::Either::B(
                        vm::function::PyComparisonValue::NotImplemented,
                    ))
                }
            };
            Ok(vm::function::Either::A(vm.ctx.new_bool(result).into()))
        } else {
            Ok(vm::function::Either::B(
                vm::function::PyComparisonValue::NotImplemented,
            ))
        }
    }

    // Lowercase attribute-style accessors (numpy compat)
    #[pygetset]
    fn c_contiguous(&self) -> bool {
        self.get_flag("C_CONTIGUOUS")
    }
    #[pygetset]
    fn f_contiguous(&self) -> bool {
        self.get_flag("F_CONTIGUOUS")
    }
    #[pygetset]
    fn owndata(&self) -> bool {
        self.get_flag("OWNDATA")
    }
    #[pygetset]
    fn writeable(&self) -> bool {
        self.get_flag("WRITEABLE")
    }
    #[pygetset(setter)]
    fn set_writeable(&self, _val: bool, _vm: &VirtualMachine) {
        // Setting writeable is a no-op (all arrays are mutable in our implementation)
    }
    #[pygetset]
    fn aligned(&self) -> bool {
        self.get_flag("ALIGNED")
    }
    #[pygetset]
    fn writebackifcopy(&self) -> bool {
        self.get_flag("WRITEBACKIFCOPY")
    }
    #[pygetset]
    fn fnc(&self) -> bool {
        self.get_flag("FNC")
    }
    #[pygetset]
    fn forc(&self) -> bool {
        self.get_flag("FORC")
    }
    #[pygetset]
    fn contiguous(&self) -> bool {
        self.get_flag("CONTIGUOUS")
    }
    #[pygetset]
    fn fortran(&self) -> bool {
        self.get_flag("FORTRAN")
    }
    #[pygetset]
    fn updateifcopy(&self) -> bool {
        self.get_flag("UPDATEIFCOPY")
    }
    #[pygetset]
    fn behaved(&self) -> bool {
        self.get_flag("BEHAVED")
    }
    #[pygetset]
    fn carray(&self) -> bool {
        self.get_flag("CARRAY")
    }
    #[pygetset]
    fn farray(&self) -> bool {
        self.get_flag("FARRAY")
    }
}

impl AsMapping for PyFlagsObj {
    fn as_mapping() -> &'static PyMappingMethods {
        use once_cell::sync::Lazy;
        static AS_MAPPING: Lazy<PyMappingMethods> = Lazy::new(|| PyMappingMethods {
            length: atomic_func!(|_mapping, _vm| { Ok(0) }),
            subscript: atomic_func!(|mapping, needle: &vm::PyObject, vm| {
                let zelf = PyFlagsObj::mapping_downcast(mapping);
                let val = zelf.__getitem__(needle.to_owned(), vm)?;
                Ok(vm.ctx.new_bool(val).into())
            }),
            ass_subscript: atomic_func!(|_mapping, _needle: &vm::PyObject, _value, vm| {
                Err(vm.new_type_error("flagsobj does not support item assignment".to_owned()))
            }),
        });
        &AS_MAPPING
    }
}

impl Representable for PyFlagsObj {
    fn repr_str(zelf: &Py<Self>, _vm: &VirtualMachine) -> PyResult<String> {
        let mut parts = Vec::new();
        let display_order = [
            "C_CONTIGUOUS",
            "F_CONTIGUOUS",
            "OWNDATA",
            "WRITEABLE",
            "ALIGNED",
            "WRITEBACKIFCOPY",
        ];
        for &k in &display_order {
            if let Some(&v) = zelf.map.get(k) {
                let vstr = if v { "True" } else { "False" };
                parts.push(format!("  {k} : {vstr}"));
            }
        }
        Ok(parts.join("\n"))
    }
}
