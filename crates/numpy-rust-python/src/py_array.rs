use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::RwLock;

use rustpython_vm as vm;
use vm::atomic_func;
use vm::builtins::{PyList, PySlice, PyStr, PyTuple};
use vm::protocol::{PyIterReturn, PyMappingMethods, PyNumberMethods, PySequenceMethods};
use vm::types::{AsMapping, AsNumber, AsSequence, IterNext, Iterable, Representable, SelfIter};
use vm::{AsObject, Py, PyObjectRef, PyPayload, PyRef, PyResult, VirtualMachine};

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
}

impl Clone for PyNdArray {
    fn clone(&self) -> Self {
        Self {
            data: RwLock::new(self.data.read().unwrap().clone()),
            is_fortran: AtomicBool::new(self.is_fortran.load(Ordering::Relaxed)),
            is_aligned: AtomicBool::new(self.is_aligned.load(Ordering::Relaxed)),
        }
    }
}

impl PyNdArray {
    pub fn from_core(data: NdArray) -> Self {
        Self {
            data: RwLock::new(data),
            is_fortran: AtomicBool::new(false),
            is_aligned: AtomicBool::new(true),
        }
    }

    pub fn from_core_fortran(data: NdArray) -> Self {
        Self {
            data: RwLock::new(data),
            is_fortran: AtomicBool::new(true),
            is_aligned: AtomicBool::new(true),
        }
    }

    pub fn inner(&self) -> std::sync::RwLockReadGuard<'_, NdArray> {
        self.data.read().unwrap()
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
        let scalar = py_obj_to_scalar(&value, data.dtype(), vm)?;
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
    vm.new_value_error(e.to_string())
}

/// Convert a Scalar to a Python object.
pub(crate) fn scalar_to_py(s: Scalar, vm: &VirtualMachine) -> PyObjectRef {
    match s {
        Scalar::Bool(v) => vm.ctx.new_bool(v).into(),
        Scalar::Int32(v) => vm.ctx.new_int(v).into(),
        Scalar::Int64(v) => vm.ctx.new_int(v).into(),
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

/// Convert a Python object to a Scalar matching the target array dtype.
/// For narrow dtypes, maps to the storage type's Scalar variant.
pub(crate) fn py_obj_to_scalar(
    obj: &PyObjectRef,
    dtype: DType,
    vm: &VirtualMachine,
) -> PyResult<Scalar> {
    // Use storage dtype for scalar conversion (narrow types map to wider storage)
    let storage = dtype.storage_dtype();
    if let Ok(f) = obj.clone().try_into_value::<f64>(vm) {
        return Ok(match storage {
            DType::Float64 => Scalar::Float64(f),
            DType::Float32 => Scalar::Float32(f as f32),
            DType::Int64 => Scalar::Int64(f as i64),
            DType::Int32 => Scalar::Int32(f as i32),
            DType::Bool => Scalar::Bool(f != 0.0),
            DType::Complex64 => Scalar::Complex64(num_complex::Complex::new(f as f32, 0.0)),
            DType::Complex128 => Scalar::Complex128(num_complex::Complex::new(f, 0.0)),
            DType::Str => Scalar::Str(f.to_string()),
            _ => unreachable!("storage_dtype maps to canonical types"),
        });
    }
    if let Ok(i) = obj.clone().try_into_value::<i64>(vm) {
        return Ok(match storage {
            DType::Float64 => Scalar::Float64(i as f64),
            DType::Float32 => Scalar::Float32(i as f32),
            DType::Int64 => Scalar::Int64(i),
            DType::Int32 => Scalar::Int32(i as i32),
            DType::Bool => Scalar::Bool(i != 0),
            DType::Complex64 => Scalar::Complex64(num_complex::Complex::new(i as f32, 0.0)),
            DType::Complex128 => Scalar::Complex128(num_complex::Complex::new(i as f64, 0.0)),
            DType::Str => Scalar::Str(i.to_string()),
            _ => unreachable!("storage_dtype maps to canonical types"),
        });
    }
    // Handle Python complex numbers
    if let Some(c) = obj.downcast_ref::<vm::builtins::PyComplex>() {
        let cx = c.to_complex();
        return Ok(match storage {
            DType::Complex64 => {
                Scalar::Complex64(num_complex::Complex::new(cx.re as f32, cx.im as f32))
            }
            DType::Complex128 => Scalar::Complex128(num_complex::Complex::new(cx.re, cx.im)),
            DType::Float64 => Scalar::Float64(cx.re),
            DType::Float32 => Scalar::Float32(cx.re as f32),
            DType::Int64 => Scalar::Int64(cx.re as i64),
            DType::Int32 => Scalar::Int32(cx.re as i32),
            DType::Bool => Scalar::Bool(cx.re != 0.0 || cx.im != 0.0),
            DType::Str => Scalar::Str(format!("({},{})", cx.re, cx.im)),
            _ => unreachable!("storage_dtype maps to canonical types"),
        });
    }
    // Handle (re, im) tuples (our complex scalar representation)
    if let Some(tup) = obj.downcast_ref::<vm::builtins::PyTuple>() {
        let elems = tup.as_slice();
        if elems.len() == 2 {
            let re = elems[0].clone().try_into_value::<f64>(vm).unwrap_or(0.0);
            let im = elems[1].clone().try_into_value::<f64>(vm).unwrap_or(0.0);
            return Ok(match storage {
                DType::Complex64 => {
                    Scalar::Complex64(num_complex::Complex::new(re as f32, im as f32))
                }
                DType::Complex128 => Scalar::Complex128(num_complex::Complex::new(re, im)),
                DType::Float64 => Scalar::Float64(re),
                DType::Float32 => Scalar::Float32(re as f32),
                DType::Int64 => Scalar::Int64(re as i64),
                DType::Int32 => Scalar::Int32(re as i32),
                DType::Bool => Scalar::Bool(re != 0.0 || im != 0.0),
                DType::Str => Scalar::Str(format!("({},{})", re, im)),
                _ => unreachable!("storage_dtype maps to canonical types"),
            });
        }
    }
    // Handle str objects for Str dtype
    if let Some(s) = obj.downcast_ref::<vm::builtins::PyStr>() {
        return Ok(match storage {
            DType::Str => Scalar::Str(s.as_str().to_owned()),
            DType::Float64 => Scalar::Float64(s.as_str().parse().unwrap_or(0.0)),
            DType::Float32 => Scalar::Float32(s.as_str().parse().unwrap_or(0.0)),
            DType::Int64 => Scalar::Int64(s.as_str().parse().unwrap_or(0)),
            DType::Int32 => Scalar::Int32(s.as_str().parse().unwrap_or(0)),
            DType::Bool => Scalar::Bool(!s.as_str().is_empty()),
            DType::Complex64 => Scalar::Complex64(num_complex::Complex::new(0.0, 0.0)),
            DType::Complex128 => Scalar::Complex128(num_complex::Complex::new(0.0, 0.0)),
            _ => unreachable!("storage_dtype maps to canonical types"),
        });
    }
    Err(vm.new_type_error("cannot convert value to array scalar".to_owned()))
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
        let s = arr.get(&[]).unwrap();
        scalar_to_py(s, vm)
    } else {
        PyNdArray::from_core(arr).into_pyobject(vm)
    }
}

/// Parse a Python dtype string to DType.
pub fn parse_dtype(s: &str, vm: &VirtualMachine) -> PyResult<DType> {
    match s {
        "bool" | "<class 'bool'>" => Ok(DType::Bool),
        "int8" | "i1" => Ok(DType::Int8),
        "int16" | "i2" => Ok(DType::Int16),
        "int32" | "i32" | "i4" => Ok(DType::Int32),
        "int64" | "i64" | "i8" | "int" | "<class 'int'>" => Ok(DType::Int64),
        "uint8" | "u1" => Ok(DType::UInt8),
        "uint16" | "u2" => Ok(DType::UInt16),
        "uint32" | "u4" => Ok(DType::UInt32),
        "uint64" | "u8" => Ok(DType::UInt64),
        "float16" | "f2" => Ok(DType::Float16),
        "float32" | "f32" | "f4" => Ok(DType::Float32),
        "float64" | "f64" | "float" | "<class 'float'>" => Ok(DType::Float64),
        // Compatibility fallback: map temporal dtypes to float64 until native
        // datetime/timedelta storage exists in the Rust core.
        "timedelta64" | "datetime64" | "m8" | "M8" | "<m8" | ">m8" | "<M8" | ">M8" => {
            Ok(DType::Float64)
        }
        "complex64" | "c64" | "c8" => Ok(DType::Complex64),
        "complex128" | "c128" | "c16" | "complex" | "<class 'complex'>" => Ok(DType::Complex128),
        "str" | "U" | "<class 'str'>" => Ok(DType::Str),
        _ if s.starts_with('S') || s.starts_with('U') => Ok(DType::Str),
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
    if let Some(tuple) = obj.downcast_ref::<PyTuple>() {
        let mut shape = Vec::new();
        for item in tuple.as_slice() {
            let n: i64 = item.clone().try_into_value(vm)?;
            shape.push(n as usize);
        }
        Ok(shape)
    } else if let Some(list) = obj.downcast_ref::<PyList>() {
        let mut shape = Vec::new();
        for item in list.borrow_vec().iter() {
            let n: i64 = item.clone().try_into_value(vm)?;
            shape.push(n as usize);
        }
        Ok(shape)
    } else if let Ok(n) = obj.clone().try_into_value::<i64>(vm) {
        Ok(vec![n as usize])
    } else {
        Err(vm.new_type_error("shape must be a tuple, list, or integer".to_owned()))
    }
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
    fn transpose_prop(&self) -> PyNdArray {
        let result = PyNdArray::from_core(self.data.read().unwrap().transpose());
        result
            .is_aligned
            .store(self.is_aligned.load(Ordering::Relaxed), Ordering::Relaxed);
        result
    }

    #[pygetset]
    fn real(&self) -> PyNdArray {
        PyNdArray::from_core(self.data.read().unwrap().real())
    }

    #[pygetset]
    fn imag(&self) -> PyNdArray {
        PyNdArray::from_core(self.data.read().unwrap().imag())
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
            vec![vm.ctx.new_int(0).into(), vm.ctx.new_bool(false).into()],
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
    fn reshape(&self, args: vm::function::FuncArgs, vm: &VirtualMachine) -> PyResult<PyNdArray> {
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
        let is_f = self.is_fortran.load(Ordering::Relaxed);
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
        let total = self.data.read().unwrap().size();
        let sh = resolve_shape(&raw, total, vm)?;

        // Perform the actual reshape
        let mut result_data = self
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
            is_aligned: AtomicBool::new(self.is_aligned.load(Ordering::Relaxed)),
        };

        Ok(result)
    }

    #[pymethod]
    fn flatten(&self) -> PyNdArray {
        PyNdArray::from_core(self.data.read().unwrap().flatten())
    }

    #[pymethod]
    fn ravel(&self) -> PyNdArray {
        PyNdArray::from_core(self.data.read().unwrap().ravel())
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
    fn astype(&self, args: vm::function::FuncArgs, vm: &VirtualMachine) -> PyResult<PyNdArray> {
        let dtype_obj = args
            .args
            .first()
            .ok_or_else(|| vm.new_type_error("astype() requires a dtype argument".to_owned()))?;
        // Accept either a string or any object convertible via str()
        let dtype_str: PyRef<PyStr> = match dtype_obj.clone().try_into_value::<PyRef<PyStr>>(vm) {
            Ok(s) => s,
            Err(_) => {
                // Convert to string using Python's str() builtin
                dtype_obj.str(vm)?
            }
        };
        let dt = parse_dtype(dtype_str.as_str(), vm)?;

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
        Ok(PyNdArray::from_core(inner.astype(dt)))
    }

    /// Returns a view of the array sharing the same underlying buffer.
    /// Clone is O(1) via ArcArray reference counting.
    /// When passed a Python type (subclass of ndarray), returns an instance of that type.
    #[pymethod]
    fn view(
        &self,
        dtype_arg: vm::function::OptionalArg<PyObjectRef>,
        vm: &VirtualMachine,
    ) -> PyResult<PyObjectRef> {
        let data = self.data.read().unwrap();
        let is_f = self.is_fortran.load(Ordering::Relaxed);
        if let Some(dtype_obj) = dtype_arg.into_option() {
            // If it's a Python type/class (e.g. MyNDArray subclass), return instance of that type
            if let Ok(py_type) = dtype_obj.clone().downcast::<vm::builtins::PyType>() {
                let result = PyNdArray {
                    data: RwLock::new(data.clone()),
                    is_fortran: AtomicBool::new(is_f),
                    is_aligned: AtomicBool::new(self.is_aligned.load(Ordering::Relaxed)),
                };
                return Ok(result.into_ref_with_type(vm, py_type)?.into());
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
    fn sum(
        &self,
        axis: vm::function::OptionalArg<PyObjectRef>,
        keepdims: vm::function::OptionalArg<bool>,
        vm: &VirtualMachine,
    ) -> PyResult<PyObjectRef> {
        let inner = self.data.read().unwrap();
        let kd = keepdims.unwrap_or(false);
        let ax = parse_axis_arg(axis, inner.ndim(), vm)?;
        match ax {
            AxisArg::None => inner.sum(None, kd),
            AxisArg::Single(a) => inner.sum(Some(a), kd),
            AxisArg::Multi(axes) => reduce_multi_axis(&inner, &axes, |arr, a| arr.sum(a, false)),
        }
        .map(|arr| ndarray_or_scalar(arr, vm))
        .map_err(|e| numpy_err(e, vm))
    }

    #[pymethod]
    fn mean(
        &self,
        axis: vm::function::OptionalArg<PyObjectRef>,
        keepdims: vm::function::OptionalArg<bool>,
        vm: &VirtualMachine,
    ) -> PyResult<PyObjectRef> {
        let inner = self.data.read().unwrap();
        let kd = keepdims.unwrap_or(false);
        let ax = parse_axis_arg(axis, inner.ndim(), vm)?;
        match ax {
            AxisArg::None => inner.mean(None, kd),
            AxisArg::Single(a) => inner.mean(Some(a), kd),
            AxisArg::Multi(axes) => reduce_multi_axis(&inner, &axes, |arr, a| arr.mean(a, false)),
        }
        .map(|arr| ndarray_or_scalar(arr, vm))
        .map_err(|e| numpy_err(e, vm))
    }

    #[pymethod]
    fn min(
        &self,
        axis: vm::function::OptionalArg<PyObjectRef>,
        keepdims: vm::function::OptionalArg<bool>,
        vm: &VirtualMachine,
    ) -> PyResult<PyObjectRef> {
        let inner = self.data.read().unwrap();
        let kd = keepdims.unwrap_or(false);
        let ax = parse_axis_arg(axis, inner.ndim(), vm)?;
        match ax {
            AxisArg::None => inner.min(None, kd),
            AxisArg::Single(a) => inner.min(Some(a), kd),
            AxisArg::Multi(axes) => reduce_multi_axis(&inner, &axes, |arr, a| arr.min(a, false)),
        }
        .map(|arr| ndarray_or_scalar(arr, vm))
        .map_err(|e| numpy_err(e, vm))
    }

    #[pymethod]
    fn max(
        &self,
        axis: vm::function::OptionalArg<PyObjectRef>,
        keepdims: vm::function::OptionalArg<bool>,
        vm: &VirtualMachine,
    ) -> PyResult<PyObjectRef> {
        let inner = self.data.read().unwrap();
        let kd = keepdims.unwrap_or(false);
        let ax = parse_axis_arg(axis, inner.ndim(), vm)?;
        match ax {
            AxisArg::None => inner.max(None, kd),
            AxisArg::Single(a) => inner.max(Some(a), kd),
            AxisArg::Multi(axes) => reduce_multi_axis(&inner, &axes, |arr, a| arr.max(a, false)),
        }
        .map(|arr| ndarray_or_scalar(arr, vm))
        .map_err(|e| numpy_err(e, vm))
    }

    #[pymethod]
    fn std(
        &self,
        axis: vm::function::OptionalArg<PyObjectRef>,
        ddof: vm::function::OptionalArg<usize>,
        keepdims: vm::function::OptionalArg<bool>,
        vm: &VirtualMachine,
    ) -> PyResult<PyObjectRef> {
        let inner = self.data.read().unwrap();
        let dd = ddof.unwrap_or(0);
        let kd = keepdims.unwrap_or(false);
        let ax = parse_axis_arg(axis, inner.ndim(), vm)?;
        match ax {
            AxisArg::None => inner.std(None, dd, kd),
            AxisArg::Single(a) => inner.std(Some(a), dd, kd),
            AxisArg::Multi(axes) => std_multi_axis(&inner, &axes, dd),
        }
        .map(|arr| ndarray_or_scalar(arr, vm))
        .map_err(|e| numpy_err(e, vm))
    }

    #[pymethod]
    fn var(
        &self,
        axis: vm::function::OptionalArg<PyObjectRef>,
        ddof: vm::function::OptionalArg<usize>,
        keepdims: vm::function::OptionalArg<bool>,
        vm: &VirtualMachine,
    ) -> PyResult<PyObjectRef> {
        let inner = self.data.read().unwrap();
        let dd = ddof.unwrap_or(0);
        let kd = keepdims.unwrap_or(false);
        let ax = parse_axis_arg(axis, inner.ndim(), vm)?;
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
    fn swapaxes(&self, axis1: usize, axis2: usize, vm: &VirtualMachine) -> PyResult<PyNdArray> {
        self.data
            .read()
            .unwrap()
            .swapaxes(axis1, axis2)
            .map(PyNdArray::from_core)
            .map_err(|e| numpy_err(e, vm))
    }

    #[pymethod]
    fn take(
        &self,
        indices: PyRef<PyNdArray>,
        axis: vm::function::OptionalArg<PyObjectRef>,
        vm: &VirtualMachine,
    ) -> PyResult<PyNdArray> {
        let ax = parse_optional_axis(axis, vm)?;
        let inner = self.data.read().unwrap();
        let idx_arr = indices.data.read().unwrap();
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
    fn prod(
        &self,
        axis: vm::function::OptionalArg<PyObjectRef>,
        keepdims: vm::function::OptionalArg<bool>,
        vm: &VirtualMachine,
    ) -> PyResult<PyObjectRef> {
        let inner = self.data.read().unwrap();
        let kd = keepdims.unwrap_or(false);
        let ax = parse_axis_arg(axis, inner.ndim(), vm)?;
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
    fn __getitem__(&self, key: PyObjectRef, vm: &VirtualMachine) -> PyResult<PyObjectRef> {
        let data = self.data.read().unwrap();

        // Integer index -> scalar or sub-array
        if let Ok(idx) = key.clone().try_into_value::<i64>(vm) {
            let resolved = if idx < 0 {
                (data.shape()[0] as i64 + idx) as usize
            } else {
                idx as usize
            };

            if data.ndim() == 1 {
                let s = data.get(&[resolved]).map_err(|e| numpy_err(e, vm))?;
                return Ok(scalar_to_py(s, vm));
            } else {
                let result = data
                    .slice(&[SliceArg::Index(idx as isize)])
                    .map_err(|e| numpy_err(e, vm))?;
                return Ok(PyNdArray::from_core(result).to_py(vm));
            }
        }

        // Single slice -> e.g. a[0:3], a[::2], a[:]
        if let Some(slice) = key.downcast_ref::<PySlice>() {
            let arg = py_slice_to_slice_arg(slice, vm)?;
            let result = data.slice(&[arg]).map_err(|e| numpy_err(e, vm))?;
            return Ok(PyNdArray::from_core(result).to_py(vm));
        }

        // Tuple index -> multi-dimensional indexing (integers and/or slices)
        if let Some(tuple) = key.downcast_ref::<PyTuple>() {
            let items = tuple.as_slice();

            // Check if any element is an ndarray (fancy indexing) or boolean ndarray
            let has_ndarray = items
                .iter()
                .any(|item| item.downcast_ref::<PyNdArray>().is_some());
            let has_slice = items
                .iter()
                .any(|item| item.downcast_ref::<PySlice>().is_some());

            if has_ndarray {
                // Multi-dimensional fancy indexing: a[arr1, arr2, ...]
                // Also handles mixed: a[:, arr] or a[arr, 3]
                return multi_dim_fancy_getitem(&data, items, vm);
            }

            if has_slice {
                let args: Vec<SliceArg> = items
                    .iter()
                    .map(|item| py_obj_to_slice_arg(item, vm))
                    .collect::<PyResult<Vec<_>>>()?;
                let result = data.slice(&args).map_err(|e| numpy_err(e, vm))?;
                return Ok(ndarray_or_scalar(result, vm));
            }

            // All integers
            let mut indices = Vec::new();
            for item in items {
                let i: i64 = item.clone().try_into_value(vm)?;
                indices.push(i as isize);
            }
            if indices.len() == data.ndim() {
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
                return Ok(scalar_to_py(s, vm));
            }
            let args: Vec<SliceArg> = indices.iter().map(|&i| SliceArg::Index(i)).collect();
            let result = data.slice(&args).map_err(|e| numpy_err(e, vm))?;
            return Ok(PyNdArray::from_core(result).to_py(vm));
        }

        // NdArray index: boolean mask or integer fancy indexing
        if let Some(arr) = key.downcast_ref::<PyNdArray>() {
            let arr_data = arr.data.read().unwrap();
            if arr_data.dtype() == DType::Bool {
                let result = data.mask_select(&arr_data).map_err(|e| numpy_err(e, vm))?;
                return Ok(PyNdArray::from_core(result).to_py(vm));
            } else if arr_data.dtype().is_integer() || arr_data.dtype().is_float() {
                let indices = extract_int_indices(&arr_data, data.shape()[0], vm)?;
                let result = data
                    .index_select(0, &indices)
                    .map_err(|e| numpy_err(e, vm))?;
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

        // Ellipsis -> return a copy/view of the entire array
        if key.is(&vm.ctx.ellipsis) {
            let result = data.clone();
            return Ok(PyNdArray::from_core(result).to_py(vm));
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
        let other = match obj_to_ndarray_nep50(other, zelf_dtype, vm) {
            Ok(arr) => arr,
            Err(_) => {
                return Ok(vm::function::Either::B(
                    vm::function::PyComparisonValue::NotImplemented,
                ))
            }
        };
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

    #[pymethod]
    fn item(&self, vm: &VirtualMachine) -> PyResult<PyObjectRef> {
        let data = self.data.read().unwrap();
        if data.size() != 1 {
            return Err(vm.new_value_error(
                "can only convert an array of size 1 to a Python scalar".to_owned(),
            ));
        }
        let s = data
            .get(&vec![0; data.ndim()])
            .map_err(|e| numpy_err(e, vm))?;
        Ok(scalar_to_py(s, vm))
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
        let shape = inner.shape();
        let itemsize = inner.dtype().itemsize();
        // Compute C-contiguous strides
        let ndim = shape.len();
        let mut strides_vec = vec![0usize; ndim];
        if ndim > 0 {
            strides_vec[ndim - 1] = itemsize;
            for i in (0..ndim - 1).rev() {
                strides_vec[i] = strides_vec[i + 1] * shape[i + 1];
            }
        }
        let py_strides: Vec<PyObjectRef> = strides_vec
            .iter()
            .map(|&s| vm.ctx.new_int(s).into())
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
    fn ptp(
        &self,
        axis: vm::function::OptionalArg<PyObjectRef>,
        vm: &VirtualMachine,
    ) -> PyResult<PyObjectRef> {
        let ax = parse_optional_axis(axis, vm)?;
        let inner = self.data.read().unwrap();
        let max_val = inner.max(ax, false).map_err(|e| numpy_err(e, vm))?;
        let min_val = inner.min(ax, false).map_err(|e| numpy_err(e, vm))?;
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
    fn put(
        &self,
        indices: PyRef<PyNdArray>,
        values: PyRef<PyNdArray>,
        vm: &VirtualMachine,
    ) -> PyResult<()> {
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
        let self_dtype = self.data.read().unwrap().dtype();
        let py_arrays: Vec<PyRef<PyNdArray>> = items
            .iter()
            .map(|item| {
                match item.clone().try_into_value::<PyRef<PyNdArray>>(vm) {
                    Ok(arr) => Ok(arr),
                    Err(_) => {
                        // Try to convert scalar to a full array matching self's shape
                        let val = py_obj_to_f64(item, vm)?;
                        let arr = PyNdArray::from_core(numpy_rust_core::full(
                            &self_shape,
                            val,
                            self_dtype,
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
        let is_f = self.is_fortran.load(Ordering::Relaxed);
        let is_c = !is_f;
        let is_aligned = self.is_aligned.load(Ordering::Relaxed);
        let mut map = HashMap::new();
        map.insert("C_CONTIGUOUS".into(), is_c);
        map.insert("F_CONTIGUOUS".into(), is_f);
        map.insert("OWNDATA".into(), true);
        map.insert("WRITEABLE".into(), true);
        map.insert("ALIGNED".into(), is_aligned);
        map.insert("WRITEBACKIFCOPY".into(), false);
        // Short aliases
        map.insert("C".into(), is_c);
        map.insert("F".into(), is_f);
        map.insert("O".into(), true);
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
        vm.ctx.none() // We always own our data
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
    fn setflags(
        &self,
        _write: vm::function::OptionalArg<bool>,
        _align: vm::function::OptionalArg<bool>,
        _uic: vm::function::OptionalArg<bool>,
    ) {
        // No-op: flags are fixed in our implementation
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
    if let Some(arr) = obj.downcast_ref::<PyNdArray>() {
        return Ok(arr.data.read().unwrap().clone());
    }
    // Preserve numpy scalar wrapper dtype when available (set by python/numpy/__init__.py).
    if let Ok(dtype_name_obj) = obj.get_attr("_numpy_dtype_name", vm) {
        if let Ok(dtype_name) = dtype_name_obj.try_into_value::<String>(vm) {
            let dt = parse_dtype(dtype_name.as_str(), vm)?;
            if dt == DType::Bool {
                let b = obj.to_owned().try_into_value::<bool>(vm)?;
                let arr = NdArray::from_vec(vec![b]);
                return arr
                    .reshape(&[])
                    .map_err(|e| vm.new_type_error(e.to_string()));
            }
            if dt.is_complex() {
                let c = obj
                    .to_owned()
                    .try_into_value::<vm::function::ArgIntoComplex>(vm)?
                    .into_complex();
                let base =
                    NdArray::from_complex128_vec(vec![num_complex::Complex::new(c.re, c.im)])
                        .reshape(&[])
                        .map_err(|e| vm.new_type_error(e.to_string()))?;
                return Ok(base.astype(dt));
            }
            let f = obj.to_owned().try_into_value::<f64>(vm)?;
            return Ok(NdArray::from_scalar(f).astype(dt));
        }
    }
    // Preserve Python bool scalars as bool dtype (not float64).
    if obj.class().is(vm.ctx.types.bool_type) {
        let b = obj.to_owned().try_into_value::<bool>(vm)?;
        let arr = NdArray::from_vec(vec![b]);
        return arr
            .reshape(&[])
            .map_err(|e| vm.new_type_error(e.to_string()));
    }
    // Try generic float conversion first (handles large ints via __float__)
    if let Ok(f) = obj.to_owned().try_into_value::<f64>(vm) {
        return Ok(NdArray::from_scalar(f));
    }
    // Try generic complex conversion
    if let Ok(c) = obj
        .to_owned()
        .try_into_value::<vm::function::ArgIntoComplex>(vm)
    {
        let z = c.into_complex();
        if z.im == 0.0 {
            return Ok(NdArray::from_scalar(z.re));
        }
        let arr = NdArray::from_complex128_vec(vec![num_complex::Complex::new(z.re, z.im)]);
        return arr
            .reshape(&[])
            .map_err(|e| vm.new_type_error(e.to_string()));
    }
    // Try float (PyFloat)
    if let Some(f) = obj.downcast_ref::<vm::builtins::PyFloat>() {
        return Ok(NdArray::from_scalar(f.to_f64()));
    }
    // Try int (PyInt) — extract i64 then convert to f64; fall back to __float__ for large ints
    if let Some(i) = obj.downcast_ref::<vm::builtins::PyInt>() {
        if let Ok(val) = i.try_to_primitive::<i64>(vm) {
            return Ok(NdArray::from_scalar(val as f64));
        }
        // Large int that overflows i64 — convert via float (gives inf/-inf for huge values)
        let f = vm.call_method(obj, "__float__", ())?;
        if let Ok(fv) = f.try_into_value::<f64>(vm) {
            return Ok(NdArray::from_scalar(fv));
        }
    }
    // Try str (PyStr) — create 0-D string array
    if let Some(s) = obj.downcast_ref::<vm::builtins::PyStr>() {
        return Ok(NdArray::from_vec(vec![s.as_str().to_owned()]));
    }
    // Try list/tuple → convert to array of floats
    if let Some(list) = obj.downcast_ref::<vm::builtins::PyList>() {
        let items = list.borrow_vec();
        let mut vals = Vec::with_capacity(items.len());
        for item in items.iter() {
            if let Some(f) = item.downcast_ref::<vm::builtins::PyFloat>() {
                vals.push(f.to_f64());
            } else if let Some(i) = item.downcast_ref::<vm::builtins::PyInt>() {
                let v: i64 = i.try_to_primitive(vm)?;
                vals.push(v as f64);
            } else {
                return Err(vm.new_type_error("expected ndarray or scalar".to_owned()));
            }
        }
        return Ok(NdArray::from_vec(vals));
    }
    if let Some(tuple) = obj.downcast_ref::<vm::builtins::PyTuple>() {
        let items = tuple.as_slice();
        let mut vals = Vec::with_capacity(items.len());
        for item in items.iter() {
            if let Some(f) = item.downcast_ref::<vm::builtins::PyFloat>() {
                vals.push(f.to_f64());
            } else if let Some(i) = item.downcast_ref::<vm::builtins::PyInt>() {
                let v: i64 = i.try_to_primitive(vm)?;
                vals.push(v as f64);
            } else {
                return Err(vm.new_type_error("expected ndarray or scalar".to_owned()));
            }
        }
        return Ok(NdArray::from_vec(vals));
    }
    Err(vm.new_type_error("expected ndarray or scalar".to_owned()))
}

/// NEP50-aware scalar conversion: plain Python scalars (int, float) adopt the target dtype
/// (weak typing), while numpy scalars and arrays keep their explicit dtype (strong typing).
fn obj_to_ndarray_nep50(
    obj: &vm::PyObject,
    target_dtype: DType,
    vm: &VirtualMachine,
) -> PyResult<NdArray> {
    // numpy array: strong typing, unchanged
    if obj.downcast_ref::<PyNdArray>().is_some() {
        return obj_to_ndarray(obj, vm);
    }
    // numpy scalar (has _numpy_dtype_name): strong typing, unchanged
    if obj.get_attr("_numpy_dtype_name", vm).is_ok() {
        return obj_to_ndarray(obj, vm);
    }
    // Python bool: stays bool (don't demote to integer/float)
    if obj.class().is(vm.ctx.types.bool_type) {
        return obj_to_ndarray(obj, vm);
    }
    // Plain Python float or int: weak typing → cast to target_dtype
    if obj.class().is(vm.ctx.types.float_type) {
        if let Ok(f) = obj.to_owned().try_into_value::<f64>(vm) {
            return Ok(NdArray::from_scalar(f).astype(target_dtype));
        }
    }
    if obj.class().is(vm.ctx.types.int_type) {
        if let Some(i) = obj.downcast_ref::<vm::builtins::PyInt>() {
            if let Ok(val) = i.try_to_primitive::<i64>(vm) {
                return Ok(NdArray::from_scalar(val as f64).astype(target_dtype));
            }
        }
    }
    // Fallback: use normal conversion
    obj_to_ndarray(obj, vm)
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
                            // Check if result contains inf
                            if let Some(arr) = result_obj.downcast_ref::<PyNdArray>() {
                                let data = arr.data.read().unwrap();
                                let has_inf = data.has_inf();
                                if has_inf {
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
    let a_arr = obj_to_ndarray(a, vm)?;
    let b_arr = obj_to_ndarray(b, vm)?;
    op(&a_arr, &b_arr)
        .map(|r| PyNdArray::from_core(r).into_pyobject(vm))
        .map_err(|e| vm.new_value_error(e.to_string()))
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
        add: Some(|a, b, vm| number_bin_op(a, b, |x, y| x + y, vm)),
        subtract: Some(|a, b, vm| number_bin_op(a, b, |x, y| x - y, vm)),
        multiply: Some(|a, b, vm| number_bin_op(a, b, |x, y| x * y, vm)),
        true_divide: Some(|a, b, vm| {
            let result = number_bin_op(a, b, |x, y| x / y, vm)?;
            // Check numpy errstate for divide-by-zero
            check_division_errstate(&result, vm)?;
            Ok(result)
        }),
        floor_divide: Some(|a, b, vm| number_bin_op(a, b, |x, y| x.floor_div(y), vm)),
        remainder: Some(|a, b, vm| number_bin_op(a, b, |x, y| x.remainder(y), vm)),
        power: Some(|a, b, _modulo, vm| {
            let a_arr = obj_to_ndarray(a, vm)?;
            let b_arr = obj_to_ndarray(b, vm)?;
            a_arr
                .pow(&b_arr)
                .map(|r| PyNdArray::from_core(r).into_pyobject(vm))
                .map_err(|e| vm.new_value_error(e.to_string()))
        }),
        negative: Some(number_neg),
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
                zelf.__getitem__(needle.to_owned(), vm)
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
            return Err(vm.new_value_error(format!(
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
    if n == 0 {
        // No array indices — fallback to slice/integer logic
        let args: Vec<SliceArg> = kinds
            .into_iter()
            .map(|k| match k {
                IdxKind::Scalar(i) => SliceArg::Index(i as isize),
                IdxKind::Slice(s) => s,
                _ => unreachable!(),
            })
            .collect();
        let result = data.slice(&args).map_err(|e| numpy_err(e, vm))?;
        return Ok(ndarray_or_scalar(result, vm));
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

/// Convert a RustPython `PySlice` to a core `SliceArg`.
fn py_slice_to_slice_arg(slice: &PySlice, vm: &VirtualMachine) -> PyResult<SliceArg> {
    let start = match &slice.start {
        Some(obj) if !vm.is_none(obj) => {
            let v: i64 = obj.clone().try_into_value(vm)?;
            Some(v as isize)
        }
        _ => None,
    };
    let stop = if vm.is_none(&slice.stop) {
        None
    } else {
        let v: i64 = slice.stop.clone().try_into_value(vm)?;
        Some(v as isize)
    };
    let step = match &slice.step {
        Some(obj) if !vm.is_none(obj) => {
            let v: i64 = obj.clone().try_into_value(vm)?;
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
    // Integer key → set single element or row
    if let Ok(idx) = key.clone().try_into_value::<i64>(vm) {
        let data = zelf.data.read().unwrap();
        let shape = data.shape().to_vec();
        let ndim = data.ndim();
        let dtype = data.dtype();
        drop(data);

        let resolved = if idx < 0 {
            (shape[0] as i64 + idx) as usize
        } else {
            idx as usize
        };

        if ndim == 1 {
            // Scalar assignment: a[i] = value
            let scalar = py_obj_to_scalar(&value, dtype, vm)?;
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
        // Cast value to target dtype if needed
        let target_dt = zelf.data.read().unwrap().dtype();
        let value_arr = if value_arr.dtype() != target_dt {
            value_arr.astype(target_dt)
        } else {
            value_arr
        };
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
        let dtype = data.dtype();
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
        let scalar = py_obj_to_scalar(&value, dtype, vm)?;
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
        // If value is a scalar (size 1), fill the entire array
        let val_data = value_arr;
        let target_dt = zelf.data.read().unwrap().dtype();
        let val_data = if val_data.dtype() != target_dt {
            val_data.astype(target_dt)
        } else {
            val_data
        };
        // Use set_slice with full slice for each dimension
        let ndim = zelf.data.read().unwrap().ndim();
        let args: Vec<SliceArg> = (0..ndim).map(|_| SliceArg::Full).collect();
        zelf.data
            .write()
            .unwrap()
            .set_slice(&args, &val_data)
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
