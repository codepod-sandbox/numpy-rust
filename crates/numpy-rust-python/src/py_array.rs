use rustpython_vm as vm;
use vm::builtins::{PyList, PyStr, PyTuple};
use vm::protocol::PyNumberMethods;
use vm::types::AsNumber;
use vm::{PyObjectRef, PyPayload, PyRef, PyResult, VirtualMachine};

use numpy_rust_core::indexing::{Scalar, SliceArg};
use numpy_rust_core::{DType, NdArray};

/// Python-visible ndarray class wrapping the core NdArray.
#[vm::pyclass(module = "numpy", name = "ndarray")]
#[derive(Debug, PyPayload)]
pub struct PyNdArray {
    data: NdArray,
}

impl Clone for PyNdArray {
    fn clone(&self) -> Self {
        Self {
            data: self.data.clone(),
        }
    }
}

impl PyNdArray {
    pub fn from_core(data: NdArray) -> Self {
        Self { data }
    }

    pub fn inner(&self) -> &NdArray {
        &self.data
    }

    pub fn to_py(self, vm: &VirtualMachine) -> PyObjectRef {
        self.into_pyobject(vm)
    }
}

/// Convert a NumpyError to a Python exception.
fn numpy_err(e: numpy_rust_core::NumpyError, vm: &VirtualMachine) -> vm::builtins::PyBaseExceptionRef {
    vm.new_value_error(e.to_string())
}

/// Convert a Scalar to a Python object.
fn scalar_to_py(s: Scalar, vm: &VirtualMachine) -> PyObjectRef {
    match s {
        Scalar::Bool(v) => vm.ctx.new_bool(v).into(),
        Scalar::Int32(v) => vm.ctx.new_int(v).into(),
        Scalar::Int64(v) => vm.ctx.new_int(v).into(),
        Scalar::Float32(v) => vm.ctx.new_float(v as f64).into(),
        Scalar::Float64(v) => vm.ctx.new_float(v).into(),
    }
}

/// Parse a Python dtype string to DType.
fn parse_dtype(s: &str, vm: &VirtualMachine) -> PyResult<DType> {
    match s {
        "bool" => Ok(DType::Bool),
        "int32" | "i32" => Ok(DType::Int32),
        "int64" | "i64" | "int" => Ok(DType::Int64),
        "float32" | "f32" => Ok(DType::Float32),
        "float64" | "f64" | "float" => Ok(DType::Float64),
        _ => Err(vm.new_type_error(format!("unsupported dtype: {s}"))),
    }
}

/// Extract a shape tuple from a Python object (tuple or list of ints).
pub fn extract_shape(obj: &PyObjectRef, vm: &VirtualMachine) -> PyResult<Vec<usize>> {
    if let Some(tuple) = obj.payload::<PyTuple>() {
        let mut shape = Vec::new();
        for item in tuple.as_slice() {
            let n: i64 = item.clone().try_into_value(vm)?;
            shape.push(n as usize);
        }
        Ok(shape)
    } else if let Some(list) = obj.payload::<PyList>() {
        let mut shape = Vec::new();
        for item in list.borrow_vec().iter() {
            let n: i64 = item.clone().try_into_value(vm)?;
            shape.push(n as usize);
        }
        Ok(shape)
    } else if let Ok(n) = obj.clone().try_into_value::<i64>(vm) {
        Ok(vec![n as usize])
    } else {
        Err(vm.new_type_error("shape must be a tuple, list, or integer".into()))
    }
}

#[vm::pyclass(with(AsNumber))]
impl PyNdArray {
    // --- Properties ---

    #[pygetset]
    fn shape(&self, vm: &VirtualMachine) -> PyObjectRef {
        let shape: Vec<PyObjectRef> = self
            .data
            .shape()
            .iter()
            .map(|&s| vm.ctx.new_int(s).into())
            .collect();
        PyTuple::new_ref(shape, &vm.ctx).into()
    }

    #[pygetset]
    fn ndim(&self) -> usize {
        self.data.ndim()
    }

    #[pygetset]
    fn size(&self) -> usize {
        self.data.size()
    }

    #[pygetset]
    fn dtype(&self) -> String {
        self.data.dtype().to_string()
    }

    #[pygetset(name = "T")]
    fn transpose_prop(&self) -> PyNdArray {
        PyNdArray::from_core(self.data.transpose())
    }

    // --- Methods ---

    #[pymethod]
    fn reshape(&self, shape: PyObjectRef, vm: &VirtualMachine) -> PyResult<PyNdArray> {
        let sh = extract_shape(&shape, vm)?;
        self.data
            .reshape(&sh)
            .map(PyNdArray::from_core)
            .map_err(|e| numpy_err(e, vm))
    }

    #[pymethod]
    fn flatten(&self) -> PyNdArray {
        PyNdArray::from_core(self.data.flatten())
    }

    #[pymethod]
    fn ravel(&self) -> PyNdArray {
        PyNdArray::from_core(self.data.ravel())
    }

    #[pymethod]
    fn copy(&self) -> PyNdArray {
        PyNdArray::from_core(self.data.copy())
    }

    #[pymethod]
    fn astype(&self, dtype: PyRef<PyStr>, vm: &VirtualMachine) -> PyResult<PyNdArray> {
        let dt = parse_dtype(dtype.as_str(), vm)?;
        Ok(PyNdArray::from_core(self.data.astype(dt)))
    }

    #[pymethod]
    fn sum(
        &self,
        axis: vm::function::OptionalArg<PyObjectRef>,
        vm: &VirtualMachine,
    ) -> PyResult<PyNdArray> {
        let ax = parse_optional_axis(axis, vm)?;
        self.data
            .sum(ax)
            .map(PyNdArray::from_core)
            .map_err(|e| numpy_err(e, vm))
    }

    #[pymethod]
    fn mean(
        &self,
        axis: vm::function::OptionalArg<PyObjectRef>,
        vm: &VirtualMachine,
    ) -> PyResult<PyNdArray> {
        let ax = parse_optional_axis(axis, vm)?;
        self.data
            .mean(ax)
            .map(PyNdArray::from_core)
            .map_err(|e| numpy_err(e, vm))
    }

    #[pymethod]
    fn min(
        &self,
        axis: vm::function::OptionalArg<PyObjectRef>,
        vm: &VirtualMachine,
    ) -> PyResult<PyNdArray> {
        let ax = parse_optional_axis(axis, vm)?;
        self.data
            .min(ax)
            .map(PyNdArray::from_core)
            .map_err(|e| numpy_err(e, vm))
    }

    #[pymethod]
    fn max(
        &self,
        axis: vm::function::OptionalArg<PyObjectRef>,
        vm: &VirtualMachine,
    ) -> PyResult<PyNdArray> {
        let ax = parse_optional_axis(axis, vm)?;
        self.data
            .max(ax)
            .map(PyNdArray::from_core)
            .map_err(|e| numpy_err(e, vm))
    }

    #[pymethod]
    fn std(
        &self,
        axis: vm::function::OptionalArg<PyObjectRef>,
        vm: &VirtualMachine,
    ) -> PyResult<PyNdArray> {
        let ax = parse_optional_axis(axis, vm)?;
        self.data
            .std(ax)
            .map(PyNdArray::from_core)
            .map_err(|e| numpy_err(e, vm))
    }

    #[pymethod]
    fn var(
        &self,
        axis: vm::function::OptionalArg<PyObjectRef>,
        vm: &VirtualMachine,
    ) -> PyResult<PyNdArray> {
        let ax = parse_optional_axis(axis, vm)?;
        self.data
            .var(ax)
            .map(PyNdArray::from_core)
            .map_err(|e| numpy_err(e, vm))
    }

    #[pymethod]
    fn argmin(&self, vm: &VirtualMachine) -> PyResult<usize> {
        self.data.argmin().map_err(|e| numpy_err(e, vm))
    }

    #[pymethod]
    fn argmax(&self, vm: &VirtualMachine) -> PyResult<usize> {
        self.data.argmax().map_err(|e| numpy_err(e, vm))
    }

    #[pymethod]
    fn all(&self) -> bool {
        self.data.all()
    }

    #[pymethod]
    fn any(&self) -> bool {
        self.data.any()
    }

    // --- Unary math ---

    #[pymethod]
    fn abs(&self) -> PyNdArray {
        PyNdArray::from_core(self.data.abs())
    }

    #[pymethod]
    fn sqrt(&self) -> PyNdArray {
        PyNdArray::from_core(self.data.sqrt())
    }

    // --- Operators ---

    #[pymethod(magic)]
    fn add(&self, other: PyObjectRef, vm: &VirtualMachine) -> PyResult<PyNdArray> {
        let other = extract_ndarray(&other, vm)?;
        (&self.data + &other.data)
            .map(PyNdArray::from_core)
            .map_err(|e| numpy_err(e, vm))
    }

    #[pymethod(magic)]
    fn sub(&self, other: PyObjectRef, vm: &VirtualMachine) -> PyResult<PyNdArray> {
        let other = extract_ndarray(&other, vm)?;
        (&self.data - &other.data)
            .map(PyNdArray::from_core)
            .map_err(|e| numpy_err(e, vm))
    }

    #[pymethod(magic)]
    fn mul(&self, other: PyObjectRef, vm: &VirtualMachine) -> PyResult<PyNdArray> {
        let other = extract_ndarray(&other, vm)?;
        (&self.data * &other.data)
            .map(PyNdArray::from_core)
            .map_err(|e| numpy_err(e, vm))
    }

    #[pymethod(magic)]
    fn truediv(&self, other: PyObjectRef, vm: &VirtualMachine) -> PyResult<PyNdArray> {
        let other = extract_ndarray(&other, vm)?;
        (&self.data / &other.data)
            .map(PyNdArray::from_core)
            .map_err(|e| numpy_err(e, vm))
    }

    #[pymethod(magic)]
    fn neg(&self) -> PyNdArray {
        PyNdArray::from_core(self.data.neg())
    }

    #[pymethod(magic)]
    fn eq(&self, other: PyObjectRef, vm: &VirtualMachine) -> PyResult<PyNdArray> {
        let other = extract_ndarray(&other, vm)?;
        self.data
            .eq(&other.data)
            .map(PyNdArray::from_core)
            .map_err(|e| numpy_err(e, vm))
    }

    #[pymethod(magic)]
    fn ne(&self, other: PyObjectRef, vm: &VirtualMachine) -> PyResult<PyNdArray> {
        let other = extract_ndarray(&other, vm)?;
        self.data
            .ne(&other.data)
            .map(PyNdArray::from_core)
            .map_err(|e| numpy_err(e, vm))
    }

    #[pymethod(magic)]
    fn lt(&self, other: PyObjectRef, vm: &VirtualMachine) -> PyResult<PyNdArray> {
        let other = extract_ndarray(&other, vm)?;
        self.data
            .lt(&other.data)
            .map(PyNdArray::from_core)
            .map_err(|e| numpy_err(e, vm))
    }

    #[pymethod(magic)]
    fn gt(&self, other: PyObjectRef, vm: &VirtualMachine) -> PyResult<PyNdArray> {
        let other = extract_ndarray(&other, vm)?;
        self.data
            .gt(&other.data)
            .map(PyNdArray::from_core)
            .map_err(|e| numpy_err(e, vm))
    }

    #[pymethod(magic)]
    fn le(&self, other: PyObjectRef, vm: &VirtualMachine) -> PyResult<PyNdArray> {
        let other = extract_ndarray(&other, vm)?;
        self.data
            .le(&other.data)
            .map(PyNdArray::from_core)
            .map_err(|e| numpy_err(e, vm))
    }

    #[pymethod(magic)]
    fn ge(&self, other: PyObjectRef, vm: &VirtualMachine) -> PyResult<PyNdArray> {
        let other = extract_ndarray(&other, vm)?;
        self.data
            .ge(&other.data)
            .map(PyNdArray::from_core)
            .map_err(|e| numpy_err(e, vm))
    }

    // --- Indexing ---

    #[pymethod(magic)]
    fn getitem(&self, key: PyObjectRef, vm: &VirtualMachine) -> PyResult<PyObjectRef> {
        // Integer index -> scalar or sub-array
        if let Ok(idx) = key.clone().try_into_value::<i64>(vm) {
            let resolved = if idx < 0 {
                (self.data.shape()[0] as i64 + idx) as usize
            } else {
                idx as usize
            };

            if self.data.ndim() == 1 {
                let s = self
                    .data
                    .get(&[resolved])
                    .map_err(|e| numpy_err(e, vm))?;
                return Ok(scalar_to_py(s, vm));
            } else {
                let result = self
                    .data
                    .slice(&[SliceArg::Index(idx as isize)])
                    .map_err(|e| numpy_err(e, vm))?;
                return Ok(PyNdArray::from_core(result).to_py(vm));
            }
        }

        // Tuple index -> multi-dimensional indexing
        if let Some(tuple) = key.payload::<PyTuple>() {
            let mut indices = Vec::new();
            for item in tuple.as_slice() {
                let i: i64 = item.clone().try_into_value(vm)?;
                indices.push(i as isize);
            }
            if indices.len() == self.data.ndim() {
                let usize_indices: Vec<usize> = indices
                    .iter()
                    .enumerate()
                    .map(|(axis, &i)| {
                        if i < 0 {
                            (self.data.shape()[axis] as isize + i) as usize
                        } else {
                            i as usize
                        }
                    })
                    .collect();
                let s = self
                    .data
                    .get(&usize_indices)
                    .map_err(|e| numpy_err(e, vm))?;
                return Ok(scalar_to_py(s, vm));
            }
            let args: Vec<SliceArg> = indices.iter().map(|&i| SliceArg::Index(i)).collect();
            let result = self.data.slice(&args).map_err(|e| numpy_err(e, vm))?;
            return Ok(PyNdArray::from_core(result).to_py(vm));
        }

        // Boolean mask
        if let Some(mask) = key.payload::<PyNdArray>() {
            if mask.data.dtype() == DType::Bool {
                let result = self
                    .data
                    .mask_select(&mask.data)
                    .map_err(|e| numpy_err(e, vm))?;
                return Ok(PyNdArray::from_core(result).to_py(vm));
            }
        }

        Err(vm.new_type_error("unsupported index type".into()))
    }

    // --- String representations ---

    #[pymethod(magic)]
    fn repr(&self) -> String {
        format!(
            "ndarray(shape={:?}, dtype={})",
            self.data.shape(),
            self.data.dtype()
        )
    }

    #[pymethod(magic)]
    fn str(&self) -> String {
        format!(
            "ndarray(shape={:?}, dtype={})",
            self.data.shape(),
            self.data.dtype()
        )
    }

    #[pymethod(magic)]
    fn len(&self) -> usize {
        if self.data.ndim() == 0 {
            0
        } else {
            self.data.shape()[0]
        }
    }
}

/// Extract a PyNdArray from a PyObjectRef.
fn extract_ndarray<'a>(obj: &'a PyObjectRef, vm: &VirtualMachine) -> PyResult<&'a PyNdArray> {
    obj.payload::<PyNdArray>()
        .ok_or_else(|| vm.new_type_error("expected ndarray".into()))
}

// --- AsNumber implementation for operator dispatch ---

fn number_bin_op(
    a: &vm::PyObject,
    b: &vm::PyObject,
    op: fn(&NdArray, &NdArray) -> numpy_rust_core::Result<NdArray>,
    vm: &VirtualMachine,
) -> PyResult {
    let a = a
        .payload::<PyNdArray>()
        .ok_or_else(|| vm.new_type_error("expected ndarray".into()))?;
    let b = b
        .payload::<PyNdArray>()
        .ok_or_else(|| vm.new_type_error("expected ndarray".into()))?;
    op(&a.data, &b.data)
        .map(|r| PyNdArray::from_core(r).into_pyobject(vm))
        .map_err(|e| vm.new_value_error(e.to_string()))
}

fn number_neg(num: vm::protocol::PyNumber, vm: &VirtualMachine) -> PyResult {
    let a = num
        .payload::<PyNdArray>()
        .ok_or_else(|| vm.new_type_error("expected ndarray".into()))?;
    Ok(PyNdArray::from_core(a.data.neg()).into_pyobject(vm))
}

impl PyNdArray {
    const AS_NUMBER: PyNumberMethods = PyNumberMethods {
        add: Some(|a, b, vm| number_bin_op(a, b, |x, y| x + y, vm)),
        subtract: Some(|a, b, vm| number_bin_op(a, b, |x, y| x - y, vm)),
        multiply: Some(|a, b, vm| number_bin_op(a, b, |x, y| x * y, vm)),
        true_divide: Some(|a, b, vm| number_bin_op(a, b, |x, y| x / y, vm)),
        negative: Some(|a, vm| number_neg(a, vm)),
        ..PyNumberMethods::NOT_IMPLEMENTED
    };
}

impl AsNumber for PyNdArray {
    fn as_number() -> &'static PyNumberMethods {
        static AS_NUMBER: PyNumberMethods = PyNdArray::AS_NUMBER;
        &AS_NUMBER
    }
}

/// Parse an optional axis argument (None means reduce all).
fn parse_optional_axis(
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
