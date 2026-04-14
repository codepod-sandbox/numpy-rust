use crate::array_data::ArrayD;
use ndarray::{Axis, IxDyn};

use crate::array::{BoxedObjectScalar, BoxedScalar};
use crate::array_data::ArrayData;
use crate::descriptor::descriptor_for_dtype;
use crate::dtype::DType;
use crate::error::{NumpyError, Result};
use crate::kernel::{ArgReductionKernelOp, ReductionKernelOp, TruthReduceKernelOp};
use crate::ops::comparison::{compare_boxed_scalars, iter_boxed_coords};
use crate::resolver::BinaryOp;
use crate::resolver::ReductionOp;
use crate::NdArray;

/// Validate an axis index against the array's dimensionality.
fn validate_axis(axis: usize, ndim: usize) -> Result<()> {
    if axis >= ndim {
        return Err(NumpyError::InvalidAxis { axis, ndim });
    }
    Ok(())
}

/// If `keepdims` is true and a specific axis was reduced, re-insert a size-1
/// dimension at `axis` so the output rank matches the input rank.
fn maybe_keepdims(
    result: NdArray,
    axis: Option<usize>,
    keepdims: bool,
    original_ndim: usize,
) -> NdArray {
    if !keepdims {
        return result;
    }
    if let Some(ax) = axis {
        let mut new_shape = result.shape().to_vec();
        new_shape.insert(ax, 1);
        result
            .reshape(&new_shape)
            .expect("keepdims reshape cannot fail")
    } else {
        // axis=None reduced everything to scalar — wrap in shape (1, 1, ..., 1)
        let new_shape = vec![1; original_ndim];
        result
            .reshape(&new_shape)
            .expect("keepdims reshape cannot fail")
    }
}

fn choose_boxed_extrema(
    lhs: BoxedScalar,
    rhs: BoxedScalar,
    reduce_max: bool,
) -> Result<BoxedScalar> {
    match (&lhs, &rhs) {
        (BoxedScalar::Datetime(a), BoxedScalar::Datetime(b))
        | (BoxedScalar::Timedelta(a), BoxedScalar::Timedelta(b)) => {
            if a.is_nat {
                return Ok(lhs);
            }
            if b.is_nat {
                return Ok(rhs);
            }
        }
        _ => {}
    }

    let cmp = compare_boxed_scalars(&lhs, &rhs)?;
    let pick_lhs = if reduce_max {
        matches!(
            cmp,
            Some(std::cmp::Ordering::Greater) | Some(std::cmp::Ordering::Equal)
        )
    } else {
        matches!(
            cmp,
            Some(std::cmp::Ordering::Less) | Some(std::cmp::Ordering::Equal)
        )
    };
    Ok(if pick_lhs { lhs } else { rhs })
}

fn empty_extrema_error() -> NumpyError {
    NumpyError::ValueError("zero-size array to reduction operation has no identity".into())
}

impl NdArray {
    /// Sum of array elements over a given axis, or all elements if axis is None.
    pub fn sum(&self, axis: Option<usize>, keepdims: bool) -> Result<NdArray> {
        if self.dtype().is_string() {
            return Err(NumpyError::TypeError(
                "sum not supported for string arrays".into(),
            ));
        }
        if self.dtype() == DType::Object {
            let result = match axis {
                None => self.reduce_all_boxed_object(BinaryOp::Add, BoxedObjectScalar::Int(0)),
                Some(ax) => {
                    self.reduce_axis_boxed_object(ax, BinaryOp::Add, BoxedObjectScalar::Int(0))
                }
            }?;
            return Ok(maybe_keepdims(result, axis, keepdims, self.ndim()));
        }
        let result = match axis {
            None => self.reduce_all_sum(),
            Some(ax) => self.reduce_axis_sum(ax),
        }?;
        Ok(maybe_keepdims(result, axis, keepdims, self.ndim()))
    }

    /// Mean of array elements. Returns Float64, or Complex128 for complex inputs.
    pub fn mean(&self, axis: Option<usize>, keepdims: bool) -> Result<NdArray> {
        if self.dtype().is_string() {
            return Err(NumpyError::TypeError(
                "mean not supported for string arrays".into(),
            ));
        }
        if self.dtype() == DType::Object {
            let sum = self.sum(axis, false)?;
            let count = match axis {
                None => self.size(),
                Some(ax) => {
                    validate_axis(ax, self.ndim())?;
                    self.shape()[ax]
                }
            } as i64;
            let divisor = NdArray::from_boxed_scalars(
                vec![BoxedScalar::Object(BoxedObjectScalar::Int(count))],
                &[],
                DType::Object,
            )?;
            let result = (&sum / &divisor)?;
            return Ok(maybe_keepdims(result, axis, keepdims, self.ndim()));
        }
        let target_dtype = if self.dtype().is_complex() {
            DType::Complex128
        } else {
            DType::Float64
        };
        let sum = self.astype(target_dtype).sum(axis, false)?;
        let count = match axis {
            None => self.size(),
            Some(ax) => {
                validate_axis(ax, self.ndim())?;
                self.shape()[ax]
            }
        };
        let divisor = NdArray::full_f64(sum.shape(), count as f64);
        let result = (&sum / &divisor)?;
        Ok(maybe_keepdims(result, axis, keepdims, self.ndim()))
    }

    /// Minimum element over a given axis, or global minimum.
    pub fn min(&self, axis: Option<usize>, keepdims: bool) -> Result<NdArray> {
        let result = match axis {
            None => self.reduce_all_min(),
            Some(ax) => self.reduce_axis_fold(ax, ReduceOp::Min),
        }?;
        Ok(maybe_keepdims(result, axis, keepdims, self.ndim()))
    }

    /// Maximum element over a given axis, or global maximum.
    pub fn max(&self, axis: Option<usize>, keepdims: bool) -> Result<NdArray> {
        let result = match axis {
            None => self.reduce_all_max(),
            Some(ax) => self.reduce_axis_fold(ax, ReduceOp::Max),
        }?;
        Ok(maybe_keepdims(result, axis, keepdims, self.ndim()))
    }

    /// Product of array elements over a given axis, or all elements if axis is None.
    pub fn prod(&self, axis: Option<usize>, keepdims: bool) -> Result<NdArray> {
        if self.dtype() == DType::Object {
            let result = match axis {
                None => self.reduce_all_boxed_object(BinaryOp::Mul, BoxedObjectScalar::Int(1)),
                Some(ax) => {
                    self.reduce_axis_boxed_object(ax, BinaryOp::Mul, BoxedObjectScalar::Int(1))
                }
            }?;
            return Ok(maybe_keepdims(result, axis, keepdims, self.ndim()));
        }
        let result = match axis {
            None => self.reduce_all_prod(),
            Some(ax) => self.reduce_axis_prod(ax),
        }?;
        Ok(maybe_keepdims(result, axis, keepdims, self.ndim()))
    }

    /// Standard deviation. Always returns Float64.
    pub fn std(&self, axis: Option<usize>, ddof: usize, keepdims: bool) -> Result<NdArray> {
        if self.dtype().is_string() {
            return Err(NumpyError::TypeError(
                "std not supported for string arrays".into(),
            ));
        }
        if self.dtype() == DType::Object {
            let var = self.var(axis, ddof, false)?;
            let result = sqrt_boxed_object_array(var)?;
            return Ok(maybe_keepdims(result, axis, keepdims, self.ndim()));
        }
        if self.dtype().is_complex() {
            return Err(NumpyError::TypeError(
                "std not supported for complex arrays (use abs() first)".into(),
            ));
        }
        let var = self.var(axis, ddof, false)?;
        let result = var.sqrt();
        Ok(maybe_keepdims(result, axis, keepdims, self.ndim()))
    }

    /// Variance. Always returns Float64.
    pub fn var(&self, axis: Option<usize>, ddof: usize, keepdims: bool) -> Result<NdArray> {
        if self.dtype().is_string() {
            return Err(NumpyError::TypeError(
                "var not supported for string arrays".into(),
            ));
        }
        if self.dtype() == DType::Object {
            let mean = self.mean(axis, false)?;
            let centered = (self - &mean)?;
            let magnitude = centered.abs();
            let squared = (&magnitude * &magnitude)?;
            let result = squared.mean(axis, false)?;
            if ddof > 0 {
                let n = match axis {
                    None => self.size(),
                    Some(ax) => {
                        validate_axis(ax, self.ndim())?;
                        self.shape()[ax]
                    }
                };
                if ddof >= n {
                    let nan = NdArray::from_boxed_scalars(
                        vec![BoxedScalar::Object(BoxedObjectScalar::Float(f64::NAN))],
                        result.shape(),
                        DType::Object,
                    )?;
                    return Ok(maybe_keepdims(nan, axis, keepdims, self.ndim()));
                }
                let correction = NdArray::from_boxed_scalars(
                    vec![BoxedScalar::Object(BoxedObjectScalar::Float(
                        n as f64 / (n - ddof) as f64,
                    ))],
                    &[],
                    DType::Object,
                )?;
                let corrected = (&result * &correction)?;
                return Ok(maybe_keepdims(corrected, axis, keepdims, self.ndim()));
            }
            return Ok(maybe_keepdims(result, axis, keepdims, self.ndim()));
        }
        if self.dtype().is_complex() {
            return Err(NumpyError::TypeError(
                "var not supported for complex arrays (use abs() first)".into(),
            ));
        }
        // var = mean(x^2) - mean(x)^2
        let float_self = self.astype(DType::Float64);
        let x_sq = (&float_self * &float_self)?;
        let mean_x_sq = x_sq.mean(axis, false)?;
        let mean_x = float_self.mean(axis, false)?;
        let mean_x_squared = (&mean_x * &mean_x)?;
        let result = (&mean_x_sq - &mean_x_squared)?;
        if ddof > 0 {
            let n = match axis {
                None => self.size(),
                Some(ax) => {
                    validate_axis(ax, self.ndim())?;
                    self.shape()[ax]
                }
            };
            if ddof >= n {
                let nan_val = NdArray::full_f64(result.shape(), f64::NAN);
                return Ok(maybe_keepdims(nan_val, axis, keepdims, self.ndim()));
            }
            let correction = NdArray::full_f64(result.shape(), n as f64 / (n - ddof) as f64);
            let corrected = (&result * &correction)?;
            return Ok(maybe_keepdims(corrected, axis, keepdims, self.ndim()));
        }
        Ok(maybe_keepdims(result, axis, keepdims, self.ndim()))
    }

    /// Index of minimum element.
    /// axis=None: flatten then find argmin (scalar Int64). axis=Some(ax): argmin along that axis.
    pub fn argmin(&self, axis: Option<usize>) -> Result<NdArray> {
        if self.dtype().is_string() {
            return Err(NumpyError::TypeError(
                "argmin not supported for string arrays".into(),
            ));
        }
        if self.dtype().is_complex() {
            return Err(NumpyError::TypeError(
                "argmin not supported for complex arrays".into(),
            ));
        }
        match axis {
            None => {
                let idx = self.reduce_all_argmin()?;
                Ok(NdArray::from_data(ArrayData::Int64(
                    ArrayD::from_elem(IxDyn(&[]), idx as i64).into_shared(),
                )))
            }
            Some(ax) => self.reduce_axis_argmin(ax),
        }
    }

    /// Index of maximum element.
    /// axis=None: flatten then find argmax (scalar Int64). axis=Some(ax): argmax along that axis.
    pub fn argmax(&self, axis: Option<usize>) -> Result<NdArray> {
        if self.dtype().is_string() {
            return Err(NumpyError::TypeError(
                "argmax not supported for string arrays".into(),
            ));
        }
        if self.dtype().is_complex() {
            return Err(NumpyError::TypeError(
                "argmax not supported for complex arrays".into(),
            ));
        }
        match axis {
            None => {
                let idx = self.reduce_all_argmax()?;
                Ok(NdArray::from_data(ArrayData::Int64(
                    ArrayD::from_elem(IxDyn(&[]), idx as i64).into_shared(),
                )))
            }
            Some(ax) => self.reduce_axis_argmax(ax),
        }
    }

    fn all_scalar(&self) -> bool {
        if self.dtype().is_boxed() {
            return iter_boxed_coords(self.shape()).all(|coord| {
                boxed_truthy(&self.get_boxed(&coord).expect("boxed reduction index"))
            });
        }
        let descriptor = descriptor_for_dtype(self.dtype());
        let kernel = descriptor
            .truth_reduce_kernel(TruthReduceKernelOp::AllTruthy)
            .unwrap_or_else(|| {
                panic!("truth reduction kernel not registered for {}", self.dtype())
            });
        kernel(&self.data()).expect("truth reduction kernel dtype mismatch")
    }

    fn any_scalar(&self) -> bool {
        if self.dtype().is_boxed() {
            return iter_boxed_coords(self.shape()).any(|coord| {
                boxed_truthy(&self.get_boxed(&coord).expect("boxed reduction index"))
            });
        }
        let descriptor = descriptor_for_dtype(self.dtype());
        let kernel = descriptor
            .truth_reduce_kernel(TruthReduceKernelOp::AnyTruthy)
            .unwrap_or_else(|| {
                panic!("truth reduction kernel not registered for {}", self.dtype())
            });
        kernel(&self.data()).expect("truth reduction kernel dtype mismatch")
    }

    pub fn all(&self) -> bool {
        self.all_scalar()
    }

    pub fn any(&self) -> bool {
        self.any_scalar()
    }

    pub fn all_reduce(&self, axis: Option<usize>, keepdims: bool) -> Result<NdArray> {
        match axis {
            None => Ok(maybe_keepdims(
                bool_scalar_array(self.all_scalar()),
                None,
                keepdims,
                self.ndim(),
            )),
            Some(axis) => self.truth_reduce_axis(axis, keepdims, true),
        }
    }

    pub fn any_reduce(&self, axis: Option<usize>, keepdims: bool) -> Result<NdArray> {
        match axis {
            None => Ok(maybe_keepdims(
                bool_scalar_array(self.any_scalar()),
                None,
                keepdims,
                self.ndim(),
            )),
            Some(axis) => self.truth_reduce_axis(axis, keepdims, false),
        }
    }

    // --- Internal helpers ---

    fn truth_reduce_axis(&self, axis: usize, keepdims: bool, reduce_all: bool) -> Result<NdArray> {
        validate_axis(axis, self.ndim())?;
        let result = if self.dtype().is_boxed() {
            let mut out_shape = self.shape().to_vec();
            out_shape.remove(axis);
            let mut out = Vec::with_capacity(out_shape.iter().product());

            for out_coord in iter_boxed_coords(&out_shape) {
                let mut full_coord = vec![0usize; self.ndim()];
                let mut out_axis = 0usize;
                for (axis_idx, slot) in full_coord.iter_mut().enumerate().take(self.ndim()) {
                    if axis_idx == axis {
                        continue;
                    }
                    *slot = out_coord[out_axis];
                    out_axis += 1;
                }

                let mut acc = reduce_all;
                for reduce_index in 0..self.shape()[axis] {
                    full_coord[axis] = reduce_index;
                    let truth =
                        boxed_truthy(&self.get_boxed(&full_coord).expect("boxed reduction index"));
                    if reduce_all {
                        acc &= truth;
                    } else {
                        acc |= truth;
                    }
                }
                out.push(acc);
            }

            NdArray::from_data(ArrayData::Bool(
                ArrayD::from_shape_vec(IxDyn(&out_shape), out)
                    .expect("truth reduce axis shape must match")
                    .into_shared(),
            ))
        } else {
            let descriptor = descriptor_for_dtype(self.dtype());
            let kernel = descriptor
                .truth_kernel(crate::kernel::TruthKernelOp::ToBool)
                .ok_or_else(|| NumpyError::TypeError("truth kernel not registered".into()))?;
            let ArrayData::Bool(values) = kernel(self.data())? else {
                unreachable!("truth kernel must produce bool arrays");
            };
            let reduced = values.map_axis(Axis(axis), |lane| {
                if reduce_all {
                    lane.iter().all(|&value| value)
                } else {
                    lane.iter().any(|&value| value)
                }
            });
            NdArray::from_data(ArrayData::Bool(reduced.into_dyn().into_shared()))
        };
        Ok(maybe_keepdims(result, Some(axis), keepdims, self.ndim()))
    }

    fn prepare_sum_reduction(&self) -> Result<(ArrayData, DType)> {
        let plan = self.descriptor().reduction_plan(ReductionOp::Sum)?;
        Ok((
            self.cast_for_execution(plan.input_cast()),
            plan.result_dtype(),
        ))
    }

    fn reduce_all_sum(&self) -> Result<NdArray> {
        let (data, result_dtype) = self.prepare_sum_reduction()?;
        let descriptor = descriptor_for_dtype(result_dtype);
        let kernel = descriptor
            .reduction_all_kernel(ReductionKernelOp::Sum)
            .ok_or_else(|| NumpyError::TypeError("sum kernel not registered".into()))?;
        Ok(NdArray::from_data(kernel(data)?))
    }

    fn reduce_axis_sum(&self, axis: usize) -> Result<NdArray> {
        validate_axis(axis, self.ndim())?;
        let (data, result_dtype) = self.prepare_sum_reduction()?;
        let descriptor = descriptor_for_dtype(result_dtype);
        let kernel = descriptor
            .reduction_axis_kernel(ReductionKernelOp::Sum)
            .ok_or_else(|| NumpyError::TypeError("sum kernel not registered".into()))?;
        Ok(NdArray::from_data(kernel(data, axis)?))
    }

    fn prepare_prod_reduction(&self) -> Result<(ArrayData, DType)> {
        let plan = self.descriptor().reduction_plan(ReductionOp::Prod)?;
        Ok((
            self.cast_for_execution(plan.input_cast()),
            plan.result_dtype(),
        ))
    }

    fn reduce_all_prod(&self) -> Result<NdArray> {
        let (data, result_dtype) = self.prepare_prod_reduction()?;
        let descriptor = descriptor_for_dtype(result_dtype);
        let kernel = descriptor
            .reduction_all_kernel(ReductionKernelOp::Prod)
            .ok_or_else(|| NumpyError::TypeError("prod kernel not registered".into()))?;
        Ok(NdArray::from_data(kernel(data)?))
    }

    fn reduce_axis_prod(&self, axis: usize) -> Result<NdArray> {
        validate_axis(axis, self.ndim())?;
        let (data, result_dtype) = self.prepare_prod_reduction()?;
        let descriptor = descriptor_for_dtype(result_dtype);
        let kernel = descriptor
            .reduction_axis_kernel(ReductionKernelOp::Prod)
            .ok_or_else(|| NumpyError::TypeError("prod kernel not registered".into()))?;
        Ok(NdArray::from_data(kernel(data, axis)?))
    }

    fn reduce_all_min(&self) -> Result<NdArray> {
        self.execute_extrema_reduction_all(ReductionOp::Min, ReductionKernelOp::Min)
    }

    fn reduce_all_max(&self) -> Result<NdArray> {
        self.execute_extrema_reduction_all(ReductionOp::Max, ReductionKernelOp::Max)
    }

    fn reduce_all_argmin(&self) -> Result<usize> {
        self.execute_arg_reduction_all(ReductionOp::ArgMin, ArgReductionKernelOp::ArgMin)
    }

    fn reduce_all_argmax(&self) -> Result<usize> {
        self.execute_arg_reduction_all(ReductionOp::ArgMax, ArgReductionKernelOp::ArgMax)
    }

    fn reduce_axis_argmin(&self, axis: usize) -> Result<NdArray> {
        self.execute_arg_reduction_axis(axis, ReductionOp::ArgMin, ArgReductionKernelOp::ArgMin)
    }

    fn reduce_axis_argmax(&self, axis: usize) -> Result<NdArray> {
        self.execute_arg_reduction_axis(axis, ReductionOp::ArgMax, ArgReductionKernelOp::ArgMax)
    }

    fn reduce_axis_fold(&self, axis: usize, op: ReduceOp) -> Result<NdArray> {
        match op {
            ReduceOp::Min => {
                self.execute_extrema_reduction_axis(axis, ReductionOp::Min, ReductionKernelOp::Min)
            }
            ReduceOp::Max => {
                self.execute_extrema_reduction_axis(axis, ReductionOp::Max, ReductionKernelOp::Max)
            }
        }
    }

    fn execute_extrema_reduction_all(
        &self,
        plan_op: ReductionOp,
        kernel_op: ReductionKernelOp,
    ) -> Result<NdArray> {
        if self.dtype().is_boxed() {
            let reduce_max = matches!(plan_op, ReductionOp::Max);
            let mut coords = iter_boxed_coords(self.shape());
            let first_coord = coords.next().ok_or_else(empty_extrema_error)?;
            let first = self.get_boxed(&first_coord)?;
            let acc = coords.try_fold(first, |acc, coord| {
                choose_boxed_extrema(acc, self.get_boxed(&coord)?, reduce_max)
            })?;
            return NdArray::from_boxed_scalars(vec![acc], &[], self.dtype());
        }
        let plan = self.descriptor().reduction_plan(plan_op)?;
        let data = self.cast_for_execution(plan.input_cast());
        let descriptor = descriptor_for_dtype(plan.result_dtype());
        let kernel = descriptor
            .reduction_all_kernel(kernel_op)
            .ok_or_else(|| NumpyError::TypeError("reduction kernel not registered".into()))?;
        let mut result = NdArray::from_data(kernel(data)?);
        if plan.result_dtype().is_narrow() {
            result.set_declared_dtype(plan.result_dtype());
        }
        Ok(result)
    }

    fn execute_extrema_reduction_axis(
        &self,
        axis: usize,
        plan_op: ReductionOp,
        kernel_op: ReductionKernelOp,
    ) -> Result<NdArray> {
        validate_axis(axis, self.ndim())?;
        if self.dtype().is_boxed() {
            if self.shape()[axis] == 0 {
                return Err(empty_extrema_error());
            }
            let reduce_max = matches!(plan_op, ReductionOp::Max);
            let mut out_shape = self.shape().to_vec();
            out_shape.remove(axis);
            let mut out = Vec::with_capacity(out_shape.iter().product());

            for out_coord in iter_boxed_coords(&out_shape) {
                let mut full_coord = vec![0usize; self.ndim()];
                let mut out_axis = 0usize;
                for (axis_idx, slot) in full_coord.iter_mut().enumerate().take(self.ndim()) {
                    if axis_idx == axis {
                        continue;
                    }
                    *slot = out_coord[out_axis];
                    out_axis += 1;
                }

                full_coord[axis] = 0;
                let mut acc = self.get_boxed(&full_coord)?;
                for reduce_index in 1..self.shape()[axis] {
                    full_coord[axis] = reduce_index;
                    acc = choose_boxed_extrema(acc, self.get_boxed(&full_coord)?, reduce_max)?;
                }
                out.push(acc);
            }

            return NdArray::from_boxed_scalars(out, &out_shape, self.dtype());
        }
        let plan = self.descriptor().reduction_plan(plan_op)?;
        let data = self.cast_for_execution(plan.input_cast());
        let descriptor = descriptor_for_dtype(plan.result_dtype());
        let kernel = descriptor
            .reduction_axis_kernel(kernel_op)
            .ok_or_else(|| NumpyError::TypeError("reduction kernel not registered".into()))?;
        let mut result = NdArray::from_data(kernel(data, axis)?);
        if plan.result_dtype().is_narrow() {
            result.set_declared_dtype(plan.result_dtype());
        }
        Ok(result)
    }

    fn execute_arg_reduction_all(
        &self,
        plan_op: ReductionOp,
        kernel_op: ArgReductionKernelOp,
    ) -> Result<usize> {
        let plan = self.descriptor().reduction_plan(plan_op)?;
        let data = self.cast_for_execution(plan.input_cast());
        let descriptor = descriptor_for_dtype(plan.input_cast().target_dtype());
        let kernel = descriptor
            .arg_reduction_all_kernel(kernel_op)
            .ok_or_else(|| NumpyError::TypeError("arg reduction kernel not registered".into()))?;
        kernel(data)
    }

    fn execute_arg_reduction_axis(
        &self,
        axis: usize,
        plan_op: ReductionOp,
        kernel_op: ArgReductionKernelOp,
    ) -> Result<NdArray> {
        validate_axis(axis, self.ndim())?;
        let plan = self.descriptor().reduction_plan(plan_op)?;
        let data = self.cast_for_execution(plan.input_cast());
        let descriptor = descriptor_for_dtype(plan.input_cast().target_dtype());
        let kernel = descriptor
            .arg_reduction_axis_kernel(kernel_op)
            .ok_or_else(|| NumpyError::TypeError("arg reduction kernel not registered".into()))?;
        Ok(NdArray::from_data(kernel(data, axis)?))
    }
}

impl NdArray {
    fn reduce_all_boxed_object(
        &self,
        op: BinaryOp,
        identity: BoxedObjectScalar,
    ) -> Result<NdArray> {
        let storage = self.storage().boxed_storage().ok_or_else(|| {
            NumpyError::TypeError("boxed object reduction requires boxed storage".into())
        })?;
        let elements = storage.elements()?;
        let reduced = reduce_boxed_object_elements(elements, op, identity)?;
        NdArray::from_boxed_scalars(vec![reduced], &[], DType::Object)
    }

    fn reduce_axis_boxed_object(
        &self,
        axis: usize,
        op: BinaryOp,
        identity: BoxedObjectScalar,
    ) -> Result<NdArray> {
        validate_axis(axis, self.ndim())?;
        let shape = self.shape();
        let mut out_shape = shape.to_vec();
        out_shape.remove(axis);
        let axis_len = shape[axis];
        let mut out = Vec::with_capacity(out_shape.iter().product::<usize>().max(1));

        for out_coord in iter_boxed_coords(&out_shape) {
            let mut lane = Vec::with_capacity(axis_len);
            for axis_idx in 0..axis_len {
                let full_coord = expand_axis_coord(&out_coord, axis, axis_idx, self.ndim());
                lane.push(self.get_boxed(&full_coord)?);
            }
            out.push(reduce_boxed_object_elements(lane, op, identity.clone())?);
        }

        NdArray::from_boxed_scalars(out, &out_shape, DType::Object)
    }
}

fn expand_axis_coord(base: &[usize], axis: usize, axis_idx: usize, ndim: usize) -> Vec<usize> {
    let mut full = Vec::with_capacity(ndim);
    let mut base_i = 0usize;
    for dim in 0..ndim {
        if dim == axis {
            full.push(axis_idx);
        } else {
            full.push(base[base_i]);
            base_i += 1;
        }
    }
    full
}

fn reduce_boxed_object_elements(
    elements: Vec<BoxedScalar>,
    op: BinaryOp,
    identity: BoxedObjectScalar,
) -> Result<BoxedScalar> {
    let mut iter = elements.into_iter();
    let Some(first) = iter.next() else {
        return Ok(BoxedScalar::Object(identity));
    };
    iter.try_fold(first, |acc, value| {
        boxed_object_binary_reduce(acc, value, op)
    })
}

fn boxed_object_binary_reduce(
    lhs: BoxedScalar,
    rhs: BoxedScalar,
    op: BinaryOp,
) -> Result<BoxedScalar> {
    match op {
        BinaryOp::Add => crate::ops::arithmetic::apply_boxed_binary(op, lhs, rhs),
        BinaryOp::Mul => crate::ops::arithmetic::apply_boxed_binary(op, lhs, rhs),
        _ => Err(NumpyError::TypeError(
            "unsupported boxed object reduction op".into(),
        )),
    }
}

fn sqrt_boxed_object_array(array: NdArray) -> Result<NdArray> {
    let storage = array
        .storage()
        .boxed_storage()
        .ok_or_else(|| NumpyError::TypeError("boxed object std requires boxed storage".into()))?;
    let elements = storage
        .elements()?
        .into_iter()
        .map(|value| match value {
            BoxedScalar::Object(BoxedObjectScalar::Bool(v)) => {
                Ok(BoxedScalar::Object(BoxedObjectScalar::Float(if v {
                    1.0
                } else {
                    0.0
                })))
            }
            BoxedScalar::Object(BoxedObjectScalar::Int(v)) => Ok(BoxedScalar::Object(
                BoxedObjectScalar::Float((v as f64).sqrt()),
            )),
            BoxedScalar::Object(BoxedObjectScalar::Float(v)) => {
                Ok(BoxedScalar::Object(BoxedObjectScalar::Float(v.sqrt())))
            }
            BoxedScalar::Object(BoxedObjectScalar::Complex(v)) => {
                Ok(BoxedScalar::Object(BoxedObjectScalar::Complex(v.sqrt())))
            }
            BoxedScalar::Object(BoxedObjectScalar::Text(_)) => Err(NumpyError::TypeError(
                "std/var not supported for string object scalars".into(),
            )),
            _ => Err(NumpyError::TypeError(
                "std/var boxed sqrt requires object boxed scalars".into(),
            )),
        })
        .collect::<Result<Vec<_>>>()?;
    NdArray::from_boxed_scalars(elements, array.shape(), DType::Object)
}

fn boxed_truthy(value: &BoxedScalar) -> bool {
    match value {
        BoxedScalar::Object(object) => match object {
            BoxedObjectScalar::Bool(v) => *v,
            BoxedObjectScalar::Int(v) => *v != 0,
            BoxedObjectScalar::Float(v) => *v != 0.0,
            BoxedObjectScalar::Complex(v) => v.re != 0.0 || v.im != 0.0,
            BoxedObjectScalar::Text(v) => !v.is_empty(),
        },
        BoxedScalar::Datetime(value) | BoxedScalar::Timedelta(value) => !value.is_nat,
    }
}

fn bool_scalar_array(value: bool) -> NdArray {
    NdArray::from_data(ArrayData::Bool(
        ArrayD::from_elem(IxDyn(&[]), value).into_shared(),
    ))
}

#[derive(Clone, Copy)]
enum ReduceOp {
    Min,
    Max,
}

#[cfg(test)]
mod tests {
    use crate::{BoxedObjectScalar, BoxedScalar, DType, NdArray};

    #[test]
    fn test_sum_all() {
        let a = NdArray::from_vec(vec![1.0_f64, 2.0, 3.0]);
        let s = a.sum(None, false).unwrap();
        assert_eq!(s.size(), 1);
        assert_eq!(s.shape(), &[]);
        assert_eq!(s.dtype(), DType::Float64);
    }

    #[test]
    fn test_sum_axis() {
        let a = NdArray::ones(&[3, 4], DType::Float64);
        let s = a.sum(Some(0), false).unwrap();
        assert_eq!(s.shape(), &[4]);
    }

    #[test]
    fn test_sum_axis_1() {
        let a = NdArray::ones(&[3, 4], DType::Float64);
        let s = a.sum(Some(1), false).unwrap();
        assert_eq!(s.shape(), &[3]);
    }

    #[test]
    fn test_sum_invalid_axis() {
        let a = NdArray::from_vec(vec![1.0_f64, 2.0]);
        assert!(a.sum(Some(5), false).is_err());
    }

    #[test]
    fn test_mean() {
        let a = NdArray::from_vec(vec![1.0_f64, 2.0, 3.0]);
        let m = a.mean(None, false).unwrap();
        assert_eq!(m.size(), 1);
        assert_eq!(m.dtype(), DType::Float64);
    }

    #[test]
    fn test_mean_axis() {
        let a = NdArray::ones(&[3, 4], DType::Float64);
        let m = a.mean(Some(0), false).unwrap();
        assert_eq!(m.shape(), &[4]);
    }

    #[test]
    fn test_min_max_all() {
        let a = NdArray::from_vec(vec![3.0_f64, 1.0, 2.0]);
        let mn = a.min(None, false).unwrap();
        let mx = a.max(None, false).unwrap();
        assert_eq!(mn.size(), 1);
        assert_eq!(mx.size(), 1);
    }

    #[test]
    fn test_min_axis() {
        let a = NdArray::ones(&[3, 4], DType::Int32);
        let mn = a.min(Some(0), false).unwrap();
        assert_eq!(mn.shape(), &[4]);
    }

    #[test]
    fn test_boxed_max_all() {
        use crate::array::{BoxedScalar, BoxedTemporalScalar};
        let a = NdArray::from_boxed_scalars(
            vec![
                BoxedScalar::Datetime(BoxedTemporalScalar {
                    value: 3,
                    unit: "D".into(),
                    is_nat: false,
                }),
                BoxedScalar::Datetime(BoxedTemporalScalar {
                    value: 5,
                    unit: "D".into(),
                    is_nat: false,
                }),
                BoxedScalar::Datetime(BoxedTemporalScalar {
                    value: 1,
                    unit: "D".into(),
                    is_nat: false,
                }),
            ],
            &[3],
            DType::Datetime64,
        )
        .unwrap();
        let mx = a.max(None, false).unwrap();
        assert_eq!(
            mx.get_boxed(&[]).unwrap(),
            BoxedScalar::Datetime(BoxedTemporalScalar {
                value: 5,
                unit: "D".into(),
                is_nat: false,
            })
        );
    }

    #[test]
    fn test_boxed_min_axis_nat_propagates() {
        use crate::array::{BoxedScalar, BoxedTemporalScalar};
        let nat = BoxedScalar::Datetime(BoxedTemporalScalar {
            value: 0,
            unit: "D".into(),
            is_nat: true,
        });
        let a = NdArray::from_boxed_scalars(
            vec![
                BoxedScalar::Datetime(BoxedTemporalScalar {
                    value: 3,
                    unit: "D".into(),
                    is_nat: false,
                }),
                nat.clone(),
                BoxedScalar::Datetime(BoxedTemporalScalar {
                    value: 5,
                    unit: "D".into(),
                    is_nat: false,
                }),
                BoxedScalar::Datetime(BoxedTemporalScalar {
                    value: 7,
                    unit: "D".into(),
                    is_nat: false,
                }),
            ],
            &[2, 2],
            DType::Datetime64,
        )
        .unwrap();
        let mn = a.min(Some(1), false).unwrap();
        assert_eq!(mn.shape(), &[2]);
        assert_eq!(mn.get_boxed(&[0]).unwrap(), nat);
    }

    #[test]
    fn test_var_std() {
        let a = NdArray::from_vec(vec![1.0_f64, 2.0, 3.0]);
        let v = a.var(None, 0, false).unwrap();
        let s = a.std(None, 0, false).unwrap();
        assert_eq!(v.dtype(), DType::Float64);
        assert_eq!(s.dtype(), DType::Float64);
    }

    #[test]
    fn test_argmin_argmax() {
        let a = NdArray::from_vec(vec![3.0_f64, 1.0, 2.0]);
        let mn = a.argmin(None).unwrap();
        assert_eq!(mn.shape(), &[]);
        let mx = a.argmax(None).unwrap();
        assert_eq!(mx.shape(), &[]);
    }

    #[test]
    fn test_argmin_axis() {
        let a = NdArray::from_vec(vec![3.0_f64, 1.0, 2.0, 6.0, 4.0, 5.0])
            .reshape(&[2, 3])
            .unwrap();
        let idx = a.argmin(Some(1)).unwrap();
        assert_eq!(idx.shape(), &[2]);
    }

    #[test]
    fn test_argmax_axis() {
        let a = NdArray::from_vec(vec![3.0_f64, 1.0, 2.0, 6.0, 4.0, 5.0])
            .reshape(&[2, 3])
            .unwrap();
        let idx = a.argmax(Some(0)).unwrap();
        assert_eq!(idx.shape(), &[3]);
    }

    #[test]
    fn test_all_any() {
        let all_true = NdArray::from_vec(vec![true, true, true]);
        let some_false = NdArray::from_vec(vec![true, false, true]);
        let all_false = NdArray::from_vec(vec![false, false, false]);

        assert!(all_true.all());
        assert!(!some_false.all());
        assert!(some_false.any());
        assert!(!all_false.any());
    }

    #[test]
    fn test_all_any_numeric() {
        let a = NdArray::from_vec(vec![1_i32, 2, 3]);
        assert!(a.all());
        assert!(a.any());

        let b = NdArray::from_vec(vec![0_i32, 0, 0]);
        assert!(!b.all());
        assert!(!b.any());
    }

    #[test]
    fn test_sum_bool() {
        let a = NdArray::from_vec(vec![true, false, true]);
        let s = a.sum(None, false).unwrap();
        assert_eq!(s.dtype(), DType::Int32);
    }

    #[test]
    fn test_prod_all() {
        let a = NdArray::from_vec(vec![1.0_f64, 2.0, 3.0, 4.0]);
        let p = a.prod(None, false).unwrap();
        assert_eq!(p.shape(), &[]);
    }

    #[test]
    fn test_prod_axis() {
        let a = NdArray::ones(&[2, 3], DType::Float64);
        let p = a.prod(Some(0), false).unwrap();
        assert_eq!(p.shape(), &[3]);
    }

    #[test]
    fn test_prod_keepdims() {
        let a = NdArray::ones(&[2, 3], DType::Float64);
        let p = a.prod(Some(1), true).unwrap();
        assert_eq!(p.shape(), &[2, 1]);
    }

    #[test]
    fn test_sum_keepdims() {
        let a = NdArray::ones(&[3, 4], DType::Float64);
        let s = a.sum(Some(0), true).unwrap();
        assert_eq!(s.shape(), &[1, 4]);
    }

    #[test]
    fn test_mean_keepdims() {
        let a = NdArray::ones(&[3, 4], DType::Float64);
        let m = a.mean(Some(1), true).unwrap();
        assert_eq!(m.shape(), &[3, 1]);
    }

    #[test]
    fn test_object_sum_prod_mean() {
        let a = NdArray::from_boxed_scalars(
            vec![
                BoxedScalar::Object(BoxedObjectScalar::Int(1)),
                BoxedScalar::Object(BoxedObjectScalar::Int(2)),
                BoxedScalar::Object(BoxedObjectScalar::Int(3)),
            ],
            &[3],
            DType::Object,
        )
        .unwrap();
        assert_eq!(
            a.sum(None, false).unwrap().get_boxed(&[]).unwrap(),
            BoxedScalar::Object(BoxedObjectScalar::Int(6))
        );
        assert_eq!(
            a.prod(None, false).unwrap().get_boxed(&[]).unwrap(),
            BoxedScalar::Object(BoxedObjectScalar::Int(6))
        );
        assert_eq!(
            a.mean(None, false).unwrap().get_boxed(&[]).unwrap(),
            BoxedScalar::Object(BoxedObjectScalar::Float(2.0))
        );
        assert_eq!(
            a.var(None, 0, false).unwrap().get_boxed(&[]).unwrap(),
            BoxedScalar::Object(BoxedObjectScalar::Float(2.0 / 3.0))
        );
    }

    #[test]
    fn test_var_ddof() {
        let a = NdArray::from_vec(vec![1.0_f64, 2.0, 3.0]);
        let v0 = a.var(None, 0, false).unwrap();
        let v1 = a.var(None, 1, false).unwrap();
        assert_eq!(v0.dtype(), DType::Float64);
        assert_eq!(v1.dtype(), DType::Float64);
    }
}
