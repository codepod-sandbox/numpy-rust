use crate::array_data::ArrayD;
use ndarray::IxDyn;

use crate::array_data::ArrayData;
use crate::descriptor::descriptor_for_dtype;
use crate::dtype::DType;
use crate::error::{NumpyError, Result};
use crate::kernel::{ArgReductionKernelOp, ReductionKernelOp, TruthReduceKernelOp};
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

impl NdArray {
    /// Sum of array elements over a given axis, or all elements if axis is None.
    pub fn sum(&self, axis: Option<usize>, keepdims: bool) -> Result<NdArray> {
        if self.dtype().is_string() {
            return Err(NumpyError::TypeError(
                "sum not supported for string arrays".into(),
            ));
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

    /// True if all elements are truthy.
    pub fn all(&self) -> bool {
        let descriptor = descriptor_for_dtype(self.dtype());
        let kernel = descriptor
            .truth_reduce_kernel(TruthReduceKernelOp::AllTruthy)
            .unwrap_or_else(|| {
                panic!("truth reduction kernel not registered for {}", self.dtype())
            });
        kernel(self.data()).expect("truth reduction kernel dtype mismatch")
    }

    /// True if any element is truthy.
    pub fn any(&self) -> bool {
        let descriptor = descriptor_for_dtype(self.dtype());
        let kernel = descriptor
            .truth_reduce_kernel(TruthReduceKernelOp::AnyTruthy)
            .unwrap_or_else(|| {
                panic!("truth reduction kernel not registered for {}", self.dtype())
            });
        kernel(self.data()).expect("truth reduction kernel dtype mismatch")
    }

    // --- Internal helpers ---

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

#[derive(Clone, Copy)]
enum ReduceOp {
    Min,
    Max,
}

#[cfg(test)]
mod tests {
    use crate::{DType, NdArray};

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
    fn test_var_ddof() {
        let a = NdArray::from_vec(vec![1.0_f64, 2.0, 3.0]);
        let v0 = a.var(None, 0, false).unwrap();
        let v1 = a.var(None, 1, false).unwrap();
        assert_eq!(v0.dtype(), DType::Float64);
        assert_eq!(v1.dtype(), DType::Float64);
    }
}
