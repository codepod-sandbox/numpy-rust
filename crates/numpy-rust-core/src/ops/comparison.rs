use crate::array_data::ArrayData;
use crate::broadcasting::{broadcast_array_data, broadcast_shape};
use crate::descriptor::descriptor_for_dtype;
use crate::error::{NumpyError, Result};
use crate::kernel::ComparisonKernelOp;
use crate::resolver::{resolve_comparison_op, ComparisonOp, ComparisonOpPlan};
use crate::NdArray;

fn comparison_kernel_op(op: ComparisonOp) -> ComparisonKernelOp {
    match op {
        ComparisonOp::Eq => ComparisonKernelOp::Eq,
        ComparisonOp::Ne => ComparisonKernelOp::Ne,
        ComparisonOp::Lt => ComparisonKernelOp::Lt,
        ComparisonOp::Le => ComparisonKernelOp::Le,
        ComparisonOp::Gt => ComparisonKernelOp::Gt,
        ComparisonOp::Ge => ComparisonKernelOp::Ge,
    }
}

fn prepare_comparison_execution(
    lhs: &NdArray,
    rhs: &NdArray,
    op: ComparisonOp,
) -> Result<(ArrayData, ArrayData, ComparisonOpPlan)> {
    let plan = resolve_comparison_op(op, lhs.dtype(), rhs.dtype())?;
    let out_shape = broadcast_shape(lhs.shape(), rhs.shape())?;

    let lhs = broadcast_array_data(&lhs.cast_for_execution(plan.lhs_cast()), &out_shape);
    let rhs = broadcast_array_data(&rhs.cast_for_execution(plan.rhs_cast()), &out_shape);

    Ok((lhs, rhs, plan))
}

fn execute_comparison(lhs: &NdArray, rhs: &NdArray, op: ComparisonOp) -> Result<NdArray> {
    let (lhs, rhs, plan) = prepare_comparison_execution(lhs, rhs, op)?;
    let descriptor = descriptor_for_dtype(plan.execution_dtype());
    let kernel = descriptor
        .comparison_kernel(comparison_kernel_op(op))
        .ok_or_else(|| NumpyError::TypeError("comparison kernel not registered".into()))?;
    Ok(NdArray::from_data(kernel(lhs, rhs)?))
}

impl NdArray {
    pub fn eq(&self, other: &NdArray) -> Result<NdArray> {
        execute_comparison(self, other, ComparisonOp::Eq)
    }

    pub fn ne(&self, other: &NdArray) -> Result<NdArray> {
        execute_comparison(self, other, ComparisonOp::Ne)
    }

    pub fn lt(&self, other: &NdArray) -> Result<NdArray> {
        execute_comparison(self, other, ComparisonOp::Lt)
    }

    pub fn le(&self, other: &NdArray) -> Result<NdArray> {
        execute_comparison(self, other, ComparisonOp::Le)
    }

    pub fn gt(&self, other: &NdArray) -> Result<NdArray> {
        execute_comparison(self, other, ComparisonOp::Gt)
    }

    pub fn ge(&self, other: &NdArray) -> Result<NdArray> {
        execute_comparison(self, other, ComparisonOp::Ge)
    }
}

pub fn complex_cmp<T: num_traits::Float>(
    a: &num_complex::Complex<T>,
    b: &num_complex::Complex<T>,
) -> std::cmp::Ordering {
    crate::kernel::complex_cmp(a, b)
}

#[cfg(test)]
mod tests {
    use crate::resolver::{resolve_comparison_op, ComparisonOp};
    use crate::{DType, NdArray};
    use num_complex::Complex;

    #[test]
    fn test_eq_same_shape() {
        let a = NdArray::from_vec(vec![1.0_f64, 2.0, 3.0]);
        let b = NdArray::from_vec(vec![1.0_f64, 9.0, 3.0]);
        let c = a.eq(&b).unwrap();
        assert_eq!(c.dtype(), DType::Bool);
        assert_eq!(c.shape(), &[3]);
    }

    #[test]
    fn test_ne() {
        let a = NdArray::from_vec(vec![1.0_f64, 2.0]);
        let b = NdArray::from_vec(vec![1.0_f64, 9.0]);
        let c = a.ne(&b).unwrap();
        assert_eq!(c.dtype(), DType::Bool);
    }

    #[test]
    fn test_lt() {
        let a = NdArray::from_vec(vec![1.0_f64, 5.0]);
        let b = NdArray::from_vec(vec![2.0_f64, 3.0]);
        let c = a.lt(&b).unwrap();
        assert_eq!(c.dtype(), DType::Bool);
    }

    #[test]
    fn test_gt_broadcast() {
        let a = NdArray::zeros(&[3, 4], DType::Float64);
        let b = NdArray::ones(&[4], DType::Float64);
        let c = a.gt(&b).unwrap();
        assert_eq!(c.shape(), &[3, 4]);
        assert_eq!(c.dtype(), DType::Bool);
    }

    #[test]
    fn test_le_cross_dtype() {
        let a = NdArray::from_vec(vec![1_i32, 2, 3]);
        let b = NdArray::from_vec(vec![1.5_f64, 2.5, 2.5]);
        let c = a.le(&b).unwrap();
        assert_eq!(c.dtype(), DType::Bool);
    }

    #[test]
    fn test_ge() {
        let a = NdArray::from_vec(vec![3_i32, 1, 2]);
        let b = NdArray::from_vec(vec![2_i32, 2, 2]);
        let c = a.ge(&b).unwrap();
        assert_eq!(c.dtype(), DType::Bool);
    }

    #[test]
    fn test_cmp_incompatible_fails() {
        let a = NdArray::zeros(&[3], DType::Float64);
        let b = NdArray::zeros(&[4], DType::Float64);
        assert!(a.eq(&b).is_err());
    }

    #[test]
    fn test_eq_complex() {
        let a = NdArray::from_vec(vec![Complex::new(1.0f64, 2.0), Complex::new(3.0, 4.0)]);
        let b = NdArray::from_vec(vec![Complex::new(1.0f64, 2.0), Complex::new(0.0, 0.0)]);
        let c = a.eq(&b).unwrap();
        assert_eq!(c.dtype(), DType::Bool);
    }

    #[test]
    fn test_lt_complex() {
        let a = NdArray::from_vec(vec![Complex::new(1.0f64, 2.0)]);
        let b = NdArray::from_vec(vec![Complex::new(3.0f64, 4.0)]);
        let c = a.lt(&b).unwrap();
        assert_eq!(c.dtype(), DType::Bool);
    }

    #[test]
    fn test_gt_complex() {
        let a = NdArray::from_vec(vec![Complex::new(3.0f64, 4.0)]);
        let b = NdArray::from_vec(vec![Complex::new(1.0f64, 2.0)]);
        let c = a.gt(&b).unwrap();
        assert_eq!(c.dtype(), DType::Bool);
    }

    #[test]
    fn test_resolve_comparison_promotes_cross_dtype_numeric() {
        let plan = resolve_comparison_op(ComparisonOp::Eq, DType::Int32, DType::Float64).unwrap();
        assert_eq!(plan.execution_dtype(), DType::Float64);
    }

    #[test]
    fn test_resolve_comparison_rejects_string_numeric_mix() {
        assert!(resolve_comparison_op(ComparisonOp::Eq, DType::Str, DType::Float64).is_err());
    }
}
