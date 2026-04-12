use std::ops;

use crate::array_data::ArrayData;
use crate::broadcasting::{broadcast_array_data, broadcast_shape};
use crate::descriptor::descriptor_for_dtype;
use crate::error::{NumpyError, Result};
use crate::kernel::ArithmeticKernelOp;
use crate::resolver::{resolve_binary_op, BinaryOp, BinaryOpPlan};
use crate::NdArray;

struct PreparedBinaryExecution {
    lhs: ArrayData,
    rhs: ArrayData,
    plan: BinaryOpPlan,
}

fn prepare_binary_execution(
    op: BinaryOp,
    lhs: &NdArray,
    rhs: &NdArray,
) -> Result<PreparedBinaryExecution> {
    let plan = resolve_binary_op(op, lhs.dtype(), rhs.dtype())?;
    let out_shape = broadcast_shape(lhs.shape(), rhs.shape())?;

    let lhs = broadcast_array_data(&lhs.cast_for_execution(plan.lhs_cast()), &out_shape);
    let rhs = broadcast_array_data(&rhs.cast_for_execution(plan.rhs_cast()), &out_shape);

    Ok(PreparedBinaryExecution { lhs, rhs, plan })
}

fn arithmetic_kernel_op(op: BinaryOp) -> ArithmeticKernelOp {
    match op {
        BinaryOp::Add => ArithmeticKernelOp::Add,
        BinaryOp::Sub => ArithmeticKernelOp::Sub,
        BinaryOp::Mul => ArithmeticKernelOp::Mul,
        BinaryOp::Div => ArithmeticKernelOp::Div,
        BinaryOp::FloorDiv => ArithmeticKernelOp::FloorDiv,
        BinaryOp::Remainder => ArithmeticKernelOp::Remainder,
        BinaryOp::Pow => ArithmeticKernelOp::Pow,
    }
}

fn execute_resolved_binary(op: BinaryOp, lhs: &NdArray, rhs: &NdArray) -> Result<NdArray> {
    let prepared = prepare_binary_execution(op, lhs, rhs)?;
    let descriptor = descriptor_for_dtype(prepared.plan.result_storage_dtype());
    let kernel = descriptor
        .binary_kernel(arithmetic_kernel_op(op))
        .ok_or_else(|| {
            NumpyError::TypeError("unsupported operand types for binary operation".into())
        })?;
    let data = kernel(prepared.lhs, prepared.rhs)?;
    Ok(NdArray::from_binary_plan_result(data, prepared.plan))
}

fn execute_real_binary(op: BinaryOp, lhs: &NdArray, rhs: &NdArray) -> Result<NdArray> {
    execute_resolved_binary(op, lhs, rhs)
}

impl ops::Add<&NdArray> for &NdArray {
    type Output = Result<NdArray>;

    fn add(self, rhs: &NdArray) -> Result<NdArray> {
        execute_resolved_binary(BinaryOp::Add, self, rhs)
    }
}

impl ops::Sub<&NdArray> for &NdArray {
    type Output = Result<NdArray>;

    fn sub(self, rhs: &NdArray) -> Result<NdArray> {
        execute_resolved_binary(BinaryOp::Sub, self, rhs)
    }
}

impl ops::Mul<&NdArray> for &NdArray {
    type Output = Result<NdArray>;

    fn mul(self, rhs: &NdArray) -> Result<NdArray> {
        execute_resolved_binary(BinaryOp::Mul, self, rhs)
    }
}

impl ops::Div<&NdArray> for &NdArray {
    type Output = Result<NdArray>;

    fn div(self, rhs: &NdArray) -> Result<NdArray> {
        execute_resolved_binary(BinaryOp::Div, self, rhs)
    }
}

impl NdArray {
    /// Element-wise power: self ** rhs.
    pub fn pow(&self, rhs: &NdArray) -> Result<NdArray> {
        execute_resolved_binary(BinaryOp::Pow, self, rhs).map_err(|err| match err {
            NumpyError::TypeError(_) if self.dtype().is_string() || rhs.dtype().is_string() => {
                NumpyError::TypeError("power not supported for string arrays".into())
            }
            other => other,
        })
    }

    /// Element-wise floor division: self // rhs (toward -inf, matching NumPy).
    pub fn floor_div(&self, rhs: &NdArray) -> Result<NdArray> {
        execute_real_binary(BinaryOp::FloorDiv, self, rhs).map_err(|err| match err {
            NumpyError::TypeError(_) if self.dtype().is_complex() || rhs.dtype().is_complex() => {
                NumpyError::TypeError("floor division not supported for complex arrays".into())
            }
            other => other,
        })
    }

    /// Element-wise remainder: self % rhs (sign of divisor, matching NumPy).
    pub fn remainder(&self, rhs: &NdArray) -> Result<NdArray> {
        execute_real_binary(BinaryOp::Remainder, self, rhs).map_err(|err| match err {
            NumpyError::TypeError(_) if self.dtype().is_complex() || rhs.dtype().is_complex() => {
                NumpyError::TypeError("remainder not supported for complex arrays".into())
            }
            other => other,
        })
    }
}

#[cfg(test)]
mod tests {
    use crate::{DType, NdArray};
    use num_complex::Complex;

    #[test]
    fn test_add_same_shape() {
        let a = NdArray::from_vec(vec![1.0_f64, 2.0, 3.0]);
        let b = NdArray::from_vec(vec![4.0_f64, 5.0, 6.0]);
        let c = (&a + &b).unwrap();
        assert_eq!(c.shape(), &[3]);
        assert_eq!(c.dtype(), DType::Float64);
    }

    #[test]
    fn test_add_type_promotion() {
        let a = NdArray::from_vec(vec![1_i32, 2, 3]);
        let b = NdArray::from_vec(vec![1.0_f64, 2.0, 3.0]);
        let c = (&a + &b).unwrap();
        assert_eq!(c.dtype(), DType::Float64);
    }

    #[test]
    fn test_add_broadcast() {
        let a = NdArray::zeros(&[3, 4], DType::Float64);
        let b = NdArray::ones(&[4], DType::Float64);
        let c = (&a + &b).unwrap();
        assert_eq!(c.shape(), &[3, 4]);
    }

    #[test]
    fn test_sub() {
        let a = NdArray::from_vec(vec![5.0_f64, 3.0]);
        let b = NdArray::from_vec(vec![1.0_f64, 1.0]);
        let c = (&a - &b).unwrap();
        assert_eq!(c.dtype(), DType::Float64);
    }

    #[test]
    fn test_mul() {
        let a = NdArray::from_vec(vec![2.0_f64, 3.0]);
        let b = NdArray::from_vec(vec![4.0_f64, 5.0]);
        let _c = (&a * &b).unwrap();
    }

    #[test]
    fn test_div() {
        let a = NdArray::from_vec(vec![10.0_f64, 20.0]);
        let b = NdArray::from_vec(vec![2.0_f64, 5.0]);
        let _c = (&a / &b).unwrap();
    }

    #[test]
    fn test_broadcast_incompatible_fails() {
        let a = NdArray::zeros(&[3, 4], DType::Float64);
        let b = NdArray::zeros(&[5], DType::Float64);
        assert!((&a + &b).is_err());
    }

    #[test]
    fn test_add_i32() {
        let a = NdArray::from_vec(vec![1_i32, 2, 3]);
        let b = NdArray::from_vec(vec![10_i32, 20, 30]);
        let c = (&a + &b).unwrap();
        assert_eq!(c.dtype(), DType::Int32);
    }

    #[test]
    fn test_mul_broadcast_scalar() {
        let a = NdArray::from_vec(vec![1.0_f64, 2.0, 3.0]);
        let b = NdArray::full_f64(&[], 2.0); // scalar
        let c = (&a * &b).unwrap();
        assert_eq!(c.shape(), &[3]);
    }

    #[test]
    fn test_add_complex() {
        let a = NdArray::from_vec(vec![Complex::new(1.0f64, 2.0), Complex::new(3.0, 4.0)]);
        let b = NdArray::from_vec(vec![Complex::new(5.0f64, 6.0), Complex::new(7.0, 8.0)]);
        let c = (&a + &b).unwrap();
        assert_eq!(c.dtype(), DType::Complex128);
        assert_eq!(c.shape(), &[2]);
    }

    #[test]
    fn test_mul_complex() {
        let a = NdArray::from_vec(vec![Complex::new(1.0f64, 1.0)]);
        let b = NdArray::from_vec(vec![Complex::new(1.0f64, -1.0)]);
        let c = (&a * &b).unwrap();
        assert_eq!(c.dtype(), DType::Complex128);
    }

    #[test]
    fn test_floor_div_complex_fails() {
        let a = NdArray::from_vec(vec![Complex::new(1.0f64, 2.0)]);
        let b = NdArray::from_vec(vec![Complex::new(1.0f64, 0.0)]);
        assert!(a.floor_div(&b).is_err());
    }

    #[test]
    fn test_remainder_complex_fails() {
        let a = NdArray::from_vec(vec![Complex::new(1.0f64, 2.0)]);
        let b = NdArray::from_vec(vec![Complex::new(1.0f64, 0.0)]);
        assert!(a.remainder(&b).is_err());
    }
}
