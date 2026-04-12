use std::ops;

use ndarray::Zip;

use crate::array_data::ArrayData;
use crate::broadcasting::{broadcast_array_data, broadcast_shape};
use crate::casting::cast_array_data;
use crate::descriptor::descriptor_for_dtype;
use crate::dtype::DType;
use crate::error::{NumpyError, Result};
use crate::kernel::ArithmeticKernelOp;
use crate::resolver::{resolve_binary_op, BinaryOp, BinaryOpPlan};
use crate::NdArray;

struct PreparedBinaryExecution {
    lhs: ArrayData,
    rhs: ArrayData,
    plan: BinaryOpPlan,
}

/// Prepare two NdArrays for a binary operation using the existing per-op dtype policy.
/// Returns `(lhs_data, rhs_data, logical_result_dtype)`.
fn prepare_binary(lhs: &NdArray, rhs: &NdArray) -> Result<(ArrayData, ArrayData, DType)> {
    if lhs.dtype().is_string() || rhs.dtype().is_string() {
        return Err(crate::error::NumpyError::TypeError(
            "arithmetic not supported for string arrays".into(),
        ));
    }
    let logical_dtype = lhs.dtype().promote(rhs.dtype());
    let storage_dtype = logical_dtype.storage_dtype();
    let out_shape = broadcast_shape(lhs.shape(), rhs.shape())?;

    let a = cast_array_data(lhs.data(), storage_dtype);
    let b = cast_array_data(rhs.data(), storage_dtype);

    let a = broadcast_array_data(&a, &out_shape);
    let b = broadcast_array_data(&b, &out_shape);

    Ok((a, b, logical_dtype))
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
    /// Integer types are cast to Float64 first (matching NumPy behavior).
    /// Complex types use Complex::powc.
    pub fn pow(&self, rhs: &NdArray) -> Result<NdArray> {
        if self.dtype().is_string() || rhs.dtype().is_string() {
            return Err(NumpyError::TypeError(
                "power not supported for string arrays".into(),
            ));
        }
        // If either is complex, work in complex domain
        if self.dtype().is_complex() || rhs.dtype().is_complex() {
            let lhs_c = self.astype(DType::Complex128);
            let rhs_c = rhs.astype(DType::Complex128);
            let (a, b, _) = prepare_binary(&lhs_c, &rhs_c)?;
            return match (a, b) {
                (ArrayData::Complex128(a), ArrayData::Complex128(b)) => {
                    let mut out = a.clone();
                    Zip::from(&mut out).and(&b).for_each(|o, &r| {
                        *o = o.powc(r);
                    });
                    Ok(NdArray::from_data(ArrayData::Complex128(out)))
                }
                _ => unreachable!("both cast to Complex128"),
            };
        }
        // Cast both to Float64 for uniform powf handling
        let lhs_f = self.astype(DType::Float64);
        let rhs_f = rhs.astype(DType::Float64);
        let (a, b, _) = prepare_binary(&lhs_f, &rhs_f)?;
        match (a, b) {
            (ArrayData::Float64(a), ArrayData::Float64(b)) => {
                let mut out = a.clone();
                Zip::from(&mut out).and(&b).for_each(|o, &r| {
                    *o = o.powf(r);
                });
                Ok(NdArray::from_data(ArrayData::Float64(out)))
            }
            _ => unreachable!("both cast to Float64"),
        }
    }

    /// Element-wise floor division: self // rhs (toward -inf, matching NumPy).
    pub fn floor_div(&self, rhs: &NdArray) -> Result<NdArray> {
        if self.dtype().is_complex() || rhs.dtype().is_complex() {
            return Err(NumpyError::TypeError(
                "floor division not supported for complex arrays".into(),
            ));
        }
        if self.dtype() == DType::Bool || rhs.dtype() == DType::Bool {
            let target = if self.dtype() == DType::Bool && rhs.dtype() == DType::Bool {
                DType::Int8
            } else if self.dtype() == DType::Bool {
                rhs.dtype()
            } else {
                self.dtype()
            };
            let storage = target.storage_dtype();
            let lhs_up = if self.dtype() == DType::Bool {
                self.astype(storage).with_declared_dtype(target)
            } else {
                self.clone()
            };
            let rhs_up = if rhs.dtype() == DType::Bool {
                rhs.astype(storage).with_declared_dtype(target)
            } else {
                rhs.clone()
            };
            return lhs_up.floor_div(&rhs_up);
        }
        let (a, b, logical_dtype) = prepare_binary(self, rhs)?;
        let data = match (a, b) {
            (ArrayData::Float64(a), ArrayData::Float64(b)) => {
                let mut out = a.clone();
                Zip::from(&mut out).and(&b).for_each(|o, &r| {
                    *o = (*o / r).floor();
                });
                ArrayData::Float64(out)
            }
            (ArrayData::Float32(a), ArrayData::Float32(b)) => {
                let mut out = a.clone();
                Zip::from(&mut out).and(&b).for_each(|o, &r| {
                    *o = (*o / r).floor();
                });
                ArrayData::Float32(out)
            }
            (ArrayData::Int64(a), ArrayData::Int64(b)) => {
                let mut out = a.clone();
                Zip::from(&mut out).and(&b).for_each(|o, &r| {
                    if r == 0 {
                        *o = 0;
                    } else if *o == i64::MIN && r == -1 {
                        *o = i64::MIN;
                    } else {
                        let d = *o / r;
                        let rem = *o % r;
                        *o = if rem != 0 && (rem ^ r) < 0 { d - 1 } else { d };
                    }
                });
                ArrayData::Int64(out)
            }
            (ArrayData::Int32(a), ArrayData::Int32(b)) => {
                let mut out = a.clone();
                Zip::from(&mut out).and(&b).for_each(|o, &r| {
                    if r == 0 {
                        *o = 0;
                    } else if *o == i32::MIN && r == -1 {
                        *o = i32::MIN;
                    } else {
                        let d = *o / r;
                        let rem = *o % r;
                        *o = if rem != 0 && (rem ^ r) < 0 { d - 1 } else { d };
                    }
                });
                ArrayData::Int32(out)
            }
            _ => {
                return Err(NumpyError::TypeError(
                    "unsupported operand types for floor division".into(),
                ));
            }
        };
        let mut result = NdArray::from_data(data);
        if logical_dtype.is_narrow() {
            result.set_declared_dtype(logical_dtype);
        }
        Ok(result)
    }

    /// Element-wise remainder: self % rhs (sign of divisor, matching NumPy).
    pub fn remainder(&self, rhs: &NdArray) -> Result<NdArray> {
        if self.dtype().is_complex() || rhs.dtype().is_complex() {
            return Err(NumpyError::TypeError(
                "remainder not supported for complex arrays".into(),
            ));
        }
        if self.dtype() == DType::Bool || rhs.dtype() == DType::Bool {
            let target = if self.dtype() == DType::Bool && rhs.dtype() == DType::Bool {
                DType::Int8
            } else if self.dtype() == DType::Bool {
                rhs.dtype()
            } else {
                self.dtype()
            };
            let storage = target.storage_dtype();
            let lhs_up = if self.dtype() == DType::Bool {
                self.astype(storage).with_declared_dtype(target)
            } else {
                self.clone()
            };
            let rhs_up = if rhs.dtype() == DType::Bool {
                rhs.astype(storage).with_declared_dtype(target)
            } else {
                rhs.clone()
            };
            return lhs_up.remainder(&rhs_up);
        }
        let (a, b, logical_dtype) = prepare_binary(self, rhs)?;
        let data = match (a, b) {
            (ArrayData::Float64(a), ArrayData::Float64(b)) => {
                let mut out = a.clone();
                Zip::from(&mut out).and(&b).for_each(|o, &r| {
                    *o = *o - (*o / r).floor() * r;
                });
                ArrayData::Float64(out)
            }
            (ArrayData::Float32(a), ArrayData::Float32(b)) => {
                let mut out = a.clone();
                Zip::from(&mut out).and(&b).for_each(|o, &r| {
                    *o = *o - (*o / r).floor() * r;
                });
                ArrayData::Float32(out)
            }
            (ArrayData::Int64(a), ArrayData::Int64(b)) => {
                let mut out = a.clone();
                Zip::from(&mut out).and(&b).for_each(|o, &r| {
                    if r == 0 || (*o == i64::MIN && r == -1) {
                        *o = 0;
                    } else {
                        let rem = *o % r;
                        *o = if rem != 0 && (rem ^ r) < 0 {
                            rem + r
                        } else {
                            rem
                        };
                    }
                });
                ArrayData::Int64(out)
            }
            (ArrayData::Int32(a), ArrayData::Int32(b)) => {
                let mut out = a.clone();
                Zip::from(&mut out).and(&b).for_each(|o, &r| {
                    if r == 0 || (*o == i32::MIN && r == -1) {
                        *o = 0;
                    } else {
                        let rem = *o % r;
                        *o = if rem != 0 && (rem ^ r) < 0 {
                            rem + r
                        } else {
                            rem
                        };
                    }
                });
                ArrayData::Int32(out)
            }
            _ => {
                return Err(NumpyError::TypeError(
                    "unsupported operand types for remainder".into(),
                ));
            }
        };
        let mut result = NdArray::from_data(data);
        if logical_dtype.is_narrow() {
            result.set_declared_dtype(logical_dtype);
        }
        Ok(result)
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
