use std::cmp::Ordering;

use crate::array::{BoxedObjectScalar, BoxedScalar, BoxedTemporalScalar};
use crate::array_data::{ArrayD, ArrayData};
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

    let lhs = broadcast_array_data(lhs.cast_for_execution(plan.lhs_cast()), &out_shape);
    let rhs = broadcast_array_data(rhs.cast_for_execution(plan.rhs_cast()), &out_shape);

    Ok((lhs, rhs, plan))
}

fn execute_comparison(lhs: &NdArray, rhs: &NdArray, op: ComparisonOp) -> Result<NdArray> {
    if lhs.dtype().is_boxed() || rhs.dtype().is_boxed() {
        return execute_boxed_comparison(lhs, rhs, op);
    }
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

pub(crate) fn boxed_execution_dtype(lhs: &NdArray, rhs: &NdArray) -> Result<crate::DType> {
    let lhs_dt = lhs.dtype();
    let rhs_dt = rhs.dtype();
    if lhs_dt == rhs_dt {
        return Ok(lhs_dt);
    }
    if lhs_dt == crate::DType::Object || rhs_dt == crate::DType::Object {
        return Ok(crate::DType::Object);
    }
    Err(NumpyError::TypeError(format!(
        "boxed operation not supported between {} and {}",
        lhs_dt, rhs_dt
    )))
}

pub(crate) fn compare_boxed_scalars(
    lhs: &BoxedScalar,
    rhs: &BoxedScalar,
) -> Result<Option<Ordering>> {
    match (lhs, rhs) {
        (BoxedScalar::Object(a), BoxedScalar::Object(b)) => compare_object_scalars(a, b),
        (BoxedScalar::Datetime(a), BoxedScalar::Datetime(b))
        | (BoxedScalar::Timedelta(a), BoxedScalar::Timedelta(b)) => {
            Ok(Some(compare_temporal_scalars(a, b)?))
        }
        _ => Err(NumpyError::TypeError(
            "boxed comparison requires matching boxed scalar kinds".into(),
        )),
    }
}

pub(crate) fn boxed_scalar_for_coord(
    input: &NdArray,
    target_dtype: crate::DType,
    coord: &[usize],
    out_shape: &[usize],
) -> Result<BoxedScalar> {
    let input_coord = broadcast_coord(coord, out_shape, input.shape());
    if input.dtype().is_boxed() {
        let scalar = input.get_boxed(&input_coord)?;
        if input.dtype() == target_dtype {
            return Ok(scalar);
        }
        return coerce_boxed_scalar_to_dtype(scalar, target_dtype);
    }

    if target_dtype != crate::DType::Object {
        return Err(NumpyError::TypeError(format!(
            "cannot coerce {} into boxed dtype {}",
            input.dtype(),
            target_dtype
        )));
    }

    let scalar = input.get(&input_coord)?;
    Ok(BoxedScalar::Object(match scalar {
        crate::indexing::Scalar::Bool(v) => BoxedObjectScalar::Bool(v),
        crate::indexing::Scalar::Int32(v) => BoxedObjectScalar::Int(v as i64),
        crate::indexing::Scalar::Int64(v) => BoxedObjectScalar::Int(v),
        crate::indexing::Scalar::Float32(v) => BoxedObjectScalar::Float(v as f64),
        crate::indexing::Scalar::Float64(v) => BoxedObjectScalar::Float(v),
        crate::indexing::Scalar::Complex64(v) => {
            BoxedObjectScalar::Complex(num_complex::Complex::new(v.re as f64, v.im as f64))
        }
        crate::indexing::Scalar::Complex128(v) => BoxedObjectScalar::Complex(v),
        crate::indexing::Scalar::Str(v) => BoxedObjectScalar::Text(v),
    }))
}

pub(crate) fn broadcast_coord(
    coord: &[usize],
    out_shape: &[usize],
    input_shape: &[usize],
) -> Vec<usize> {
    if input_shape.is_empty() {
        return vec![];
    }
    let offset = out_shape.len().saturating_sub(input_shape.len());
    input_shape
        .iter()
        .enumerate()
        .map(
            |(axis, &dim)| {
                if dim == 1 {
                    0
                } else {
                    coord[offset + axis]
                }
            },
        )
        .collect()
}

fn execute_boxed_comparison(lhs: &NdArray, rhs: &NdArray, op: ComparisonOp) -> Result<NdArray> {
    let out_shape = broadcast_shape(lhs.shape(), rhs.shape())?;
    let execution_dtype = boxed_execution_dtype(lhs, rhs)?;
    let mut out = Vec::with_capacity(out_shape.iter().product());

    for coord in iter_boxed_coords(&out_shape) {
        let lhs_scalar = boxed_scalar_for_coord(lhs, execution_dtype, &coord, &out_shape)?;
        let rhs_scalar = boxed_scalar_for_coord(rhs, execution_dtype, &coord, &out_shape)?;
        let cmp = compare_boxed_scalars(&lhs_scalar, &rhs_scalar);
        let value = match op {
            ComparisonOp::Eq => cmp
                .as_ref()
                .ok()
                .and_then(|ord| *ord)
                .map(|ord| ord == Ordering::Equal)
                .unwrap_or(false),
            ComparisonOp::Ne => cmp
                .as_ref()
                .ok()
                .and_then(|ord| *ord)
                .map(|ord| ord != Ordering::Equal)
                .unwrap_or(true),
            ComparisonOp::Lt => cmp
                .as_ref()
                .ok()
                .and_then(|ord| *ord)
                .map(|ord| ord == Ordering::Less)
                .unwrap_or(false),
            ComparisonOp::Le => cmp
                .as_ref()
                .ok()
                .and_then(|ord| *ord)
                .map(|ord| matches!(ord, Ordering::Less | Ordering::Equal))
                .unwrap_or(false),
            ComparisonOp::Gt => cmp
                .as_ref()
                .ok()
                .and_then(|ord| *ord)
                .map(|ord| ord == Ordering::Greater)
                .unwrap_or(false),
            ComparisonOp::Ge => cmp
                .as_ref()
                .ok()
                .and_then(|ord| *ord)
                .map(|ord| matches!(ord, Ordering::Greater | Ordering::Equal))
                .unwrap_or(false),
        };
        if matches!(
            op,
            ComparisonOp::Lt | ComparisonOp::Le | ComparisonOp::Gt | ComparisonOp::Ge
        ) && cmp.is_err()
        {
            cmp?;
        }
        out.push(value);
    }

    Ok(NdArray::from_data(ArrayData::Bool(
        ArrayD::from_shape_vec(ndarray::IxDyn(out_shape.as_slice()), out)
            .unwrap()
            .into_shared(),
    )))
}

pub(crate) fn iter_boxed_coords(shape: &[usize]) -> impl Iterator<Item = Vec<usize>> + '_ {
    let size: usize = shape.iter().product();
    (0..size).map(move |mut idx| {
        if shape.is_empty() {
            return vec![];
        }
        let mut coord = vec![0usize; shape.len()];
        for d in (0..shape.len()).rev() {
            coord[d] = idx % shape[d];
            idx /= shape[d];
        }
        coord
    })
}

fn coerce_boxed_scalar_to_dtype(value: BoxedScalar, target: crate::DType) -> Result<BoxedScalar> {
    match (value, target) {
        (BoxedScalar::Object(value), crate::DType::Object) => Ok(BoxedScalar::Object(value)),
        (BoxedScalar::Datetime(value), crate::DType::Datetime64) => {
            Ok(BoxedScalar::Datetime(value))
        }
        (BoxedScalar::Timedelta(value), crate::DType::Timedelta64) => {
            Ok(BoxedScalar::Timedelta(value))
        }
        (BoxedScalar::Datetime(value), crate::DType::Object) => {
            Err(NumpyError::TypeError(format!(
                "cannot coerce datetime64 scalar {:?} into object boxed runtime yet",
                value
            )))
        }
        (BoxedScalar::Timedelta(value), crate::DType::Object) => {
            Err(NumpyError::TypeError(format!(
                "cannot coerce timedelta64 scalar {:?} into object boxed runtime yet",
                value
            )))
        }
        (_, _) => Err(NumpyError::TypeError("unsupported boxed coercion".into())),
    }
}

fn compare_object_scalars(
    lhs: &BoxedObjectScalar,
    rhs: &BoxedObjectScalar,
) -> Result<Option<Ordering>> {
    use BoxedObjectScalar as O;
    let ord = match (lhs, rhs) {
        (O::Text(a), O::Text(b)) => Some(a.cmp(b)),
        (O::Complex(a), O::Complex(b)) => Some(complex_cmp(a, b)),
        (O::Complex(a), other) => Some(complex_cmp(a, &object_to_complex(other)?)),
        (other, O::Complex(b)) => Some(complex_cmp(&object_to_complex(other)?, b)),
        _ => numeric_value(lhs)?
            .partial_cmp(&numeric_value(rhs)?)
            .map(Some)
            .unwrap_or(None),
    };
    Ok(ord)
}

fn object_to_complex(value: &BoxedObjectScalar) -> Result<num_complex::Complex<f64>> {
    Ok(match value {
        BoxedObjectScalar::Complex(v) => *v,
        BoxedObjectScalar::Bool(v) => num_complex::Complex::new(if *v { 1.0 } else { 0.0 }, 0.0),
        BoxedObjectScalar::Int(v) => num_complex::Complex::new(*v as f64, 0.0),
        BoxedObjectScalar::Float(v) => num_complex::Complex::new(*v, 0.0),
        BoxedObjectScalar::Text(_) => {
            return Err(NumpyError::TypeError(
                "cannot compare complex and string object scalars".into(),
            ))
        }
    })
}

fn numeric_value(value: &BoxedObjectScalar) -> Result<f64> {
    Ok(match value {
        BoxedObjectScalar::Bool(v) => {
            if *v {
                1.0
            } else {
                0.0
            }
        }
        BoxedObjectScalar::Int(v) => *v as f64,
        BoxedObjectScalar::Float(v) => *v,
        BoxedObjectScalar::Text(_) => {
            return Err(NumpyError::TypeError(
                "cannot order string object scalars against numeric values".into(),
            ))
        }
        BoxedObjectScalar::Complex(_) => {
            return Err(NumpyError::TypeError(
                "complex object ordering is handled separately".into(),
            ))
        }
    })
}

fn compare_temporal_scalars(
    lhs: &BoxedTemporalScalar,
    rhs: &BoxedTemporalScalar,
) -> Result<Ordering> {
    let target_unit = if lhs.unit == rhs.unit {
        lhs.unit.clone()
    } else {
        common_time_unit(lhs.unit.as_str(), rhs.unit.as_str()).to_string()
    };
    let lhs = to_common_unit(lhs.value, lhs.unit.as_str(), target_unit.as_str());
    let rhs = to_common_unit(rhs.value, rhs.unit.as_str(), target_unit.as_str());
    Ok(lhs.cmp(&rhs))
}

fn to_common_unit(value: i64, from_unit: &str, to_unit: &str) -> i64 {
    let days = match from_unit {
        "Y" => value as f64 * 365.0,
        "M" => value as f64 * 30.0,
        "W" => value as f64 * 7.0,
        "D" => value as f64,
        "h" => value as f64 / 24.0,
        "m" => value as f64 / 1440.0,
        "s" => value as f64 / 86400.0,
        "ms" => value as f64 / 86_400_000.0,
        "us" => value as f64 / 86_400_000_000.0,
        "ns" => value as f64 / 86_400_000_000_000.0,
        _ => value as f64,
    };
    match to_unit {
        "Y" => (days / 365.0) as i64,
        "M" => (days / 30.0) as i64,
        "W" => (days / 7.0) as i64,
        "D" => days as i64,
        "h" => (days * 24.0) as i64,
        "m" => (days * 1440.0) as i64,
        "s" => (days * 86400.0) as i64,
        "ms" => (days * 86_400_000.0) as i64,
        "us" => (days * 86_400_000_000.0) as i64,
        "ns" => (days * 86_400_000_000_000.0) as i64,
        _ => days as i64,
    }
}

fn common_time_unit<'a>(lhs: &'a str, rhs: &'a str) -> &'a str {
    const ORDER: [&str; 10] = ["Y", "M", "W", "D", "h", "m", "s", "ms", "us", "ns"];
    let lhs_idx = ORDER.iter().position(|unit| *unit == lhs).unwrap_or(3);
    let rhs_idx = ORDER.iter().position(|unit| *unit == rhs).unwrap_or(3);
    ORDER[lhs_idx.max(rhs_idx)]
}

#[cfg(test)]
mod tests {
    use crate::array::{BoxedObjectScalar, BoxedScalar};
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
    fn test_boxed_object_nan_comparisons_are_unordered_not_errors() {
        let lhs = NdArray::from_boxed_scalars(
            vec![
                BoxedScalar::Object(BoxedObjectScalar::Float(f64::NAN)),
                BoxedScalar::Object(BoxedObjectScalar::Float(1.0)),
            ],
            &[2],
            DType::Object,
        )
        .unwrap();
        let rhs = NdArray::from_boxed_scalars(
            vec![
                BoxedScalar::Object(BoxedObjectScalar::Float(f64::NAN)),
                BoxedScalar::Object(BoxedObjectScalar::Float(2.0)),
            ],
            &[2],
            DType::Object,
        )
        .unwrap();

        assert!(lhs.eq(&rhs).is_ok());
        assert!(lhs.ne(&rhs).is_ok());
        assert!(lhs.le(&rhs).is_ok());
        assert!(lhs.lt(&rhs).is_ok());
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
