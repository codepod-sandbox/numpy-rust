use std::ops;

use num_complex::Complex;

use crate::array::{BoxedObjectScalar, BoxedScalar};
use crate::array_data::ArrayData;
use crate::broadcasting::{broadcast_array_data, broadcast_shape};
use crate::descriptor::descriptor_for_dtype;
use crate::error::{NumpyError, Result};
use crate::kernel::ArithmeticKernelOp;
use crate::ops::comparison::{broadcast_coord, iter_boxed_coords};
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

    let lhs = broadcast_array_data(lhs.cast_for_execution(plan.lhs_cast()), &out_shape);
    let rhs = broadcast_array_data(rhs.cast_for_execution(plan.rhs_cast()), &out_shape);

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
    if lhs.dtype().is_boxed() || rhs.dtype().is_boxed() {
        return execute_boxed_binary(op, lhs, rhs);
    }
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

fn execute_boxed_binary(op: BinaryOp, lhs: &NdArray, rhs: &NdArray) -> Result<NdArray> {
    let out_shape = broadcast_shape(lhs.shape(), rhs.shape())?;
    let mut out = Vec::with_capacity(out_shape.iter().product());

    for coord in iter_boxed_coords(&out_shape) {
        let lhs_scalar = boxed_scalar_for_arithmetic_coord(lhs, &coord, &out_shape)?;
        let rhs_scalar = boxed_scalar_for_arithmetic_coord(rhs, &coord, &out_shape)?;
        out.push(apply_boxed_binary(op, lhs_scalar, rhs_scalar)?);
    }

    let result_dtype = if out.is_empty() {
        boxed_binary_result_dtype(op, lhs.dtype(), rhs.dtype())?
    } else {
        boxed_scalar_dtype(&out[0])
    };
    NdArray::from_boxed_scalars(out, &out_shape, result_dtype)
}

fn boxed_scalar_for_arithmetic_coord(
    input: &NdArray,
    coord: &[usize],
    out_shape: &[usize],
) -> Result<BoxedScalar> {
    let input_coord = broadcast_coord(coord, out_shape, input.shape());
    if input.dtype().is_boxed() {
        return input.get_boxed(&input_coord);
    }
    let scalar = input.get(&input_coord)?;
    Ok(BoxedScalar::Object(match scalar {
        crate::indexing::Scalar::Bool(v) => BoxedObjectScalar::Bool(v),
        crate::indexing::Scalar::Int32(v) => BoxedObjectScalar::Int(v as i64),
        crate::indexing::Scalar::Int64(v) => BoxedObjectScalar::Int(v),
        crate::indexing::Scalar::Float32(v) => BoxedObjectScalar::Float(v as f64),
        crate::indexing::Scalar::Float64(v) => BoxedObjectScalar::Float(v),
        crate::indexing::Scalar::Complex64(v) => {
            BoxedObjectScalar::Complex(Complex::new(v.re as f64, v.im as f64))
        }
        crate::indexing::Scalar::Complex128(v) => BoxedObjectScalar::Complex(v),
        crate::indexing::Scalar::Str(v) => BoxedObjectScalar::Text(v),
    }))
}

fn boxed_scalar_dtype(value: &BoxedScalar) -> crate::DType {
    match value {
        BoxedScalar::Object(_) => crate::DType::Object,
        BoxedScalar::Datetime(_) => crate::DType::Datetime64,
        BoxedScalar::Timedelta(_) => crate::DType::Timedelta64,
    }
}

fn boxed_binary_result_dtype(
    op: BinaryOp,
    lhs: crate::DType,
    rhs: crate::DType,
) -> Result<crate::DType> {
    use crate::DType::{Datetime64, Object, Timedelta64};
    match (lhs, rhs) {
        (Datetime64, Datetime64) => match op {
            BinaryOp::Sub => Ok(Timedelta64),
            _ => Err(NumpyError::TypeError(format!(
                "boxed operation {:?} not supported between datetime64 and datetime64",
                op
            ))),
        },
        (Datetime64, Timedelta64) => match op {
            BinaryOp::Add | BinaryOp::Sub => Ok(Datetime64),
            _ => Err(NumpyError::TypeError(format!(
                "boxed operation {:?} not supported between datetime64 and timedelta64",
                op
            ))),
        },
        (Timedelta64, Datetime64) => match op {
            BinaryOp::Add => Ok(Datetime64),
            _ => Err(NumpyError::TypeError(format!(
                "boxed operation {:?} not supported between timedelta64 and datetime64",
                op
            ))),
        },
        (Timedelta64, Timedelta64) => Ok(Timedelta64),
        (Timedelta64, Object) | (Object, Timedelta64) => Ok(Timedelta64),
        (Object, Object) => Ok(Object),
        _ => Err(NumpyError::TypeError(format!(
            "boxed operation {:?} not supported between {} and {}",
            op, lhs, rhs
        ))),
    }
}

pub(crate) fn apply_boxed_binary(
    op: BinaryOp,
    lhs: BoxedScalar,
    rhs: BoxedScalar,
) -> Result<BoxedScalar> {
    match (lhs, rhs) {
        (BoxedScalar::Object(lhs), BoxedScalar::Object(rhs)) => {
            apply_boxed_object_binary(op, lhs, rhs).map(BoxedScalar::Object)
        }
        (BoxedScalar::Timedelta(lhs), BoxedScalar::Timedelta(rhs)) => {
            apply_boxed_timedelta_binary(op, lhs, rhs).map(BoxedScalar::Timedelta)
        }
        (BoxedScalar::Datetime(lhs), BoxedScalar::Datetime(rhs)) => {
            apply_boxed_datetime_binary(op, lhs, rhs)
        }
        (BoxedScalar::Datetime(lhs), BoxedScalar::Timedelta(rhs)) => {
            apply_boxed_datetime_timedelta_binary(op, lhs, rhs)
        }
        (BoxedScalar::Timedelta(lhs), BoxedScalar::Datetime(rhs)) => {
            apply_boxed_timedelta_datetime_binary(op, lhs, rhs)
        }
        (BoxedScalar::Timedelta(lhs), BoxedScalar::Object(rhs)) => {
            apply_boxed_timedelta_numeric_binary(op, lhs, rhs).map(BoxedScalar::Timedelta)
        }
        (BoxedScalar::Object(lhs), BoxedScalar::Timedelta(rhs)) => {
            apply_boxed_numeric_timedelta_binary(op, lhs, rhs).map(BoxedScalar::Timedelta)
        }
        (lhs, rhs) => Err(NumpyError::TypeError(format!(
            "boxed arithmetic {:?} is not supported between {:?} and {:?}",
            op, lhs, rhs
        ))),
    }
}

fn apply_boxed_object_binary(
    op: BinaryOp,
    lhs: BoxedObjectScalar,
    rhs: BoxedObjectScalar,
) -> Result<BoxedObjectScalar> {
    use BoxedObjectScalar as O;

    match op {
        BinaryOp::Add => match (lhs, rhs) {
            (O::Text(a), O::Text(b)) => Ok(O::Text(a + &b)),
            (lhs, rhs) => numeric_object_binary(lhs, rhs, |a, b| a + b, |a, b| a + b, |a, b| a + b),
        },
        BinaryOp::Sub => numeric_object_binary(lhs, rhs, |a, b| a - b, |a, b| a - b, |a, b| a - b),
        BinaryOp::Mul => match (lhs, rhs) {
            (O::Text(text), O::Int(times)) => Ok(O::Text(repeat_text(text, times))),
            (O::Text(text), O::Bool(flag)) => {
                Ok(O::Text(repeat_text(text, if flag { 1 } else { 0 })))
            }
            (O::Int(times), O::Text(text)) => Ok(O::Text(repeat_text(text, times))),
            (O::Bool(flag), O::Text(text)) => {
                Ok(O::Text(repeat_text(text, if flag { 1 } else { 0 })))
            }
            (lhs, rhs) => numeric_object_binary(lhs, rhs, |a, b| a * b, |a, b| a * b, |a, b| a * b),
        },
        BinaryOp::Div => numeric_object_div(lhs, rhs),
        BinaryOp::FloorDiv => numeric_object_floor_div(lhs, rhs),
        BinaryOp::Remainder => numeric_object_remainder(lhs, rhs),
        BinaryOp::Pow => numeric_object_pow(lhs, rhs),
    }
}

fn repeat_text(text: String, times: i64) -> String {
    if times <= 0 {
        String::new()
    } else {
        text.repeat(times as usize)
    }
}

fn common_time_unit<'a>(lhs: &'a str, rhs: &'a str) -> &'a str {
    const ORDER: [&str; 10] = ["Y", "M", "W", "D", "h", "m", "s", "ms", "us", "ns"];
    if lhs == rhs {
        return lhs;
    }
    if lhs == "generic" {
        return rhs;
    }
    if rhs == "generic" {
        return lhs;
    }
    let lhs_idx = ORDER.iter().position(|unit| *unit == lhs).unwrap_or(3);
    let rhs_idx = ORDER.iter().position(|unit| *unit == rhs).unwrap_or(3);
    ORDER[lhs_idx.max(rhs_idx)]
}

fn to_common_unit(value: i64, from_unit: &str, to_unit: &str) -> i64 {
    if from_unit == to_unit || from_unit == "generic" || to_unit == "generic" {
        return value;
    }
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

fn temporal_target_unit(lhs: &str, rhs: &str) -> String {
    common_time_unit(lhs, rhs).to_owned()
}

fn apply_boxed_timedelta_binary(
    op: BinaryOp,
    lhs: crate::array::BoxedTemporalScalar,
    rhs: crate::array::BoxedTemporalScalar,
) -> Result<crate::array::BoxedTemporalScalar> {
    let unit = temporal_target_unit(lhs.unit.as_str(), rhs.unit.as_str());
    if lhs.is_nat || rhs.is_nat {
        return Ok(crate::array::BoxedTemporalScalar {
            value: 0,
            unit,
            is_nat: true,
        });
    }
    let lhs_value = to_common_unit(lhs.value, lhs.unit.as_str(), unit.as_str());
    let rhs_value = to_common_unit(rhs.value, rhs.unit.as_str(), unit.as_str());
    let value = match op {
        BinaryOp::Add => lhs_value + rhs_value,
        BinaryOp::Sub => lhs_value - rhs_value,
        BinaryOp::Mul
        | BinaryOp::Div
        | BinaryOp::FloorDiv
        | BinaryOp::Remainder
        | BinaryOp::Pow => {
            return Err(NumpyError::TypeError(format!(
                "boxed arithmetic {:?} is not supported between timedelta64 values",
                op
            )))
        }
    };
    Ok(crate::array::BoxedTemporalScalar {
        value,
        unit,
        is_nat: false,
    })
}

fn apply_boxed_datetime_binary(
    op: BinaryOp,
    lhs: crate::array::BoxedTemporalScalar,
    rhs: crate::array::BoxedTemporalScalar,
) -> Result<BoxedScalar> {
    match op {
        BinaryOp::Sub => {
            let unit = temporal_target_unit(lhs.unit.as_str(), rhs.unit.as_str());
            if lhs.is_nat || rhs.is_nat {
                return Ok(BoxedScalar::Timedelta(crate::array::BoxedTemporalScalar {
                    value: 0,
                    unit,
                    is_nat: true,
                }));
            }
            let lhs_value = to_common_unit(lhs.value, lhs.unit.as_str(), unit.as_str());
            let rhs_value = to_common_unit(rhs.value, rhs.unit.as_str(), unit.as_str());
            Ok(BoxedScalar::Timedelta(crate::array::BoxedTemporalScalar {
                value: lhs_value - rhs_value,
                unit,
                is_nat: false,
            }))
        }
        _ => Err(NumpyError::TypeError(format!(
            "boxed arithmetic {:?} is not supported between datetime64 values",
            op
        ))),
    }
}

fn apply_boxed_datetime_timedelta_binary(
    op: BinaryOp,
    lhs: crate::array::BoxedTemporalScalar,
    rhs: crate::array::BoxedTemporalScalar,
) -> Result<BoxedScalar> {
    match op {
        BinaryOp::Add | BinaryOp::Sub => {
            let unit = if lhs.unit == "generic" {
                rhs.unit.clone()
            } else {
                lhs.unit.clone()
            };
            if lhs.is_nat || rhs.is_nat {
                return Ok(BoxedScalar::Datetime(crate::array::BoxedTemporalScalar {
                    value: 0,
                    unit,
                    is_nat: true,
                }));
            }
            let rhs_value = to_common_unit(rhs.value, rhs.unit.as_str(), unit.as_str());
            let value = if matches!(op, BinaryOp::Add) {
                lhs.value + rhs_value
            } else {
                lhs.value - rhs_value
            };
            Ok(BoxedScalar::Datetime(crate::array::BoxedTemporalScalar {
                value,
                unit,
                is_nat: false,
            }))
        }
        _ => Err(NumpyError::TypeError(format!(
            "boxed arithmetic {:?} is not supported between datetime64 and timedelta64 values",
            op
        ))),
    }
}

fn apply_boxed_timedelta_datetime_binary(
    op: BinaryOp,
    lhs: crate::array::BoxedTemporalScalar,
    rhs: crate::array::BoxedTemporalScalar,
) -> Result<BoxedScalar> {
    match op {
        BinaryOp::Add => apply_boxed_datetime_timedelta_binary(op, rhs, lhs),
        _ => Err(NumpyError::TypeError(format!(
            "boxed arithmetic {:?} is not supported between timedelta64 and datetime64 values",
            op
        ))),
    }
}

fn object_to_i64_checked(value: &BoxedObjectScalar) -> Result<i64> {
    use BoxedObjectScalar as O;
    match value {
        O::Bool(v) => Ok(if *v { 1 } else { 0 }),
        O::Int(v) => Ok(*v),
        O::Float(v) => Ok(*v as i64),
        O::Complex(_) | O::Text(_) => Err(NumpyError::TypeError(
            "temporal arithmetic requires a real numeric scalar".into(),
        )),
    }
}

fn apply_boxed_timedelta_numeric_binary(
    op: BinaryOp,
    lhs: crate::array::BoxedTemporalScalar,
    rhs: BoxedObjectScalar,
) -> Result<crate::array::BoxedTemporalScalar> {
    let rhs = object_to_i64_checked(&rhs)?;
    if lhs.is_nat {
        return Ok(lhs);
    }
    let value = match op {
        BinaryOp::Mul => lhs.value * rhs,
        BinaryOp::Div => lhs.value / rhs,
        BinaryOp::FloorDiv => lhs.value / rhs,
        _ => {
            return Err(NumpyError::TypeError(format!(
                "boxed arithmetic {:?} is not supported between timedelta64 and numeric object values",
                op
            )))
        }
    };
    Ok(crate::array::BoxedTemporalScalar {
        value,
        unit: lhs.unit,
        is_nat: false,
    })
}

fn apply_boxed_numeric_timedelta_binary(
    op: BinaryOp,
    lhs: BoxedObjectScalar,
    rhs: crate::array::BoxedTemporalScalar,
) -> Result<crate::array::BoxedTemporalScalar> {
    let lhs = object_to_i64_checked(&lhs)?;
    if rhs.is_nat {
        return Ok(rhs);
    }
    let value = match op {
        BinaryOp::Mul => lhs * rhs.value,
        _ => {
            return Err(NumpyError::TypeError(format!(
                "boxed arithmetic {:?} is not supported between numeric object values and timedelta64",
                op
            )))
        }
    };
    Ok(crate::array::BoxedTemporalScalar {
        value,
        unit: rhs.unit,
        is_nat: false,
    })
}

fn numeric_object_binary(
    lhs: BoxedObjectScalar,
    rhs: BoxedObjectScalar,
    complex_op: impl FnOnce(Complex<f64>, Complex<f64>) -> Complex<f64>,
    float_op: impl FnOnce(f64, f64) -> f64,
    int_op: impl FnOnce(i64, i64) -> i64,
) -> Result<BoxedObjectScalar> {
    use BoxedObjectScalar as O;

    if matches!(lhs, O::Text(_)) || matches!(rhs, O::Text(_)) {
        return Err(NumpyError::TypeError(
            "operation is not supported for string object scalars".into(),
        ));
    }

    if matches!(lhs, O::Complex(_)) || matches!(rhs, O::Complex(_)) {
        return Ok(O::Complex(complex_op(
            object_to_complex(&lhs),
            object_to_complex(&rhs),
        )));
    }

    if matches!(lhs, O::Float(_)) || matches!(rhs, O::Float(_)) {
        return Ok(O::Float(float_op(object_to_f64(&lhs), object_to_f64(&rhs))));
    }

    Ok(O::Int(int_op(object_to_i64(&lhs), object_to_i64(&rhs))))
}

fn numeric_object_floor_div(
    lhs: BoxedObjectScalar,
    rhs: BoxedObjectScalar,
) -> Result<BoxedObjectScalar> {
    use BoxedObjectScalar as O;

    if matches!(lhs, O::Text(_))
        || matches!(rhs, O::Text(_))
        || matches!(lhs, O::Complex(_))
        || matches!(rhs, O::Complex(_))
    {
        return Err(NumpyError::TypeError(
            "floor division is not supported for these object scalars".into(),
        ));
    }

    if matches!(lhs, O::Float(_)) || matches!(rhs, O::Float(_)) {
        if object_to_f64(&rhs) == 0.0 {
            return Err(NumpyError::ValueError("division by zero".into()));
        }
        return Ok(O::Float(
            (object_to_f64(&lhs) / object_to_f64(&rhs)).floor(),
        ));
    }

    let a = object_to_i64(&lhs);
    let b = object_to_i64(&rhs);
    if b == 0 {
        return Err(NumpyError::ValueError("division by zero".into()));
    }
    let q = a / b;
    let r = a % b;
    Ok(O::Int(if r != 0 && ((r > 0) != (b > 0)) {
        q - 1
    } else {
        q
    }))
}

fn numeric_object_div(lhs: BoxedObjectScalar, rhs: BoxedObjectScalar) -> Result<BoxedObjectScalar> {
    use BoxedObjectScalar as O;

    if matches!(lhs, O::Text(_)) || matches!(rhs, O::Text(_)) {
        return Err(NumpyError::TypeError(
            "division is not supported for string object scalars".into(),
        ));
    }

    if matches!(lhs, O::Complex(_)) || matches!(rhs, O::Complex(_)) {
        return Ok(O::Complex(
            object_to_complex(&lhs) / object_to_complex(&rhs),
        ));
    }

    Ok(O::Float(object_to_f64(&lhs) / object_to_f64(&rhs)))
}

fn numeric_object_remainder(
    lhs: BoxedObjectScalar,
    rhs: BoxedObjectScalar,
) -> Result<BoxedObjectScalar> {
    use BoxedObjectScalar as O;

    if matches!(lhs, O::Text(_))
        || matches!(rhs, O::Text(_))
        || matches!(lhs, O::Complex(_))
        || matches!(rhs, O::Complex(_))
    {
        return Err(NumpyError::TypeError(
            "remainder is not supported for these object scalars".into(),
        ));
    }

    if matches!(lhs, O::Float(_)) || matches!(rhs, O::Float(_)) {
        let a = object_to_f64(&lhs);
        let b = object_to_f64(&rhs);
        if b == 0.0 {
            return Err(NumpyError::ValueError("division by zero".into()));
        }
        return Ok(O::Float(a - (a / b).floor() * b));
    }

    let a = object_to_i64(&lhs);
    let b = object_to_i64(&rhs);
    if b == 0 {
        return Err(NumpyError::ValueError("division by zero".into()));
    }
    let mut r = a % b;
    if r != 0 && ((r > 0) != (b > 0)) {
        r += b;
    }
    Ok(O::Int(r))
}

fn numeric_object_pow(lhs: BoxedObjectScalar, rhs: BoxedObjectScalar) -> Result<BoxedObjectScalar> {
    use BoxedObjectScalar as O;

    if matches!(lhs, O::Text(_)) || matches!(rhs, O::Text(_)) {
        return Err(NumpyError::TypeError(
            "power is not supported for string object scalars".into(),
        ));
    }

    if matches!(lhs, O::Complex(_)) || matches!(rhs, O::Complex(_)) {
        return Ok(O::Complex(
            object_to_complex(&lhs).powc(object_to_complex(&rhs)),
        ));
    }

    if matches!(lhs, O::Float(_)) || matches!(rhs, O::Float(_)) {
        return Ok(O::Float(object_to_f64(&lhs).powf(object_to_f64(&rhs))));
    }

    let base = object_to_i64(&lhs);
    let exp = object_to_i64(&rhs);
    if exp < 0 {
        return Ok(O::Float((base as f64).powf(exp as f64)));
    }
    Ok(O::Int(base.pow(exp as u32)))
}

fn object_to_i64(value: &BoxedObjectScalar) -> i64 {
    match value {
        BoxedObjectScalar::Bool(v) => {
            if *v {
                1
            } else {
                0
            }
        }
        BoxedObjectScalar::Int(v) => *v,
        BoxedObjectScalar::Float(v) => *v as i64,
        BoxedObjectScalar::Complex(v) => v.re as i64,
        BoxedObjectScalar::Text(_) => unreachable!("text handled before numeric conversion"),
    }
}

fn object_to_f64(value: &BoxedObjectScalar) -> f64 {
    match value {
        BoxedObjectScalar::Bool(v) => {
            if *v {
                1.0
            } else {
                0.0
            }
        }
        BoxedObjectScalar::Int(v) => *v as f64,
        BoxedObjectScalar::Float(v) => *v,
        BoxedObjectScalar::Complex(v) => v.re,
        BoxedObjectScalar::Text(_) => unreachable!("text handled before numeric conversion"),
    }
}

fn object_to_complex(value: &BoxedObjectScalar) -> Complex<f64> {
    match value {
        BoxedObjectScalar::Bool(v) => Complex::new(if *v { 1.0 } else { 0.0 }, 0.0),
        BoxedObjectScalar::Int(v) => Complex::new(*v as f64, 0.0),
        BoxedObjectScalar::Float(v) => Complex::new(*v, 0.0),
        BoxedObjectScalar::Complex(v) => *v,
        BoxedObjectScalar::Text(_) => unreachable!("text handled before numeric conversion"),
    }
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
    use crate::{BoxedObjectScalar, BoxedScalar, BoxedTemporalScalar, DType, NdArray};
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
    fn test_div_int_promotes_to_float64() {
        let a = NdArray::from_vec(vec![3_i32, 4]);
        let b = NdArray::from_vec(vec![2_i32, 2]);
        let c = (&a / &b).unwrap();
        assert_eq!(c.dtype(), DType::Float64);
        assert_eq!(c.to_flat_f64_vec(), vec![1.5, 2.0]);
    }

    #[test]
    fn test_div_bool_promotes_to_float64() {
        let a = NdArray::from_vec(vec![true, false]);
        let b = NdArray::full_f64(&[], 2.0);
        let c = (&a / &b).unwrap();
        assert_eq!(c.dtype(), DType::Float64);
        assert_eq!(c.to_flat_f64_vec(), vec![0.5, 0.0]);
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

    #[test]
    fn test_add_boxed_object_arrays() {
        let a = NdArray::from_boxed_scalars(
            vec![
                BoxedScalar::Object(BoxedObjectScalar::Int(1)),
                BoxedScalar::Object(BoxedObjectScalar::Int(2)),
            ],
            &[2],
            DType::Object,
        )
        .unwrap();
        let b = NdArray::from_boxed_scalars(
            vec![
                BoxedScalar::Object(BoxedObjectScalar::Int(10)),
                BoxedScalar::Object(BoxedObjectScalar::Int(20)),
            ],
            &[2],
            DType::Object,
        )
        .unwrap();
        let out = (&a + &b).unwrap();
        assert_eq!(
            out.get_boxed(&[0]).unwrap(),
            BoxedScalar::Object(BoxedObjectScalar::Int(11))
        );
        assert_eq!(
            out.get_boxed(&[1]).unwrap(),
            BoxedScalar::Object(BoxedObjectScalar::Int(22))
        );
    }

    #[test]
    fn test_mul_boxed_object_string_repeat() {
        let a = NdArray::from_boxed_scalars(
            vec![BoxedScalar::Object(BoxedObjectScalar::Text("ab".into()))],
            &[1],
            DType::Object,
        )
        .unwrap();
        let b = NdArray::from_boxed_scalars(
            vec![BoxedScalar::Object(BoxedObjectScalar::Int(3))],
            &[1],
            DType::Object,
        )
        .unwrap();
        let out = (&a * &b).unwrap();
        assert_eq!(
            out.get_boxed(&[0]).unwrap(),
            BoxedScalar::Object(BoxedObjectScalar::Text("ababab".into()))
        );
    }

    #[test]
    fn test_sub_boxed_timedelta_generic_and_ns() {
        let a = NdArray::from_boxed_scalars(
            vec![BoxedScalar::Timedelta(BoxedTemporalScalar {
                value: 0,
                unit: "generic".into(),
                is_nat: false,
            })],
            &[1],
            DType::Timedelta64,
        )
        .unwrap();
        let b = NdArray::from_boxed_scalars(
            vec![BoxedScalar::Timedelta(BoxedTemporalScalar {
                value: 1,
                unit: "ns".into(),
                is_nat: false,
            })],
            &[1],
            DType::Timedelta64,
        )
        .unwrap();
        let out = (&a - &b).unwrap();
        assert_eq!(
            out.get_boxed(&[0]).unwrap(),
            BoxedScalar::Timedelta(BoxedTemporalScalar {
                value: -1,
                unit: "ns".into(),
                is_nat: false,
            })
        );
    }

    #[test]
    fn test_sub_boxed_datetime_yields_timedelta() {
        let a = NdArray::from_boxed_scalars(
            vec![BoxedScalar::Datetime(BoxedTemporalScalar {
                value: 10,
                unit: "D".into(),
                is_nat: false,
            })],
            &[1],
            DType::Datetime64,
        )
        .unwrap();
        let b = NdArray::from_boxed_scalars(
            vec![BoxedScalar::Datetime(BoxedTemporalScalar {
                value: 3,
                unit: "D".into(),
                is_nat: false,
            })],
            &[1],
            DType::Datetime64,
        )
        .unwrap();
        let out = (&a - &b).unwrap();
        assert_eq!(
            out.get_boxed(&[0]).unwrap(),
            BoxedScalar::Timedelta(BoxedTemporalScalar {
                value: 7,
                unit: "D".into(),
                is_nat: false,
            })
        );
    }

    #[test]
    fn test_add_boxed_datetime_and_timedelta() {
        let a = NdArray::from_boxed_scalars(
            vec![BoxedScalar::Datetime(BoxedTemporalScalar {
                value: 10,
                unit: "D".into(),
                is_nat: false,
            })],
            &[1],
            DType::Datetime64,
        )
        .unwrap();
        let b = NdArray::from_boxed_scalars(
            vec![BoxedScalar::Timedelta(BoxedTemporalScalar {
                value: 2,
                unit: "D".into(),
                is_nat: false,
            })],
            &[1],
            DType::Timedelta64,
        )
        .unwrap();
        let out = (&a + &b).unwrap();
        assert_eq!(
            out.get_boxed(&[0]).unwrap(),
            BoxedScalar::Datetime(BoxedTemporalScalar {
                value: 12,
                unit: "D".into(),
                is_nat: false,
            })
        );
    }

    #[test]
    fn test_mul_boxed_timedelta_by_object_scalar() {
        let a = NdArray::from_boxed_scalars(
            vec![BoxedScalar::Timedelta(BoxedTemporalScalar {
                value: 5,
                unit: "D".into(),
                is_nat: false,
            })],
            &[1],
            DType::Timedelta64,
        )
        .unwrap();
        let b = NdArray::from_boxed_scalars(
            vec![BoxedScalar::Object(BoxedObjectScalar::Int(3))],
            &[1],
            DType::Object,
        )
        .unwrap();
        let out = (&a * &b).unwrap();
        assert_eq!(
            out.get_boxed(&[0]).unwrap(),
            BoxedScalar::Timedelta(BoxedTemporalScalar {
                value: 15,
                unit: "D".into(),
                is_nat: false,
            })
        );
    }
}
