use crate::array_data::ArrayData;
use crate::dtype::DType;
use crate::error::{NumpyError, Result};
use crate::NdArray;

/// One-dimensional linear interpolation.
/// x: x-coordinates at which to evaluate
/// xp: x-coordinates of data points (must be sorted ascending)
/// fp: y-coordinates of data points (same length as xp)
/// Values outside xp range are clamped to fp[0] and fp[last].
pub fn interp(x: &NdArray, xp: &NdArray, fp: &NdArray) -> Result<NdArray> {
    let x_f = x.astype(DType::Float64).flatten();
    let xp_f = xp.astype(DType::Float64).flatten();
    let fp_f = fp.astype(DType::Float64).flatten();

    let ArrayData::Float64(x_arr) = x_f.data() else {
        unreachable!()
    };
    let ArrayData::Float64(xp_arr) = xp_f.data() else {
        unreachable!()
    };
    let ArrayData::Float64(fp_arr) = fp_f.data() else {
        unreachable!()
    };

    if xp_arr.len() != fp_arr.len() {
        return Err(NumpyError::ValueError(
            "xp and fp must have same length".into(),
        ));
    }
    if xp_arr.is_empty() {
        return Err(NumpyError::ValueError("xp must not be empty".into()));
    }

    let xp_slice: Vec<f64> = xp_arr.iter().copied().collect();
    let fp_slice: Vec<f64> = fp_arr.iter().copied().collect();

    let mut result = Vec::with_capacity(x_arr.len());
    for &xi in x_arr.iter() {
        // NaN input → NaN output
        if xi.is_nan() {
            result.push(f64::NAN);
            continue;
        }
        // Binary search: partition_point returns first index where xp[i] >= xi
        // (i.e., first index not satisfying v < xi)
        let idx = xp_slice.partition_point(|&v| v < xi);
        if idx == 0 {
            // xi <= xp_slice[0] → left boundary; NaN in xp[0] gives NaN
            if xp_slice[0].is_nan() {
                result.push(f64::NAN);
            } else {
                result.push(fp_slice[0]);
            }
        } else if idx == xp_slice.len() {
            // xi >= xp_slice[last] → right boundary; NaN in xp[-1] gives NaN
            let last = xp_slice.len() - 1;
            if xp_slice[last].is_nan() {
                result.push(f64::NAN);
            } else {
                result.push(*fp_slice.last().unwrap());
            }
        } else {
            let lo = idx - 1;
            let hi = idx;
            // Check for NaN in the bracket that would make interpolation undefined
            if xp_slice[lo].is_nan() || xp_slice[hi].is_nan() {
                result.push(f64::NAN);
                continue;
            }
            // Check for NaN in fp values
            if fp_slice[lo].is_nan() || fp_slice[hi].is_nan() {
                result.push(f64::NAN);
                continue;
            }
            // Exact match at hi: return fp[hi] directly to avoid inf-inf = NaN
            if xi == xp_slice[hi] {
                result.push(fp_slice[hi]);
                continue;
            }
            let denom = xp_slice[hi] - xp_slice[lo];
            if denom == 0.0 {
                result.push(fp_slice[lo]);
            } else if denom.is_infinite() {
                // Infinite bracket width: handle -inf/+inf xp boundaries
                // If xp[lo] = -inf and xp[hi] is finite: t → 1
                // If xp[lo] is finite and xp[hi] = +inf: t → 0
                if xp_slice[lo].is_infinite() && xp_slice[lo] < 0.0 {
                    // xp[lo] = -inf: result approaches fp[hi]
                    result.push(fp_slice[hi]);
                } else {
                    // xp[hi] = +inf: result stays at fp[lo]
                    result.push(fp_slice[lo]);
                }
            } else {
                let t = (xi - xp_slice[lo]) / denom;
                let v = fp_slice[lo] + t * (fp_slice[hi] - fp_slice[lo]);
                result.push(v);
            }
        }
    }

    Ok(NdArray::from_vec(result))
}

/// Compute numerical gradient using central differences.
/// For 1-D arrays: returns single array.
/// spacing: uniform spacing between elements (default 1.0).
/// Uses central differences for interior points, forward/backward at edges.
pub fn gradient_1d(f: &NdArray, spacing: f64) -> Result<NdArray> {
    let arr = f.astype(DType::Float64).flatten();
    let ArrayData::Float64(data) = arr.data() else {
        unreachable!()
    };
    let n = data.len();

    if n < 2 {
        return Err(NumpyError::ValueError(
            "gradient requires at least 2 elements".into(),
        ));
    }

    let mut result = vec![0.0_f64; n];

    // Forward difference at start
    result[0] = (data[1] - data[0]) / spacing;
    // Backward difference at end
    result[n - 1] = (data[n - 1] - data[n - 2]) / spacing;
    // Central differences for interior
    for i in 1..n - 1 {
        result[i] = (data[i + 1] - data[i - 1]) / (2.0 * spacing);
    }

    Ok(NdArray::from_vec(result))
}

/// N-D gradient: compute gradient along each axis.
/// Returns one array per axis.
pub fn gradient_nd(f: &NdArray, spacing: f64) -> Result<Vec<NdArray>> {
    let arr = f.astype(DType::Float64);
    let ArrayData::Float64(data) = arr.data() else {
        unreachable!()
    };
    let shape = data.shape().to_vec();
    let ndim = shape.len();

    if ndim == 0 {
        return Err(NumpyError::ValueError(
            "gradient requires at least 1-D array".into(),
        ));
    }

    let mut results = Vec::with_capacity(ndim);

    for (ax, &n) in shape.iter().enumerate() {
        if n < 2 {
            return Err(NumpyError::ValueError(
                "gradient requires at least 2 elements along each axis".into(),
            ));
        }

        let mut grad = data.clone();

        // Iterate over all positions using lanes
        for mut lane in grad.lanes_mut(ndarray::Axis(ax)) {
            let vals: Vec<f64> = lane.iter().copied().collect();
            // Forward difference at start
            lane[0] = (vals[1] - vals[0]) / spacing;
            // Backward difference at end
            lane[n - 1] = (vals[n - 1] - vals[n - 2]) / spacing;
            // Central differences
            for i in 1..n - 1 {
                lane[i] = (vals[i + 1] - vals[i - 1]) / (2.0 * spacing);
            }
        }

        results.push(NdArray::from_data(ArrayData::Float64(grad)));
    }

    Ok(results)
}

/// Full-featured gradient: supports non-uniform spacing, edge_order, axis selection.
/// spacing: either scalar (uniform) or array (coordinate values along axis).
/// edge_order: 1 or 2 for boundary accuracy.
/// axes: which axes to compute gradient for (None = all).
/// Returns one NdArray per axis.
pub fn gradient_full(
    f: &NdArray,
    spacings: &[GradientSpacing],
    edge_order: usize,
    axes: &[usize],
) -> Result<Vec<NdArray>> {
    if edge_order != 1 && edge_order != 2 {
        return Err(NumpyError::ValueError("'edge_order' must be 1 or 2".into()));
    }

    let arr = f.astype(DType::Float64);
    let ArrayData::Float64(data) = arr.data() else {
        unreachable!()
    };
    let shape = data.shape().to_vec();
    let ndim = shape.len();

    if ndim == 0 {
        return Err(NumpyError::ValueError(
            "gradient requires at least 1-D array".into(),
        ));
    }

    let mut results = Vec::with_capacity(axes.len());

    for (sp_idx, &ax) in axes.iter().enumerate() {
        let n = shape[ax];
        let min_size = edge_order + 1;
        if n < min_size {
            return Err(NumpyError::ValueError(format!(
                "Shape of array too small to calculate a numerical gradient, \
                 at least {} elements are required for edge_order={}",
                min_size, edge_order
            )));
        }

        let spacing = if sp_idx < spacings.len() {
            &spacings[sp_idx]
        } else {
            &GradientSpacing::Uniform(1.0)
        };

        let mut grad = data.clone();

        match spacing {
            GradientSpacing::Uniform(dx) => {
                let dx = *dx;
                for mut lane in grad.lanes_mut(ndarray::Axis(ax)) {
                    let vals: Vec<f64> = lane.iter().copied().collect();
                    // Interior: central differences
                    for i in 1..n - 1 {
                        lane[i] = (vals[i + 1] - vals[i - 1]) / (2.0 * dx);
                    }
                    // Boundaries
                    if edge_order == 1 {
                        lane[0] = (vals[1] - vals[0]) / dx;
                        lane[n - 1] = (vals[n - 1] - vals[n - 2]) / dx;
                    } else {
                        lane[0] = (-3.0 * vals[0] + 4.0 * vals[1] - vals[2]) / (2.0 * dx);
                        lane[n - 1] =
                            (vals[n - 3] - 4.0 * vals[n - 2] + 3.0 * vals[n - 1]) / (2.0 * dx);
                    }
                }
            }
            GradientSpacing::Coordinates(x) => {
                // Compute consecutive differences
                let diffs: Vec<f64> = x.windows(2).map(|w| w[1] - w[0]).collect();
                for mut lane in grad.lanes_mut(ndarray::Axis(ax)) {
                    let vals: Vec<f64> = lane.iter().copied().collect();
                    // Interior: weighted central differences for non-uniform grid
                    for i in 1..n - 1 {
                        let dx1 = diffs[i - 1];
                        let dx2 = diffs[i];
                        lane[i] = (vals[i + 1] * dx1 * dx1 + (dx2 * dx2 - dx1 * dx1) * vals[i]
                            - vals[i - 1] * dx2 * dx2)
                            / (dx1 * dx2 * (dx1 + dx2));
                    }
                    // Boundaries
                    if edge_order == 1 {
                        lane[0] = (vals[1] - vals[0]) / diffs[0];
                        lane[n - 1] = (vals[n - 1] - vals[n - 2]) / diffs[n - 2];
                    } else {
                        // 2nd order: left boundary
                        let dx1 = diffs[0];
                        let dx2 = diffs[1];
                        let a = -(2.0 * dx1 + dx2) / (dx1 * (dx1 + dx2));
                        let b = (dx1 + dx2) / (dx1 * dx2);
                        let c = -dx1 / (dx2 * (dx1 + dx2));
                        lane[0] = a * vals[0] + b * vals[1] + c * vals[2];
                        // 2nd order: right boundary
                        let dx1 = diffs[n - 3];
                        let dx2 = diffs[n - 2];
                        let a = dx2 / (dx1 * (dx1 + dx2));
                        let b = -(dx2 + dx1) / (dx1 * dx2);
                        let c = (2.0 * dx2 + dx1) / (dx2 * (dx1 + dx2));
                        lane[n - 1] = a * vals[n - 3] + b * vals[n - 2] + c * vals[n - 1];
                    }
                }
            }
        }

        results.push(NdArray::from_data(ArrayData::Float64(grad)));
    }

    Ok(results)
}

/// Spacing specification for gradient computation.
pub enum GradientSpacing {
    /// Uniform spacing between elements.
    Uniform(f64),
    /// Coordinate values along the axis.
    Coordinates(Vec<f64>),
}

/// Trapezoidal numerical integration along axis.
/// y: values array
/// x: optional x-coordinates (if None, uses uniform spacing dx)
/// dx: spacing when x is None (default 1.0)
/// axis: integration axis (None = last axis)
pub fn trapz(y: &NdArray, x: Option<&NdArray>, dx: f64, axis: Option<i64>) -> Result<NdArray> {
    let ndim = y.ndim() as i64;
    let axis_idx = match axis {
        Some(a) if a < 0 => (ndim + a) as usize,
        Some(a) => a as usize,
        None => {
            if ndim > 0 {
                (ndim - 1) as usize
            } else {
                0
            }
        }
    };
    if axis_idx >= y.ndim() {
        return Err(NumpyError::ValueError(format!(
            "axis {} is out of bounds for array of dimension {}",
            axis_idx,
            y.ndim()
        )));
    }

    let y_f = y.astype(DType::Float64);
    let ArrayData::Float64(y_arr) = y_f.data() else {
        unreachable!()
    };

    let n = y.shape()[axis_idx];
    if n < 2 {
        return Err(NumpyError::ValueError(
            "trapz requires at least 2 elements along integration axis".into(),
        ));
    }

    // Build dx array along axis_idx
    let dx_vals: Vec<f64> = if let Some(x_arr) = x {
        let x_f = x_arr.astype(DType::Float64);
        let ArrayData::Float64(xa) = x_f.data() else {
            unreachable!()
        };
        let x_flat: Vec<f64> = xa.iter().copied().collect();
        x_flat.windows(2).map(|w| w[1] - w[0]).collect()
    } else {
        vec![dx; n - 1]
    };

    // For 1-D case:
    if y.ndim() == 1 {
        let y_flat: Vec<f64> = y_arr.iter().copied().collect();
        let result: f64 = y_flat
            .windows(2)
            .zip(dx_vals.iter())
            .map(|(w, &d)| 0.5 * (w[0] + w[1]) * d)
            .sum();
        return Ok(NdArray::from_data(ArrayData::Float64(
            ndarray::ArrayD::from_elem(ndarray::IxDyn(&[]), result).into_shared(),
        )));
    }

    // Multi-dimensional: reduce along axis_idx
    let shape = y.shape().to_vec();
    let mut out_shape = shape.clone();
    out_shape.remove(axis_idx);
    let out_n: usize = out_shape.iter().product();
    let mut result_flat = vec![0.0_f64; out_n];

    for (out_idx, out_val) in result_flat.iter_mut().enumerate() {
        let mut rem = out_idx;
        let mut in_idx_base = vec![0usize; y.ndim()];
        let mut ax_out = out_shape.len();
        for d in (0..y.ndim()).rev() {
            if d == axis_idx {
                continue;
            }
            ax_out -= 1;
            let dim = out_shape[ax_out];
            in_idx_base[d] = rem % dim;
            rem /= dim;
        }
        let mut sum = 0.0;
        for (i, &d) in dx_vals.iter().enumerate() {
            in_idx_base[axis_idx] = i;
            let v0 = y_arr[ndarray::IxDyn(&in_idx_base)];
            in_idx_base[axis_idx] = i + 1;
            let v1 = y_arr[ndarray::IxDyn(&in_idx_base)];
            sum += 0.5 * (v0 + v1) * d;
        }
        *out_val = sum;
    }

    let result = ndarray::ArrayD::from_shape_vec(out_shape, result_flat)
        .map_err(|e| NumpyError::ValueError(e.to_string()))?
        .into_shared();
    Ok(NdArray::from_data(ArrayData::Float64(result)))
}

/// Cumulative trapezoidal integration. Returns array with length n-1 along axis.
pub fn cumulative_trapezoid(
    y: &NdArray,
    x: Option<&NdArray>,
    dx: f64,
    axis: Option<i64>,
) -> Result<NdArray> {
    let ndim = y.ndim() as i64;
    let axis_idx = match axis {
        Some(a) if a < 0 => (ndim + a) as usize,
        Some(a) => a as usize,
        None => {
            if ndim > 0 {
                (ndim - 1) as usize
            } else {
                0
            }
        }
    };
    if axis_idx >= y.ndim() {
        return Err(NumpyError::ValueError(format!(
            "axis {} is out of bounds for array of dimension {}",
            axis_idx,
            y.ndim()
        )));
    }

    let y_f = y.astype(DType::Float64);
    let ArrayData::Float64(y_arr) = y_f.data() else {
        unreachable!()
    };
    let n = y.shape()[axis_idx];
    if n < 2 {
        return Err(NumpyError::ValueError(
            "cumulative_trapezoid requires at least 2 elements".into(),
        ));
    }

    let dx_vals: Vec<f64> = if let Some(x_arr) = x {
        let x_f = x_arr.astype(DType::Float64);
        let ArrayData::Float64(xa) = x_f.data() else {
            unreachable!()
        };
        let x_flat: Vec<f64> = xa.iter().copied().collect();
        x_flat.windows(2).map(|w| w[1] - w[0]).collect()
    } else {
        vec![dx; n - 1]
    };

    let mut out_shape = y.shape().to_vec();
    out_shape[axis_idx] = n - 1;
    let out_n: usize = out_shape.iter().product();
    let mut result_flat = vec![0.0_f64; out_n];

    if y.ndim() == 1 {
        let y_flat: Vec<f64> = y_arr.iter().copied().collect();
        let mut cumsum = 0.0;
        for (i, (&d, w)) in dx_vals.iter().zip(y_flat.windows(2)).enumerate() {
            cumsum += 0.5 * (w[0] + w[1]) * d;
            result_flat[i] = cumsum;
        }
        let result = ndarray::ArrayD::from_shape_vec(out_shape, result_flat)
            .map_err(|e| NumpyError::ValueError(e.to_string()))?
            .into_shared();
        return Ok(NdArray::from_data(ArrayData::Float64(result)));
    }

    for (out_flat, out_cell) in result_flat.iter_mut().enumerate() {
        let mut rem = out_flat;
        let mut out_multi = vec![0usize; y.ndim()];
        for d in (0..y.ndim()).rev() {
            out_multi[d] = rem % out_shape[d];
            rem /= out_shape[d];
        }
        let seg = out_multi[axis_idx];
        let mut cumsum = 0.0;
        let mut in_idx = out_multi.clone();
        for (k, &d) in dx_vals.iter().enumerate().take(seg + 1) {
            in_idx[axis_idx] = k;
            let v0 = y_arr[ndarray::IxDyn(&in_idx)];
            in_idx[axis_idx] = k + 1;
            let v1 = y_arr[ndarray::IxDyn(&in_idx)];
            cumsum += 0.5 * (v0 + v1) * d;
        }
        *out_cell = cumsum;
    }

    let result = ndarray::ArrayD::from_shape_vec(out_shape, result_flat)
        .map_err(|e| NumpyError::ValueError(e.to_string()))?
        .into_shared();
    Ok(NdArray::from_data(ArrayData::Float64(result)))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::array_data::ArrayData;

    #[test]
    fn test_interp_basic() {
        let x = NdArray::from_vec(vec![1.5_f64, 2.5]);
        let xp = NdArray::from_vec(vec![1.0_f64, 2.0, 3.0]);
        let fp = NdArray::from_vec(vec![10.0_f64, 20.0, 30.0]);
        let result = interp(&x, &xp, &fp).unwrap();
        let ArrayData::Float64(arr) = result.data() else {
            panic!()
        };
        assert!((arr[[0]] - 15.0).abs() < 1e-10);
        assert!((arr[[1]] - 25.0).abs() < 1e-10);
    }

    #[test]
    fn test_interp_clamp() {
        let x = NdArray::from_vec(vec![0.0_f64, 10.0]);
        let xp = NdArray::from_vec(vec![1.0_f64, 3.0]);
        let fp = NdArray::from_vec(vec![5.0_f64, 15.0]);
        let result = interp(&x, &xp, &fp).unwrap();
        let ArrayData::Float64(arr) = result.data() else {
            panic!()
        };
        assert!((arr[[0]] - 5.0).abs() < 1e-10); // clamped to fp[0]
        assert!((arr[[1]] - 15.0).abs() < 1e-10); // clamped to fp[-1]
    }

    #[test]
    fn test_gradient_1d() {
        // f = [1, 2, 4, 7, 11], spacing=1
        // gradient = [1, 1.5, 2.5, 3.5, 4] (forward, central, central, central, backward)
        let f = NdArray::from_vec(vec![1.0_f64, 2.0, 4.0, 7.0, 11.0]);
        let g = gradient_1d(&f, 1.0).unwrap();
        let ArrayData::Float64(arr) = g.data() else {
            panic!()
        };
        assert!((arr[[0]] - 1.0).abs() < 1e-10);
        assert!((arr[[1]] - 1.5).abs() < 1e-10);
        assert!((arr[[2]] - 2.5).abs() < 1e-10);
        assert!((arr[[3]] - 3.5).abs() < 1e-10);
        assert!((arr[[4]] - 4.0).abs() < 1e-10);
    }

    #[test]
    fn test_gradient_1d_spacing() {
        let f = NdArray::from_vec(vec![0.0_f64, 2.0, 4.0]);
        let g = gradient_1d(&f, 0.5).unwrap();
        let ArrayData::Float64(arr) = g.data() else {
            panic!()
        };
        // forward: (2-0)/0.5=4, central: (4-0)/(2*0.5)=4, backward: (4-2)/0.5=4
        assert!((arr[[0]] - 4.0).abs() < 1e-10);
        assert!((arr[[1]] - 4.0).abs() < 1e-10);
        assert!((arr[[2]] - 4.0).abs() < 1e-10);
    }

    fn f64_vals(r: &crate::NdArray) -> Vec<f64> {
        let ArrayData::Float64(a) = r.data() else {
            panic!("expected Float64")
        };
        a.iter().copied().collect()
    }

    #[test]
    fn test_trapz_basic() {
        // trapz([1, 2, 3], dx=1) = 4.0
        let y = NdArray::from_vec(vec![1.0_f64, 2.0, 3.0]);
        let r = trapz(&y, None, 1.0, None).unwrap();
        let vals = f64_vals(&r);
        assert!((vals[0] - 4.0).abs() < 1e-10, "trapz = {}", vals[0]);
    }

    #[test]
    fn test_trapz_with_x() {
        // trapz([1,2,3], x=[0,1,3]) = 0.5*(1+2)*1 + 0.5*(2+3)*2 = 1.5 + 5 = 6.5
        let y = NdArray::from_vec(vec![1.0_f64, 2.0, 3.0]);
        let x = NdArray::from_vec(vec![0.0_f64, 1.0, 3.0]);
        let r = trapz(&y, Some(&x), 1.0, None).unwrap();
        let vals = f64_vals(&r);
        assert!((vals[0] - 6.5).abs() < 1e-10, "trapz = {}", vals[0]);
    }

    #[test]
    fn test_cumulative_trapezoid() {
        // cumtrapz([1,2,3], dx=1) = [1.5, 4.0]
        let y = NdArray::from_vec(vec![1.0_f64, 2.0, 3.0]);
        let r = cumulative_trapezoid(&y, None, 1.0, None).unwrap();
        let vals = f64_vals(&r);
        assert_eq!(vals.len(), 2);
        assert!((vals[0] - 1.5).abs() < 1e-10);
        assert!((vals[1] - 4.0).abs() < 1e-10);
    }

    #[test]
    fn test_gradient_nd() {
        let f = NdArray::from_vec(vec![1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0])
            .reshape(&[2, 3])
            .unwrap();
        let grads = gradient_nd(&f, 1.0).unwrap();
        assert_eq!(grads.len(), 2);
        assert_eq!(grads[0].shape(), &[2, 3]); // gradient along axis 0
        assert_eq!(grads[1].shape(), &[2, 3]); // gradient along axis 1
    }
}
