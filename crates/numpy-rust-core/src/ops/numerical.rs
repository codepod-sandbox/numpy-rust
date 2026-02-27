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

    let ArrayData::Float64(x_arr) = &x_f.data else {
        unreachable!()
    };
    let ArrayData::Float64(xp_arr) = &xp_f.data else {
        unreachable!()
    };
    let ArrayData::Float64(fp_arr) = &fp_f.data else {
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
        if xi <= xp_slice[0] {
            result.push(fp_slice[0]);
        } else if xi >= *xp_slice.last().unwrap() {
            result.push(*fp_slice.last().unwrap());
        } else {
            // Binary search for interval
            let idx = xp_slice.partition_point(|&v| v < xi);
            let lo = idx - 1;
            let hi = idx;
            let t = (xi - xp_slice[lo]) / (xp_slice[hi] - xp_slice[lo]);
            result.push(fp_slice[lo] + t * (fp_slice[hi] - fp_slice[lo]));
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
    let ArrayData::Float64(data) = &arr.data else {
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
    let ArrayData::Float64(data) = &arr.data else {
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
