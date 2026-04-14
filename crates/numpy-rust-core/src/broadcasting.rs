use crate::array_data::ArrayData;
use crate::error::{NumpyError, Result};

/// Compute the broadcast-compatible output shape for two input shapes,
/// following NumPy's broadcasting rules.
pub fn broadcast_shape(a: &[usize], b: &[usize]) -> Result<Vec<usize>> {
    let ndim = a.len().max(b.len());
    let mut result = Vec::with_capacity(ndim);
    for i in 0..ndim {
        let da = if i < ndim - a.len() {
            1
        } else {
            a[i - (ndim - a.len())]
        };
        let db = if i < ndim - b.len() {
            1
        } else {
            b[i - (ndim - b.len())]
        };
        if da == db {
            result.push(da);
        } else if da == 1 {
            result.push(db);
        } else if db == 1 {
            result.push(da);
        } else {
            return Err(NumpyError::BroadcastError(a.to_vec(), b.to_vec()));
        }
    }
    Ok(result)
}

/// Broadcast an ArrayData to the given target shape using ndarray's stride tricks.
/// Panics if the shapes are not broadcast-compatible (caller must validate first).
pub fn broadcast_array_data(data: impl AsRef<ArrayData>, target_shape: &[usize]) -> ArrayData {
    let data = data.as_ref();
    if data.shape() == target_shape {
        return data.clone();
    }
    data.broadcast_to(target_shape)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{DType, NdArray};

    #[test]
    fn test_same_shape() {
        assert_eq!(broadcast_shape(&[3, 4], &[3, 4]).unwrap(), vec![3, 4]);
    }

    #[test]
    fn test_scalar_broadcast() {
        assert_eq!(broadcast_shape(&[3, 4], &[]).unwrap(), vec![3, 4]);
    }

    #[test]
    fn test_1d_broadcast() {
        assert_eq!(broadcast_shape(&[3, 4], &[4]).unwrap(), vec![3, 4]);
    }

    #[test]
    fn test_2d_broadcast() {
        assert_eq!(broadcast_shape(&[3, 1], &[1, 4]).unwrap(), vec![3, 4]);
    }

    #[test]
    fn test_incompatible_shapes() {
        assert!(broadcast_shape(&[3, 4], &[3, 5]).is_err());
    }

    #[test]
    fn test_higher_rank() {
        assert_eq!(broadcast_shape(&[2, 1, 5], &[3, 1]).unwrap(), vec![2, 3, 5]);
    }

    #[test]
    fn test_broadcast_array_data_noop() {
        let a = NdArray::zeros(&[3, 4], DType::Float64);
        let b = broadcast_array_data(a.data(), &[3, 4]);
        assert_eq!(b.shape(), &[3, 4]);
    }

    #[test]
    fn test_broadcast_array_data_expand() {
        let a = NdArray::ones(&[4], DType::Float64);
        let b = broadcast_array_data(a.data(), &[3, 4]);
        assert_eq!(b.shape(), &[3, 4]);
    }
}
