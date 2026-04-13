use crate::array::NdArray;
use crate::error::{NumpyError, Result};
use crate::indexing::Scalar;

/// A single named column in a structured array.
#[derive(Debug)]
pub struct FieldSpec {
    pub name: String,
    pub data: NdArray,
}

/// Columnar structured array: each field is a separate homogeneous column.
/// `shape` is the logical shape of the record array (not including fields).
#[derive(Debug)]
pub struct StructArrayData {
    pub fields: Vec<FieldSpec>,
    pub shape: Vec<usize>,
}

impl StructArrayData {
    pub fn new(fields: Vec<FieldSpec>, shape: Vec<usize>) -> Self {
        Self { fields, shape }
    }

    /// Get a column by field name.
    pub fn field(&self, name: &str) -> Option<&NdArray> {
        self.fields.iter().find(|f| f.name == name).map(|f| &f.data)
    }

    /// Get a mutable reference to a column by field name.
    pub fn field_mut(&mut self, name: &str) -> Option<&mut NdArray> {
        self.fields
            .iter_mut()
            .find(|f| f.name == name)
            .map(|f| &mut f.data)
    }

    /// Ordered list of field names.
    pub fn field_names(&self) -> Vec<&str> {
        self.fields.iter().map(|f| f.name.as_str()).collect()
    }

    /// Number of records (first dimension of shape, 0 if shape is empty).
    pub fn len(&self) -> usize {
        self.shape.first().copied().unwrap_or(0)
    }

    /// Returns true if the array has no records.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Extract one row as a Vec of Scalars (one per field, in field order).
    /// Supports negative indexing: -1 = last row.
    pub fn get_row(&self, idx: isize) -> Result<Vec<Scalar>> {
        let n = self.len();
        if n == 0 {
            return Err(NumpyError::ValueError(
                "index out of bounds: array has 0 rows".into(),
            ));
        }
        let actual_idx = if idx < 0 {
            let pos = n as isize + idx;
            if pos < 0 {
                return Err(NumpyError::ValueError(format!(
                    "index {} is out of bounds for axis 0 with size {}",
                    idx, n
                )));
            }
            pos as usize
        } else {
            let pos = idx as usize;
            if pos >= n {
                return Err(NumpyError::ValueError(format!(
                    "index {} is out of bounds for axis 0 with size {}",
                    idx, n
                )));
            }
            pos
        };
        let mut row = Vec::with_capacity(self.fields.len());
        for field in &self.fields {
            let scalar = field.data.get(&[actual_idx])?;
            row.push(scalar);
        }
        Ok(row)
    }

    /// Extract a contiguous range of rows [start, end) as a new StructArrayData.
    pub fn slice_rows(&self, start: usize, end: usize) -> Result<Self> {
        use crate::indexing::SliceArg;
        let n = self.len();
        if end > n {
            return Err(NumpyError::ValueError(format!(
                "slice_rows: end {} out of bounds for array with {} rows",
                end, n
            )));
        }
        if start > end {
            return Err(NumpyError::ValueError(format!(
                "slice_rows: start {} > end {}",
                start, end
            )));
        }
        let sliced_fields: Vec<FieldSpec> = self
            .fields
            .iter()
            .map(|f| {
                let sliced = f.data.slice(&[SliceArg::Range {
                    start: Some(start as isize),
                    stop: Some(end as isize),
                    step: 1,
                }])?;
                Ok(FieldSpec {
                    name: f.name.clone(),
                    data: sliced,
                })
            })
            .collect::<Result<Vec<_>>>()?;
        Ok(StructArrayData::new(sliced_fields, vec![end - start]))
    }

    /// Replace a column. New data must have the same shape as `self.shape`.
    pub fn set_field(&mut self, name: &str, data: NdArray) -> Result<()> {
        if data.shape() != self.shape.as_slice() {
            return Err(NumpyError::ValueError(format!(
                "shape mismatch: field data has shape {:?}, expected {:?}",
                data.shape(),
                self.shape
            )));
        }
        match self.fields.iter_mut().find(|f| f.name == name) {
            Some(field) => {
                field.data = data;
                Ok(())
            }
            None => Err(NumpyError::ValueError(format!("no field named '{}'", name))),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    fn make_int_col(vals: Vec<i64>) -> NdArray {
        NdArray::from_vec(vals)
    }

    fn make_float_col(vals: Vec<f64>) -> NdArray {
        NdArray::from_vec(vals)
    }

    #[test]
    fn test_field_access() {
        let sa = StructArrayData::new(
            vec![
                FieldSpec {
                    name: "x".into(),
                    data: make_int_col(vec![1, 2, 3]),
                },
                FieldSpec {
                    name: "y".into(),
                    data: make_float_col(vec![1.5, 2.5, 3.5]),
                },
            ],
            vec![3],
        );
        assert!(sa.field("x").is_some());
        assert!(sa.field("y").is_some());
        assert!(sa.field("z").is_none());
        assert_eq!(sa.field_names(), vec!["x", "y"]);
        assert_eq!(sa.len(), 3);
    }

    #[test]
    fn test_get_row() {
        let sa = StructArrayData::new(
            vec![
                FieldSpec {
                    name: "x".into(),
                    data: make_int_col(vec![10, 20, 30]),
                },
                FieldSpec {
                    name: "y".into(),
                    data: make_float_col(vec![1.1, 2.2, 3.3]),
                },
            ],
            vec![3],
        );
        let row = sa.get_row(0).unwrap();
        assert_eq!(row.len(), 2);
        assert!(matches!(row[0], Scalar::Int64(10)));
        let row = sa.get_row(-1).unwrap();
        assert!(matches!(row[0], Scalar::Int64(30)));
        assert!(sa.get_row(3).is_err());
        assert!(sa.get_row(-4).is_err());
    }

    #[test]
    fn test_slice_rows() {
        let sa = StructArrayData::new(
            vec![
                FieldSpec {
                    name: "x".into(),
                    data: make_int_col(vec![10, 20, 30, 40]),
                },
                FieldSpec {
                    name: "y".into(),
                    data: make_float_col(vec![1.1, 2.2, 3.3, 4.4]),
                },
            ],
            vec![4],
        );
        let sliced = sa.slice_rows(1, 3).unwrap();
        assert_eq!(sliced.len(), 2);
        assert_eq!(sliced.shape, vec![2]);
        let row0 = sliced.get_row(0).unwrap();
        assert!(matches!(row0[0], Scalar::Int64(20)));
        let row1 = sliced.get_row(1).unwrap();
        assert!(matches!(row1[0], Scalar::Int64(30)));
        // empty slice
        let empty = sa.slice_rows(2, 2).unwrap();
        assert_eq!(empty.len(), 0);
        // out of bounds
        assert!(sa.slice_rows(0, 5).is_err());
    }

    #[test]
    fn test_set_field() {
        let mut sa = StructArrayData::new(
            vec![FieldSpec {
                name: "x".into(),
                data: make_int_col(vec![1, 2, 3]),
            }],
            vec![3],
        );
        let new_col = make_int_col(vec![10, 20, 30]);
        sa.set_field("x", new_col).unwrap();
        let row = sa.get_row(0).unwrap();
        assert!(matches!(row[0], Scalar::Int64(10)));
        // wrong shape → error
        assert!(sa.set_field("x", make_int_col(vec![1, 2])).is_err());
        // unknown field → error
        assert!(sa.set_field("z", make_int_col(vec![1, 2, 3])).is_err());
    }
}
