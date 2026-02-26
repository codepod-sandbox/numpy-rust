use ndarray::ArrayD;

use crate::array_data::ArrayData;
use crate::error::{NumpyError, Result};
use crate::NdArray;

/// Helper: ensure the array is of string dtype.
fn require_string(arr: &NdArray) -> Result<&ArrayD<String>> {
    match &arr.data {
        ArrayData::Str(a) => Ok(a),
        _ => Err(NumpyError::TypeError(
            "string operation requires string array".into(),
        )),
    }
}

impl NdArray {
    /// Convert each string element to uppercase.
    pub fn str_upper(&self) -> Result<NdArray> {
        let a = require_string(self)?;
        Ok(NdArray::from_data(ArrayData::Str(
            a.mapv(|s| s.to_uppercase()),
        )))
    }

    /// Convert each string element to lowercase.
    pub fn str_lower(&self) -> Result<NdArray> {
        let a = require_string(self)?;
        Ok(NdArray::from_data(ArrayData::Str(
            a.mapv(|s| s.to_lowercase()),
        )))
    }

    /// Capitalize the first character of each string element.
    pub fn str_capitalize(&self) -> Result<NdArray> {
        let a = require_string(self)?;
        Ok(NdArray::from_data(ArrayData::Str(a.mapv(|s| {
            let mut chars = s.chars();
            match chars.next() {
                None => String::new(),
                Some(first) => {
                    let upper: String = first.to_uppercase().collect();
                    upper + &chars.as_str().to_lowercase()
                }
            }
        }))))
    }

    /// Strip leading and trailing whitespace from each element.
    pub fn str_strip(&self) -> Result<NdArray> {
        let a = require_string(self)?;
        Ok(NdArray::from_data(ArrayData::Str(
            a.mapv(|s| s.trim().to_string()),
        )))
    }

    /// Return the length of each string element as an Int64 array.
    pub fn str_len(&self) -> Result<NdArray> {
        let a = require_string(self)?;
        Ok(NdArray::from_data(ArrayData::Int64(
            a.mapv(|s| s.chars().count() as i64),
        )))
    }

    /// Test whether each string element starts with the given prefix, returning a Bool array.
    pub fn str_startswith(&self, prefix: &str) -> Result<NdArray> {
        let a = require_string(self)?;
        let prefix = prefix.to_string();
        Ok(NdArray::from_data(ArrayData::Bool(
            a.mapv(|s| s.starts_with(prefix.as_str())),
        )))
    }

    /// Test whether each string element ends with the given suffix, returning a Bool array.
    pub fn str_endswith(&self, suffix: &str) -> Result<NdArray> {
        let a = require_string(self)?;
        let suffix = suffix.to_string();
        Ok(NdArray::from_data(ArrayData::Bool(
            a.mapv(|s| s.ends_with(suffix.as_str())),
        )))
    }

    /// Replace occurrences of `old` with `new` in each string element.
    pub fn str_replace(&self, old: &str, new: &str) -> Result<NdArray> {
        let a = require_string(self)?;
        let old = old.to_string();
        let new = new.to_string();
        Ok(NdArray::from_data(ArrayData::Str(
            a.mapv(|s| s.replace(old.as_str(), new.as_str())),
        )))
    }
}

#[cfg(test)]
mod tests {
    use crate::{DType, NdArray};

    #[test]
    fn test_str_upper() {
        let a = NdArray::from_vec(vec!["hello".to_string(), "world".to_string()]);
        let b = a.str_upper().unwrap();
        assert_eq!(b.dtype(), DType::Str);
        if let crate::ArrayData::Str(arr) = b.data() {
            assert_eq!(arr[[0]], "HELLO");
            assert_eq!(arr[[1]], "WORLD");
        } else {
            panic!("expected Str");
        }
    }

    #[test]
    fn test_str_lower() {
        let a = NdArray::from_vec(vec!["HELLO".to_string(), "WORLD".to_string()]);
        let b = a.str_lower().unwrap();
        assert_eq!(b.dtype(), DType::Str);
        if let crate::ArrayData::Str(arr) = b.data() {
            assert_eq!(arr[[0]], "hello");
            assert_eq!(arr[[1]], "world");
        } else {
            panic!("expected Str");
        }
    }

    #[test]
    fn test_str_capitalize() {
        let a = NdArray::from_vec(vec!["hello world".to_string(), "fOO".to_string()]);
        let b = a.str_capitalize().unwrap();
        if let crate::ArrayData::Str(arr) = b.data() {
            assert_eq!(arr[[0]], "Hello world");
            assert_eq!(arr[[1]], "Foo");
        } else {
            panic!("expected Str");
        }
    }

    #[test]
    fn test_str_strip() {
        let a = NdArray::from_vec(vec!["  hello  ".to_string(), "\tworld\n".to_string()]);
        let b = a.str_strip().unwrap();
        if let crate::ArrayData::Str(arr) = b.data() {
            assert_eq!(arr[[0]], "hello");
            assert_eq!(arr[[1]], "world");
        } else {
            panic!("expected Str");
        }
    }

    #[test]
    fn test_str_len() {
        let a = NdArray::from_vec(vec!["hi".to_string(), "hello".to_string()]);
        let b = a.str_len().unwrap();
        assert_eq!(b.dtype(), DType::Int64);
        if let crate::ArrayData::Int64(arr) = b.data() {
            assert_eq!(arr[[0]], 2);
            assert_eq!(arr[[1]], 5);
        } else {
            panic!("expected Int64");
        }
    }

    #[test]
    fn test_str_startswith_endswith() {
        let a = NdArray::from_vec(vec!["hello".to_string(), "world".to_string()]);
        let sw = a.str_startswith("he").unwrap();
        assert_eq!(sw.dtype(), DType::Bool);
        if let crate::ArrayData::Bool(arr) = sw.data() {
            assert!(arr[[0]]);
            assert!(!arr[[1]]);
        } else {
            panic!("expected Bool");
        }

        let ew = a.str_endswith("ld").unwrap();
        if let crate::ArrayData::Bool(arr) = ew.data() {
            assert!(!arr[[0]]);
            assert!(arr[[1]]);
        } else {
            panic!("expected Bool");
        }
    }

    #[test]
    fn test_str_replace() {
        let a = NdArray::from_vec(vec!["hello world".to_string(), "foo bar".to_string()]);
        let b = a.str_replace("o", "0").unwrap();
        if let crate::ArrayData::Str(arr) = b.data() {
            assert_eq!(arr[[0]], "hell0 w0rld");
            assert_eq!(arr[[1]], "f00 bar");
        } else {
            panic!("expected Str");
        }
    }

    #[test]
    fn test_non_string_error() {
        let a = NdArray::from_vec(vec![1.0_f64, 2.0, 3.0]);
        assert!(a.str_upper().is_err());
        assert!(a.str_lower().is_err());
        assert!(a.str_capitalize().is_err());
        assert!(a.str_strip().is_err());
        assert!(a.str_len().is_err());
        assert!(a.str_startswith("x").is_err());
        assert!(a.str_endswith("x").is_err());
        assert!(a.str_replace("a", "b").is_err());
    }
}
