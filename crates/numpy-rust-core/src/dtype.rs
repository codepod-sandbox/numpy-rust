/// Supported data types, mirroring NumPy's core numeric dtypes.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum DType {
    Bool,
    Int32,
    Int64,
    Float32,
    Float64,
}

impl DType {
    /// Returns the common type that both `self` and `other` can be promoted to,
    /// following NumPy's type promotion rules (simplified for 5 types).
    ///
    /// Promotion lattice:
    ///   Bool -> Int32 -> Int64 -> Float64
    ///                    Float32 -> Float64
    ///   i64 + f32 -> f64 (to avoid precision loss)
    pub fn promote(self, other: DType) -> DType {
        if self == other {
            return self;
        }
        let (hi, lo) = if self.rank() >= other.rank() {
            (self, other)
        } else {
            (other, self)
        };
        // Special case: mixing i64 with f32 promotes to f64 to avoid precision loss
        if (hi == DType::Float32 && lo == DType::Int64)
            || (hi == DType::Int64 && lo == DType::Float32)
        {
            return DType::Float64;
        }
        hi
    }

    /// Numeric rank for promotion ordering.
    fn rank(self) -> u8 {
        match self {
            DType::Bool => 0,
            DType::Int32 => 1,
            DType::Int64 => 2,
            DType::Float32 => 3,
            DType::Float64 => 4,
        }
    }

    /// Size in bytes of a single element.
    pub fn itemsize(self) -> usize {
        match self {
            DType::Bool => 1,
            DType::Int32 | DType::Float32 => 4,
            DType::Int64 | DType::Float64 => 8,
        }
    }

    /// Returns true if this is a floating-point type.
    pub fn is_float(self) -> bool {
        matches!(self, DType::Float32 | DType::Float64)
    }

    /// Returns true if this is an integer type.
    pub fn is_integer(self) -> bool {
        matches!(self, DType::Int32 | DType::Int64)
    }
}

impl std::fmt::Display for DType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            DType::Bool => write!(f, "bool"),
            DType::Int32 => write!(f, "int32"),
            DType::Int64 => write!(f, "int64"),
            DType::Float32 => write!(f, "float32"),
            DType::Float64 => write!(f, "float64"),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_promote_same_type() {
        assert_eq!(DType::Float64.promote(DType::Float64), DType::Float64);
        assert_eq!(DType::Int32.promote(DType::Int32), DType::Int32);
    }

    #[test]
    fn test_promote_bool_widens() {
        assert_eq!(DType::Bool.promote(DType::Int32), DType::Int32);
        assert_eq!(DType::Bool.promote(DType::Float64), DType::Float64);
    }

    #[test]
    fn test_promote_int_to_float() {
        assert_eq!(DType::Int32.promote(DType::Float32), DType::Float32);
        assert_eq!(DType::Int64.promote(DType::Float64), DType::Float64);
    }

    #[test]
    fn test_promote_int32_int64() {
        assert_eq!(DType::Int32.promote(DType::Int64), DType::Int64);
    }

    #[test]
    fn test_promote_mixed_int_float() {
        // i64 + f32 -> f64 (to avoid precision loss)
        assert_eq!(DType::Int64.promote(DType::Float32), DType::Float64);
    }

    #[test]
    fn test_promote_is_symmetric() {
        let pairs = [
            (DType::Int32, DType::Float64),
            (DType::Bool, DType::Int64),
            (DType::Float32, DType::Int32),
            (DType::Int64, DType::Float32),
        ];
        for (a, b) in pairs {
            assert_eq!(a.promote(b), b.promote(a), "promote({a:?}, {b:?}) not symmetric");
        }
    }

    #[test]
    fn test_itemsize() {
        assert_eq!(DType::Bool.itemsize(), 1);
        assert_eq!(DType::Int32.itemsize(), 4);
        assert_eq!(DType::Float64.itemsize(), 8);
    }

    #[test]
    fn test_is_float() {
        assert!(DType::Float32.is_float());
        assert!(DType::Float64.is_float());
        assert!(!DType::Int32.is_float());
        assert!(!DType::Bool.is_float());
    }

    #[test]
    fn test_display() {
        assert_eq!(format!("{}", DType::Float64), "float64");
        assert_eq!(format!("{}", DType::Bool), "bool");
    }
}
