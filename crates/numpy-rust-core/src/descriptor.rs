use crate::kernel::{
    binary_kernel_for_dtype, comparison_kernel_for_dtype, sum_all_kernel_for_dtype,
    sum_axis_kernel_for_dtype, ArithmeticKernelOp, BinaryArrayKernel, ComparisonArrayKernel,
    ComparisonKernelOp, ReduceAllArrayKernel, ReduceAxisArrayKernel,
};
use crate::resolver::{resolve_binary_op, resolve_cast, BinaryOp, CastPlan, CastingRule};
use crate::{DType, NumpyError, Result};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DTypeKind {
    Bool,
    SignedInteger,
    UnsignedInteger,
    Float,
    Complex,
    String,
}

#[derive(Debug)]
pub struct DTypeDescriptor {
    pub id: DType,
    pub name: &'static str,
    pub itemsize: usize,
    pub kind: DTypeKind,
}

impl DTypeDescriptor {
    pub fn name(&self) -> &'static str {
        self.name
    }

    pub fn itemsize(&self) -> usize {
        self.itemsize
    }

    pub fn kind(&self) -> DTypeKind {
        self.kind
    }

    pub fn binary_kernel(&self, op: ArithmeticKernelOp) -> Option<BinaryArrayKernel> {
        binary_kernel_for_dtype(self.id, op)
    }

    pub fn comparison_kernel(&self, op: ComparisonKernelOp) -> Option<ComparisonArrayKernel> {
        comparison_kernel_for_dtype(self.id, op)
    }

    pub fn sum_all_kernel(&self) -> Option<ReduceAllArrayKernel> {
        sum_all_kernel_for_dtype(self.id)
    }

    pub fn sum_axis_kernel(&self) -> Option<ReduceAxisArrayKernel> {
        sum_axis_kernel_for_dtype(self.id)
    }

    pub fn sum_plan(&self) -> Result<ReductionPlan> {
        if matches!(self.kind, DTypeKind::String) {
            return Err(NumpyError::TypeError(
                "sum not supported for string arrays".into(),
            ));
        }

        let add_plan = resolve_binary_op(BinaryOp::Add, self.id, self.id)?;
        let result_dtype = add_plan.result_storage_dtype();
        let input_cast = resolve_cast(self.id, result_dtype, CastingRule::Unsafe)?;

        Ok(ReductionPlan {
            input_cast,
            result_dtype,
        })
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ReductionPlan {
    input_cast: CastPlan,
    result_dtype: DType,
}

impl ReductionPlan {
    pub fn input_cast(&self) -> CastPlan {
        self.input_cast
    }

    pub fn result_dtype(&self) -> DType {
        self.result_dtype
    }

    pub fn result_storage_dtype(&self) -> DType {
        self.result_dtype.storage_dtype()
    }
}

static BOOL_DESCRIPTOR: DTypeDescriptor = DTypeDescriptor {
    id: DType::Bool,
    name: "bool",
    itemsize: 1,
    kind: DTypeKind::Bool,
};

static INT8_DESCRIPTOR: DTypeDescriptor = DTypeDescriptor {
    id: DType::Int8,
    name: "int8",
    itemsize: 1,
    kind: DTypeKind::SignedInteger,
};

static INT16_DESCRIPTOR: DTypeDescriptor = DTypeDescriptor {
    id: DType::Int16,
    name: "int16",
    itemsize: 2,
    kind: DTypeKind::SignedInteger,
};

static INT32_DESCRIPTOR: DTypeDescriptor = DTypeDescriptor {
    id: DType::Int32,
    name: "int32",
    itemsize: 4,
    kind: DTypeKind::SignedInteger,
};

static INT64_DESCRIPTOR: DTypeDescriptor = DTypeDescriptor {
    id: DType::Int64,
    name: "int64",
    itemsize: 8,
    kind: DTypeKind::SignedInteger,
};

static UINT8_DESCRIPTOR: DTypeDescriptor = DTypeDescriptor {
    id: DType::UInt8,
    name: "uint8",
    itemsize: 1,
    kind: DTypeKind::UnsignedInteger,
};

static UINT16_DESCRIPTOR: DTypeDescriptor = DTypeDescriptor {
    id: DType::UInt16,
    name: "uint16",
    itemsize: 2,
    kind: DTypeKind::UnsignedInteger,
};

static UINT32_DESCRIPTOR: DTypeDescriptor = DTypeDescriptor {
    id: DType::UInt32,
    name: "uint32",
    itemsize: 4,
    kind: DTypeKind::UnsignedInteger,
};

static UINT64_DESCRIPTOR: DTypeDescriptor = DTypeDescriptor {
    id: DType::UInt64,
    name: "uint64",
    itemsize: 8,
    kind: DTypeKind::UnsignedInteger,
};

static FLOAT16_DESCRIPTOR: DTypeDescriptor = DTypeDescriptor {
    id: DType::Float16,
    name: "float16",
    itemsize: 2,
    kind: DTypeKind::Float,
};

static FLOAT32_DESCRIPTOR: DTypeDescriptor = DTypeDescriptor {
    id: DType::Float32,
    name: "float32",
    itemsize: 4,
    kind: DTypeKind::Float,
};

static FLOAT64_DESCRIPTOR: DTypeDescriptor = DTypeDescriptor {
    id: DType::Float64,
    name: "float64",
    itemsize: 8,
    kind: DTypeKind::Float,
};

static COMPLEX64_DESCRIPTOR: DTypeDescriptor = DTypeDescriptor {
    id: DType::Complex64,
    name: "complex64",
    itemsize: 8,
    kind: DTypeKind::Complex,
};

static COMPLEX128_DESCRIPTOR: DTypeDescriptor = DTypeDescriptor {
    id: DType::Complex128,
    name: "complex128",
    itemsize: 16,
    kind: DTypeKind::Complex,
};

static STR_DESCRIPTOR: DTypeDescriptor = DTypeDescriptor {
    id: DType::Str,
    name: "str",
    itemsize: 0,
    kind: DTypeKind::String,
};

pub fn descriptor_for_dtype(dtype: DType) -> &'static DTypeDescriptor {
    match dtype {
        DType::Bool => &BOOL_DESCRIPTOR,
        DType::Int8 => &INT8_DESCRIPTOR,
        DType::Int16 => &INT16_DESCRIPTOR,
        DType::Int32 => &INT32_DESCRIPTOR,
        DType::Int64 => &INT64_DESCRIPTOR,
        DType::UInt8 => &UINT8_DESCRIPTOR,
        DType::UInt16 => &UINT16_DESCRIPTOR,
        DType::UInt32 => &UINT32_DESCRIPTOR,
        DType::UInt64 => &UINT64_DESCRIPTOR,
        DType::Float16 => &FLOAT16_DESCRIPTOR,
        DType::Float32 => &FLOAT32_DESCRIPTOR,
        DType::Float64 => &FLOAT64_DESCRIPTOR,
        DType::Complex64 => &COMPLEX64_DESCRIPTOR,
        DType::Complex128 => &COMPLEX128_DESCRIPTOR,
        DType::Str => &STR_DESCRIPTOR,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kernel::ArithmeticKernelOp;

    #[test]
    fn test_sum_plan_bool_uses_int32_accumulator() {
        let plan = descriptor_for_dtype(DType::Bool).sum_plan().unwrap();
        assert_eq!(plan.input_cast().target_storage_dtype(), DType::Int32);
        assert_eq!(plan.result_dtype(), DType::Int32);
    }

    #[test]
    fn test_sum_plan_int32_preserves_int32_result() {
        let plan = descriptor_for_dtype(DType::Int32).sum_plan().unwrap();
        assert_eq!(plan.input_cast().target_storage_dtype(), DType::Int32);
        assert_eq!(plan.result_dtype(), DType::Int32);
    }

    #[test]
    fn test_float64_descriptor_registers_add_kernel() {
        assert!(descriptor_for_dtype(DType::Float64)
            .binary_kernel(ArithmeticKernelOp::Add)
            .is_some());
    }

    #[test]
    fn test_float64_descriptor_does_not_register_sub_kernel_yet() {
        assert!(descriptor_for_dtype(DType::Float64)
            .binary_kernel(ArithmeticKernelOp::Sub)
            .is_none());
    }
}
