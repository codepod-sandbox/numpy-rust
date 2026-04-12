use crate::kernel::{
    arg_reduction_all_kernel_for_dtype, arg_reduction_axis_kernel_for_dtype,
    binary_kernel_for_dtype, bitwise_binary_kernel_for_dtype, bitwise_unary_kernel_for_dtype,
    comparison_kernel_for_dtype, decompose_unary_kernel_for_dtype, dot_kernel_for_dtype,
    math_binary_kernel_for_dtype, math_unary_kernel_for_dtype, predicate_kernel_for_dtype,
    predicate_presence_kernel_for_dtype, real_binary_kernel_for_dtype, real_unary_kernel_for_dtype,
    reduction_all_kernel_for_dtype, reduction_axis_kernel_for_dtype, truth_kernel_for_dtype,
    truth_reduce_kernel_for_dtype, value_unary_kernel_for_dtype, where_kernel_for_dtype,
    ArgReduceAllKernel, ArgReduceAxisKernel, ArgReductionKernelOp, ArithmeticKernelOp,
    BinaryArrayKernel, BinaryMathArrayKernel, BitwiseBinaryArrayKernel, BitwiseBinaryKernelOp,
    BitwiseUnaryArrayKernel, BitwiseUnaryKernelOp, ComparisonArrayKernel, ComparisonKernelOp,
    DecomposeUnaryArrayKernel, DecomposeUnaryKernelOp, DotArrayKernel, DotKernelOp,
    MathBinaryKernelOp, MathUnaryKernelOp, PredicateArrayKernel, PredicateKernelOp,
    PredicatePresenceKernel, PredicatePresenceOp, RealBinaryArrayKernel, RealBinaryKernelOp,
    RealUnaryKernelOp, ReduceAllArrayKernel, ReduceAxisArrayKernel, ReductionKernelOp,
    TruthArrayKernel, TruthKernelOp, TruthReduceKernel, TruthReduceKernelOp, UnaryArrayKernel,
    ValueUnaryKernelOp, WhereArrayKernel, WhereKernelOp,
};
use crate::resolver::{resolve_reduction_op, ReductionOp, ReductionPlan};
use crate::DType;

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

    pub fn dot_kernel(&self, op: DotKernelOp) -> Option<DotArrayKernel> {
        dot_kernel_for_dtype(self.id, op)
    }

    pub fn where_kernel(&self, op: WhereKernelOp) -> Option<WhereArrayKernel> {
        where_kernel_for_dtype(self.id, op)
    }

    pub fn predicate_kernel(&self, op: PredicateKernelOp) -> Option<PredicateArrayKernel> {
        predicate_kernel_for_dtype(self.id, op)
    }

    pub fn predicate_presence_kernel(
        &self,
        op: PredicatePresenceOp,
    ) -> Option<PredicatePresenceKernel> {
        predicate_presence_kernel_for_dtype(self.id, op)
    }

    pub fn truth_kernel(&self, op: TruthKernelOp) -> Option<TruthArrayKernel> {
        truth_kernel_for_dtype(self.id, op)
    }

    pub fn truth_reduce_kernel(&self, op: TruthReduceKernelOp) -> Option<TruthReduceKernel> {
        truth_reduce_kernel_for_dtype(self.id, op)
    }

    pub fn bitwise_unary_kernel(
        &self,
        op: BitwiseUnaryKernelOp,
    ) -> Option<BitwiseUnaryArrayKernel> {
        bitwise_unary_kernel_for_dtype(self.id, op)
    }

    pub fn bitwise_binary_kernel(
        &self,
        op: BitwiseBinaryKernelOp,
    ) -> Option<BitwiseBinaryArrayKernel> {
        bitwise_binary_kernel_for_dtype(self.id, op)
    }

    pub fn math_unary_kernel(&self, op: MathUnaryKernelOp) -> Option<UnaryArrayKernel> {
        math_unary_kernel_for_dtype(self.id, op)
    }

    pub fn value_unary_kernel(&self, op: ValueUnaryKernelOp) -> Option<UnaryArrayKernel> {
        value_unary_kernel_for_dtype(self.id, op)
    }

    pub fn real_unary_kernel(&self, op: RealUnaryKernelOp) -> Option<UnaryArrayKernel> {
        real_unary_kernel_for_dtype(self.id, op)
    }

    pub fn real_binary_kernel(&self, op: RealBinaryKernelOp) -> Option<RealBinaryArrayKernel> {
        real_binary_kernel_for_dtype(self.id, op)
    }

    pub fn decompose_unary_kernel(
        &self,
        op: DecomposeUnaryKernelOp,
    ) -> Option<DecomposeUnaryArrayKernel> {
        decompose_unary_kernel_for_dtype(self.id, op)
    }

    pub fn math_binary_kernel(&self, op: MathBinaryKernelOp) -> Option<BinaryMathArrayKernel> {
        math_binary_kernel_for_dtype(self.id, op)
    }

    pub fn reduction_all_kernel(&self, op: ReductionKernelOp) -> Option<ReduceAllArrayKernel> {
        reduction_all_kernel_for_dtype(self.id, op)
    }

    pub fn reduction_axis_kernel(&self, op: ReductionKernelOp) -> Option<ReduceAxisArrayKernel> {
        reduction_axis_kernel_for_dtype(self.id, op)
    }

    pub fn arg_reduction_all_kernel(&self, op: ArgReductionKernelOp) -> Option<ArgReduceAllKernel> {
        arg_reduction_all_kernel_for_dtype(self.id, op)
    }

    pub fn arg_reduction_axis_kernel(
        &self,
        op: ArgReductionKernelOp,
    ) -> Option<ArgReduceAxisKernel> {
        arg_reduction_axis_kernel_for_dtype(self.id, op)
    }

    pub fn reduction_plan(&self, op: ReductionOp) -> crate::Result<ReductionPlan> {
        resolve_reduction_op(op, self.id)
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
        let plan = descriptor_for_dtype(DType::Bool)
            .reduction_plan(ReductionOp::Sum)
            .unwrap();
        assert_eq!(plan.input_cast().target_storage_dtype(), DType::Int32);
        assert_eq!(plan.result_dtype(), DType::Int32);
    }

    #[test]
    fn test_sum_plan_int32_preserves_int32_result() {
        let plan = descriptor_for_dtype(DType::Int32)
            .reduction_plan(ReductionOp::Sum)
            .unwrap();
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
