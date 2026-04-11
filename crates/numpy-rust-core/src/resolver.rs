use crate::DType;

#[derive(Debug, Clone, Copy)]
pub enum BinaryOp {
    Add,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BinaryOpPlan {
    output_dtype: DType,
}

impl BinaryOpPlan {
    pub fn output_dtype(&self) -> DType {
        self.output_dtype
    }
}

pub fn resolve_binary_op(op: BinaryOp, lhs: DType, rhs: DType) -> crate::Result<BinaryOpPlan> {
    let _ = op;
    Ok(BinaryOpPlan {
        output_dtype: lhs.promote(rhs),
    })
}
