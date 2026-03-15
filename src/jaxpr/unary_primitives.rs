use pyo3::prelude::*;

use super::Var;

#[derive(Debug, FromPyObject, Eq, PartialEq)]
#[pyo3(from_item_all)]
pub struct IntegerPowParams {
    pub y: i32,
}

#[derive(Debug, FromPyObject)]
pub struct IntegerPowEqn {
    pub invars: [Var; 1],
    pub outvars: [Var; 1],
    pub params: IntegerPowParams,
}

impl IntegerPowEqn {
    pub const NAME: &'static str = "integer_pow";
}

#[derive(Debug, FromPyObject)]
pub struct SinEqn {
    pub invars: [Var; 1],
    pub outvars: [Var; 1],
}

impl SinEqn {
    pub const NAME: &'static str = "sin";
}
