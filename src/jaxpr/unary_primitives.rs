use pyo3::prelude::*;

use super::{Atom, Var};

#[derive(Debug, FromPyObject, Eq, PartialEq)]
#[pyo3(from_item_all)]
pub struct IntegerPowParams {
    pub y: i32,
}

#[derive(Debug, FromPyObject)]
pub struct IntegerPowEqn<'py> {
    pub invars: [Atom<'py>; 1],
    pub outvars: [Var; 1],
    pub params: IntegerPowParams,
}

impl<'py> IntegerPowEqn<'py> {
    pub const NAME: &'static str = "integer_pow";
}
