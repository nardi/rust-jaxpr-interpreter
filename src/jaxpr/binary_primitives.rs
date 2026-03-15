use pyo3::prelude::*;

use super::{Atom, Var};

#[derive(Debug, FromPyObject)]
pub struct AddEqn<'py> {
    pub invars: [Atom<'py>; 2],
    pub outvars: [Var; 1],
}

impl<'py> AddEqn<'py> {
    pub const NAME: &'static str = "add";
}

#[derive(Debug, FromPyObject)]
pub struct MulEqn<'py> {
    pub invars: [Atom<'py>; 2],
    pub outvars: [Var; 1],
}

impl<'py> MulEqn<'py> {
    pub const NAME: &'static str = "mul";
}
