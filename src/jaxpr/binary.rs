use enum_dispatch::enum_dispatch;
use pyo3::{exceptions::PyTypeError, prelude::*};

use super::{Atom, Var};

// These imports are necessary for enum_dispatch to work. TODO: figure out why, it kind of ruins the
// separation of the IR and the interpreter.
use crate::interpreter::binary::EvalBinaryJaxprPrimitive;
use ndarray::{ArrayD, ArrayViewD};

#[derive(Debug, Eq, PartialEq)]
pub struct AddPrimitive;

impl AddPrimitive {
    pub const NAME: &'static str = "add";
}

#[derive(Debug, Eq, PartialEq)]
pub struct MulPrimitive;

impl MulPrimitive {
    pub const NAME: &'static str = "mul";
}

#[derive(Debug)]
#[enum_dispatch]
pub enum BinaryJaxprPrimitive {
    Add(AddPrimitive),
    Mul(MulPrimitive),
}
#[derive(Debug)]
pub struct BinaryJaxprEqn<'py> {
    pub invars: [Atom<'py>; 2],
    pub outvars: [Var; 1],
    pub primitive: BinaryJaxprPrimitive,
}

impl<'a, 'py> FromPyObject<'a, 'py> for BinaryJaxprEqn<'py> {
    type Error = PyErr;

    fn extract(obj: Borrowed<'a, 'py, PyAny>) -> PyResult<Self> {
        let primitive_name: String = obj.getattr("primitive")?.getattr("name")?.extract()?;
        let _params = obj.getattr("params")?;

        let primitive = match primitive_name.as_str() {
            AddPrimitive::NAME => BinaryJaxprPrimitive::Add(AddPrimitive),
            MulPrimitive::NAME => BinaryJaxprPrimitive::Mul(MulPrimitive),
            _ => Err(PyTypeError::new_err(format!(
                "Primitive `{primitive_name}` is not a known binary primitive"
            )))?,
        };

        Ok(BinaryJaxprEqn {
            invars: obj.getattr("invars")?.extract()?,
            outvars: obj.getattr("outvars")?.extract()?,
            primitive: primitive,
        })
    }
}
