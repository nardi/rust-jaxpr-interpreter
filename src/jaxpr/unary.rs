use enum_dispatch::enum_dispatch;
use pyo3::{exceptions::PyTypeError, prelude::*};

use super::Var;

// These imports are necessary for enum_dispatch to work. TODO: figure out why, it kind of ruins the
// separation of the IR and the interpreter.
use crate::interpreter::unary::EvalUnaryJaxprPrimitive;
use ndarray::{ArrayD, ArrayViewD};

#[derive(Debug, FromPyObject, Eq, PartialEq)]
#[pyo3(from_item_all)]
pub struct IntegerPowPrimitive {
    pub y: i32,
}

impl IntegerPowPrimitive {
    pub const NAME: &'static str = "integer_pow";
}

#[derive(Debug, Eq, PartialEq)]
pub struct SinPrimitive;

impl SinPrimitive {
    pub const NAME: &'static str = "sin";
}

#[derive(Debug)]
#[enum_dispatch]
pub enum UnaryJaxprPrimitive {
    IntegerPow(IntegerPowPrimitive),
    Sin(SinPrimitive),
}
#[derive(Debug)]
pub struct UnaryJaxprEqn {
    pub invars: [Var; 1],
    pub outvars: [Var; 1],
    pub primitive: UnaryJaxprPrimitive,
}

impl<'a, 'py> FromPyObject<'a, 'py> for UnaryJaxprEqn {
    type Error = PyErr;

    fn extract(obj: Borrowed<'a, 'py, PyAny>) -> PyResult<Self> {
        let primitive_name: String = obj.getattr("primitive")?.getattr("name")?.extract()?;
        let params = obj.getattr("params")?;

        let primitive = match primitive_name.as_str() {
            IntegerPowPrimitive::NAME => UnaryJaxprPrimitive::IntegerPow(params.extract()?),
            SinPrimitive::NAME => UnaryJaxprPrimitive::Sin(SinPrimitive),
            _ => Err(PyTypeError::new_err(format!(
                "Primitive `{primitive_name}` is not a known unary primitive"
            )))?,
        };

        Ok(UnaryJaxprEqn {
            invars: obj.getattr("invars")?.extract()?,
            outvars: obj.getattr("outvars")?.extract()?,
            primitive: primitive,
        })
    }
}
