use numpy::PyArrayLikeDyn;
use pyo3::prelude::*;

use crate::boxed_slice::BoxedSlice;

#[derive(Debug, FromPyObject, Eq, PartialEq, Hash, Clone)]
pub struct ShapedArray {
    pub shape: BoxedSlice<i32>,
}

#[derive(Debug, FromPyObject, Eq, PartialEq, Hash, Clone)]
pub enum AbstractValue {
    ShapedArray(ShapedArray),
}

#[derive(Debug, FromPyObject, Eq, PartialEq, Hash, Clone)]
pub struct Var {
    pub count: i64,
    pub aval: AbstractValue,
}

#[allow(dead_code)]
#[derive(Debug, FromPyObject)]
pub struct Literal<'py> {
    pub val: PyArrayLikeDyn<'py, f64>,
    pub aval: AbstractValue,
}

#[allow(dead_code)]
#[derive(Debug, FromPyObject)]
pub enum Atom<'py> {
    Var(Var),
    Literal(Literal<'py>),
}

#[derive(Debug, FromPyObject, Eq, PartialEq)]
#[pyo3(from_item_all)]
pub struct IntegerPowParams {
    pub y: i32,
}

#[allow(dead_code)]
#[derive(Debug, FromPyObject)]
pub struct IntegerPowEqn<'py> {
    pub invars: [Atom<'py>; 1],
    pub outvars: [Var; 1],
    pub params: IntegerPowParams,
}

impl<'py> IntegerPowEqn<'py> {
    const NAME: &'static str = "integer_pow";
}

#[derive(Debug)]
pub enum JaxprEqn<'py> {
    IntegerPow(IntegerPowEqn<'py>),
}

impl<'a, 'py> FromPyObject<'a, 'py> for JaxprEqn<'py> {
    type Error = PyErr;

    fn extract(obj: Borrowed<'a, 'py, PyAny>) -> PyResult<Self> {
        let primitive_name: String = obj.getattr("primitive")?.getattr("name")?.extract()?;

        Ok(match primitive_name.as_str() {
            IntegerPowEqn::NAME => JaxprEqn::IntegerPow(obj.extract()?),
            _ => todo!(),
        })
    }
}

#[allow(dead_code)]
#[derive(Debug, FromPyObject)]
pub struct Jaxpr<'py> {
    pub constvars: BoxedSlice<Var>,
    pub invars: BoxedSlice<Var>,
    pub outvars: BoxedSlice<Atom<'py>>,
    pub eqns: BoxedSlice<JaxprEqn<'py>>,
}
