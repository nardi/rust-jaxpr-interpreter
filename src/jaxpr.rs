use enum_dispatch::enum_dispatch;
use numpy::{AllowTypeChange, PyArrayLikeDyn};
use pyo3::{prelude::*, types::PyDict};

use crate::boxed_slice::BoxedSlice;

// These imports are necessary for enum_dispatch to work. TODO: figure out why, it kind of ruins the
// separation of the IR and the interpreter.
use crate::interpreter::{EvalJaxprEqn, Interpreter};

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
    pub count: isize,
    pub aval: AbstractValue,
}

// Tries to call the __array__ function on a Python object, before converting it to the specified
// type.
// TODO: figure out why the generic version complains about lifetimes.
fn try_array_dunder<'a, 'py>(
    obj: &Bound<'py, PyAny>,
) -> Result<PyArrayLikeDyn<'py, f64, AllowTypeChange>, PyErr> {
    obj.getattr("__array__")
        .and_then(|attr| attr.call0())
        .and_then(|arr| arr.extract::<PyArrayLikeDyn<'py, f64, AllowTypeChange>>())
        .or_else(|_| obj.extract::<PyArrayLikeDyn<'py, f64, AllowTypeChange>>())
}

#[allow(dead_code)]
#[derive(Debug, FromPyObject)]
pub struct Literal<'py> {
    // JAX sometimes provides custom TypedNdArray objects that act as empty iterators, which leads
    // to incorrect conversion. Using `try_array_dunder` we convert it to a Numpy array before
    // calling `extract`.
    #[pyo3(from_py_with = try_array_dunder)]
    pub val: PyArrayLikeDyn<'py, f64, AllowTypeChange>,
    pub aval: AbstractValue,
}

#[allow(dead_code)]
#[derive(Debug, FromPyObject)]
pub enum Atom<'py> {
    Var(Var),
    Literal(Literal<'py>),
}

#[derive(Debug, FromPyObject)]
pub struct UnknownPrimitive {
    pub name: String,
}
#[allow(dead_code)]
#[derive(Debug, FromPyObject)]
pub struct UnknownEqn<'py> {
    pub invars: BoxedSlice<Atom<'py>>,
    pub outvars: BoxedSlice<Var>,
    pub primitive: UnknownPrimitive,
    pub params: Bound<'py, PyDict>,
}

pub mod binary_primitives;
pub mod unary_primitives;

use binary_primitives::BinaryJaxprEqn;
use unary_primitives::UnaryJaxprEqn;

#[derive(Debug, FromPyObject)]
#[enum_dispatch]
pub enum JaxprEqn<'py> {
    Unary(UnaryJaxprEqn),
    Binary(BinaryJaxprEqn<'py>),
    Unknown(UnknownEqn<'py>),
}

#[allow(dead_code)]
#[derive(Debug, FromPyObject)]
pub struct Jaxpr<'py> {
    pub constvars: BoxedSlice<Var>,
    pub invars: BoxedSlice<Var>,
    pub outvars: BoxedSlice<Atom<'py>>,
    pub eqns: BoxedSlice<JaxprEqn<'py>>,
}
