use numpy::{AllowTypeChange, PyArrayLikeDyn};
use pyo3::{prelude::*, types::PyDict};

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
    pub count: isize,
    pub aval: AbstractValue,
}

// Tries to call the __array__ function on a Python object, before converting it to the specified
// type.
// TODO: figure out why the generic version complains about lifetimes.
fn try_array_dunder<'a, 'py>(
    obj: &Bound<'py, PyAny>,
) -> Result<PyArrayLikeDyn<'py, f64, AllowTypeChange>, PyErr>
// where
//     T: FromPyObject<'a, 'py, Error = PyErr>,
//     'py: 'a,
{
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
    const NAME: &'static str = "integer_pow";
}

#[derive(Debug, FromPyObject)]
pub struct AddEqn<'py> {
    pub invars: [Atom<'py>; 2],
    pub outvars: [Var; 1],
}

impl<'py> AddEqn<'py> {
    const NAME: &'static str = "add";
}

#[derive(Debug, FromPyObject)]
pub struct MulEqn<'py> {
    pub invars: [Atom<'py>; 2],
    pub outvars: [Var; 1],
}

impl<'py> MulEqn<'py> {
    const NAME: &'static str = "mul";
}

#[derive(Debug)]
pub enum JaxprEqn<'py> {
    IntegerPow(IntegerPowEqn<'py>),
    Add(AddEqn<'py>),
    Mul(MulEqn<'py>),
    Unknown(UnknownEqn<'py>),
}

impl<'a, 'py> FromPyObject<'a, 'py> for JaxprEqn<'py> {
    type Error = PyErr;

    fn extract(obj: Borrowed<'a, 'py, PyAny>) -> PyResult<Self> {
        let primitive_name: String = obj.getattr("primitive")?.getattr("name")?.extract()?;

        Ok(match primitive_name.as_str() {
            IntegerPowEqn::NAME => JaxprEqn::IntegerPow(obj.extract()?),
            AddEqn::NAME => JaxprEqn::Add(obj.extract()?),
            MulEqn::NAME => JaxprEqn::Mul(obj.extract()?),
            _ => JaxprEqn::Unknown(obj.extract()?),
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
