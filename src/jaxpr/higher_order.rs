use pyo3::{exceptions::PyTypeError, prelude::*};

use crate::boxed_slice::BoxedSlice;

use super::{ClosedJaxpr, Var};

#[derive(Debug)]
pub struct HigherOrderJaxprEqn<'py> {
    pub invars: BoxedSlice<Var>,
    pub outvars: BoxedSlice<Var>,
    pub params: ClosedJaxpr<'py>,
}

impl<'a, 'py> FromPyObject<'a, 'py> for HigherOrderJaxprEqn<'py> {
    type Error = PyErr;

    fn extract(obj: Borrowed<'a, 'py, PyAny>) -> PyResult<Self> {
        let primitive_name: String = obj.getattr("primitive")?.getattr("name")?.extract()?;
        let params = obj.getattr("params")?;

        let inner_closed_jaxpr = match primitive_name.as_str() {
            "jit" => params.get_item("jaxpr")?,
            "custom_jvp_call" => params.get_item("call_jaxpr")?,
            _ => Err(PyTypeError::new_err(format!(
                "Primitive `{primitive_name}` is not a known higher-order primitive"
            )))?,
        };

        Ok(HigherOrderJaxprEqn {
            invars: obj.getattr("invars")?.extract()?,
            outvars: obj.getattr("outvars")?.extract()?,
            params: inner_closed_jaxpr.extract()?,
        })
    }
}
