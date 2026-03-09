use std::ops::Deref;

use pyo3::prelude::*;

#[derive(Debug, Eq, PartialEq, Hash, Clone)]
pub struct BoxedSlice<T>(Box<[T]>);

impl<T> Deref for BoxedSlice<T> {
    type Target = [T];

    fn deref(&self) -> &Self::Target {
        self.0.deref()
    }
}

impl<'a, 'py, T> FromPyObject<'a, 'py> for BoxedSlice<T>
where
    Vec<T>: pyo3::FromPyObject<'a, 'py>,
{
    type Error = <Vec<T> as pyo3::FromPyObject<'a, 'py>>::Error;

    fn extract(obj: Borrowed<'a, 'py, PyAny>) -> Result<Self, Self::Error> {
        Ok(BoxedSlice(obj.extract::<Vec<T>>()?.into_boxed_slice()))
    }
}
