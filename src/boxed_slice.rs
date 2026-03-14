use std::ops::Deref;

use pyo3::prelude::*;

// A simple newtype wrapper around Box<[T]>, in order to implement FromPyObject for it.
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
    Vec<T>: FromPyObject<'a, 'py>,
{
    type Error = <Vec<T> as FromPyObject<'a, 'py>>::Error;

    fn extract(obj: Borrowed<'a, 'py, PyAny>) -> Result<Self, Self::Error> {
        Ok(BoxedSlice(obj.extract::<Vec<T>>()?.into_boxed_slice()))
    }
}
