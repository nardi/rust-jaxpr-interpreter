use ndarray::{ArrayD, ArrayViewD};
use pyo3::PyErr;

use crate::jaxpr::JaxprEqn;

#[derive(Debug)]
pub struct Interpreter;

impl Interpreter {
    pub fn eval_eqn(
        &self,
        eqn: &JaxprEqn,
        invals: &[ArrayViewD<f64>],
    ) -> Result<Vec<ArrayD<f64>>, PyErr> {
        match eqn {
            JaxprEqn::IntegerPow(integer_pow) => Ok(invals
                .iter()
                .map(|a| a.powi(integer_pow.params.y))
                .collect::<Vec<_>>()),
        }
    }
}
