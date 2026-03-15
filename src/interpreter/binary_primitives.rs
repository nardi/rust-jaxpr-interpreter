use enum_dispatch::enum_dispatch;
use ndarray::{ArrayD, ArrayViewD};
use pyo3::PyErr;

use crate::jaxpr::binary_primitives::{AddPrimitive, BinaryJaxprEqn, MulPrimitive};

use super::{EvalJaxprEqn, Interpreter, JaxprValue};

#[enum_dispatch(BinaryJaxprPrimitive)]
pub trait EvalBinaryJaxprPrimitive {
    fn eval_primitive(&self, lhs: &ArrayViewD<f64>, rhs: &ArrayViewD<f64>) -> ArrayD<f64>;
}

impl<'py> EvalJaxprEqn<'py> for BinaryJaxprEqn<'py> {
    fn eval_eqn(&'py self, interpreter: &mut Interpreter<'py>) -> Result<(), PyErr> {
        interpreter.write_one(
            &self.outvars[0],
            JaxprValue::Local({
                let invals = interpreter.read_or_resolve(&self.invars)?;
                self.primitive.eval_primitive(&invals[0], &invals[1])
            }),
        );
        Ok(())
    }
}

impl EvalBinaryJaxprPrimitive for AddPrimitive {
    fn eval_primitive(&self, lhs: &ArrayViewD<f64>, rhs: &ArrayViewD<f64>) -> ArrayD<f64> {
        lhs + rhs
    }
}

impl EvalBinaryJaxprPrimitive for MulPrimitive {
    fn eval_primitive(&self, lhs: &ArrayViewD<f64>, rhs: &ArrayViewD<f64>) -> ArrayD<f64> {
        lhs * rhs
    }
}
