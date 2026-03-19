use enum_dispatch::enum_dispatch;
use ndarray::{ArrayD, ArrayViewD};
use pyo3::PyErr;

use crate::jaxpr::unary::{IntegerPowPrimitive, SinPrimitive, UnaryJaxprEqn};

use super::{EvalJaxprEqn, Interpreter, JaxprValue};

#[enum_dispatch(UnaryJaxprPrimitive)]
pub trait EvalUnaryJaxprPrimitive {
    fn eval_primitive(&self, val: ArrayViewD<f64>) -> ArrayD<f64>;
}

impl<'py> EvalJaxprEqn<'py> for UnaryJaxprEqn {
    fn eval_eqn(&'py self, interpreter: &mut Interpreter<'py>) -> Result<(), PyErr> {
        interpreter.write_one(
            &self.outvars[0],
            JaxprValue::Local(
                self.primitive
                    .eval_primitive(interpreter.read_one(&self.invars[0])?.view()),
            ),
        );
        Ok(())
    }
}

impl EvalUnaryJaxprPrimitive for IntegerPowPrimitive {
    fn eval_primitive(&self, val: ArrayViewD<f64>) -> ArrayD<f64> {
        val.powi(self.y)
    }
}

impl EvalUnaryJaxprPrimitive for SinPrimitive {
    fn eval_primitive(&self, val: ArrayViewD<f64>) -> ArrayD<f64> {
        val.sin()
    }
}
