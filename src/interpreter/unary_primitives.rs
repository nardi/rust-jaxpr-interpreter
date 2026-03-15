use pyo3::PyErr;

use crate::jaxpr::unary_primitives::{IntegerPowEqn, SinEqn};

use super::{EvalJaxprEqn, Interpreter, JaxprValue};

impl<'py> EvalJaxprEqn<'py> for IntegerPowEqn {
    fn eval_eqn(&'py self, interpreter: &mut Interpreter<'py>) -> Result<(), PyErr> {
        interpreter.write_one(
            &self.outvars[0],
            JaxprValue::Local(
                interpreter
                    .read_one(&self.invars[0])?
                    .view()
                    .powi(self.params.y),
            ),
        );
        Ok(())
    }
}

impl<'py> EvalJaxprEqn<'py> for SinEqn {
    fn eval_eqn(&'py self, interpreter: &mut Interpreter<'py>) -> Result<(), PyErr> {
        interpreter.write_one(
            &self.outvars[0],
            JaxprValue::Local(interpreter.read_one(&self.invars[0])?.view().sin()),
        );
        Ok(())
    }
}
