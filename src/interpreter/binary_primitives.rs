use pyo3::PyErr;

use crate::jaxpr::binary_primitives::{AddEqn, MulEqn};

use super::{EvalJaxprEqn, Interpreter, JaxprValue};

impl<'py> EvalJaxprEqn<'py> for AddEqn<'py> {
    fn eval_eqn(&'py self, interpreter: &mut Interpreter<'py>) -> Result<(), PyErr> {
        interpreter.write_one(
            &self.outvars[0],
            JaxprValue::Local({
                let invals = interpreter.read_or_resolve(&self.invars)?;
                &invals[0] + &invals[1]
            }),
        );
        Ok(())
    }
}

impl<'py> EvalJaxprEqn<'py> for MulEqn<'py> {
    fn eval_eqn(&'py self, interpreter: &mut Interpreter<'py>) -> Result<(), PyErr> {
        interpreter.write_one(
            &self.outvars[0],
            JaxprValue::Local({
                let invals = interpreter.read_or_resolve(&self.invars)?;
                &invals[0] * &invals[1]
            }),
        );
        Ok(())
    }
}
