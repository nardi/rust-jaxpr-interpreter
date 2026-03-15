use pyo3::PyErr;

use crate::jaxpr::{AddEqn, MulEqn};

use super::{EvalJaxprEqn, Interpreter, JaxprResult};

impl<'py> EvalJaxprEqn<'py> for AddEqn<'py> {
    fn eval_eqn(&'py self, interpreter: &mut Interpreter<'py>) -> Result<(), PyErr> {
        interpreter.write_one(
            &self.outvars[0],
            JaxprResult::Local({
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
            JaxprResult::Local({
                let invals = interpreter.read_or_resolve(&self.invars)?;
                &invals[0] * &invals[1]
            }),
        );
        Ok(())
    }
}
