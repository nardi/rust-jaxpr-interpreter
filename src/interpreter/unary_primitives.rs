use pyo3::PyErr;

use crate::jaxpr::IntegerPowEqn;

use super::{EvalJaxprEqn, Interpreter, JaxprResult};

impl<'py> EvalJaxprEqn<'py> for IntegerPowEqn<'py> {
    fn eval_eqn(&'py self, interpreter: &mut Interpreter<'py>) -> Result<(), PyErr> {
        interpreter.write_one(
            &self.outvars[0],
            JaxprResult::Local(
                interpreter
                    .read_or_resolve_one(&self.invars[0])?
                    .powi(self.params.y),
            ),
        );
        Ok(())
    }
}
