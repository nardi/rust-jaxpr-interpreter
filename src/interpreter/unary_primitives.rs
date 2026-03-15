use pyo3::PyErr;

use crate::jaxpr::unary_primitives::IntegerPowEqn;

use super::{EvalJaxprEqn, Interpreter, JaxprValue};

impl<'py> EvalJaxprEqn<'py> for IntegerPowEqn<'py> {
    fn eval_eqn(&'py self, interpreter: &mut Interpreter<'py>) -> Result<(), PyErr> {
        interpreter.write_one(
            &self.outvars[0],
            JaxprValue::Local(
                interpreter
                    .read_or_resolve_one(&self.invars[0])?
                    .powi(self.params.y),
            ),
        );
        Ok(())
    }
}
