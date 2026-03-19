use pyo3::PyErr;

use crate::jaxpr::higher_order::HigherOrderJaxprEqn;

use super::{EvalJaxprEqn, Interpreter, JaxprValue};

impl<'py> EvalJaxprEqn<'py> for HigherOrderJaxprEqn<'py> {
    fn eval_eqn(&'py self, interpreter: &mut Interpreter<'py>) -> Result<(), PyErr> {
        let args = interpreter
            .read(&self.invars)?
            .into_iter()
            .map(|arg| arg.view());
        let consts = self
            .params
            .consts
            .iter()
            .map(|arr_like| arr_like.as_array());
        let outvals = Interpreter::eval_jaxpr(&self.params.jaxpr, consts, args)?;
        interpreter.write(
            &self.outvars,
            outvals.into_iter().map(|arr| JaxprValue::Local(arr)),
        );
        Ok(())
    }
}
