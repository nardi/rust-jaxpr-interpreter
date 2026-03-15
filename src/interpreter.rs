use std::{collections::HashMap, iter::zip};

use enum_dispatch::enum_dispatch;
use ndarray::{ArrayD, ArrayViewD};
use numpy::PyArrayMethods;
use pyo3::{PyErr, exceptions::PyKeyError};

use crate::jaxpr::{Atom, Jaxpr, UnknownEqn, Var};

#[derive(Debug)]
enum JaxprResult<'py> {
    External(ArrayViewD<'py, f64>),
    Local(ArrayD<f64>),
}

impl<'py> JaxprResult<'py> {
    pub fn view(&self) -> ArrayViewD<'_, f64> {
        match self {
            JaxprResult::External(view) => view.clone(),
            JaxprResult::Local(arr) => arr.view(),
        }
    }

    pub fn to_owned(self) -> ArrayD<f64> {
        match self {
            JaxprResult::External(view) => view.to_owned(),
            JaxprResult::Local(arr) => arr,
        }
    }
}

#[enum_dispatch(JaxprEqn)]
pub trait EvalJaxprEqn<'py> {
    fn eval_eqn(&'py self, interpreter: &mut Interpreter<'py>) -> Result<(), PyErr>;
}

#[derive(Debug)]
pub struct Interpreter<'py> {
    env: HashMap<Var, JaxprResult<'py>>,
}

impl<'py> Interpreter<'py> {
    pub fn new() -> Interpreter<'py> {
        Interpreter {
            env: HashMap::new(),
        }
    }

    fn read_one(&self, var: &Var) -> Result<&JaxprResult<'py>, PyErr> {
        self.env
            .get(var)
            .ok_or(PyKeyError::new_err("Tried to read unknown variable"))
    }

    fn read_or_resolve_one(&self, atom: &'py Atom<'py>) -> Result<ArrayViewD<'_, f64>, PyErr> {
        match atom {
            Atom::Var(var) => Ok(self.read_one(var)?.view()),
            Atom::Literal(lit) => Ok(lit.val.as_array()),
        }
    }

    fn read_or_resolve(
        &self,
        atoms: &'py [Atom<'py>],
    ) -> Result<Box<[ArrayViewD<'_, f64>]>, PyErr> {
        Ok(atoms
            .iter()
            .map(|atom| self.read_or_resolve_one(atom))
            .collect::<Result<Vec<_>, _>>()?
            .into_boxed_slice())
    }

    fn take_one(&mut self, var: &Var) -> Result<JaxprResult<'py>, PyErr> {
        self.env
            .remove(var)
            .ok_or(PyKeyError::new_err("Tried to take unknown variable"))
    }

    fn take_or_resolve_one(&mut self, atom: &'py Atom<'py>) -> Result<ArrayD<f64>, PyErr> {
        match atom {
            Atom::Var(var) => Ok(self.take_one(var)?.to_owned()),
            Atom::Literal(lit) => Ok(lit.val.to_owned_array()),
        }
    }

    fn take_or_resolve(&mut self, atoms: &'py [Atom<'py>]) -> Result<Box<[ArrayD<f64>]>, PyErr> {
        Ok(atoms
            .iter()
            .map(|atom| self.take_or_resolve_one(atom))
            .collect::<Result<Vec<_>, _>>()?
            .into_boxed_slice())
    }

    fn write_one(&mut self, var: &Var, val: JaxprResult<'py>) {
        self.env.insert(var.clone(), val);
    }

    fn write(&mut self, vars: &[Var], vals: impl Iterator<Item = JaxprResult<'py>>) {
        for (var, val) in zip(vars, vals) {
            self.write_one(var, val);
        }
    }

    pub fn eval_jaxpr(
        jaxpr: &'py Jaxpr<'py>,
        consts: impl Iterator<Item = ArrayViewD<'py, f64>>,
        args: impl Iterator<Item = ArrayViewD<'py, f64>>,
    ) -> Result<Box<[ArrayD<f64>]>, PyErr> {
        let mut interpreter = Interpreter::new();

        // Write args and consts into the env.
        interpreter.write(&jaxpr.constvars, consts.map(JaxprResult::External));
        interpreter.write(&jaxpr.invars, args.map(JaxprResult::External));

        // Loop over equations, executing each one.
        for eqn in jaxpr.eqns.iter() {
            eqn.eval_eqn(&mut interpreter)?;
        }

        // Take out the outputs (with ownership).
        interpreter.take_or_resolve(&jaxpr.outvars)
    }
}

impl<'py> EvalJaxprEqn<'py> for UnknownEqn<'py> {
    fn eval_eqn(&'py self, interpreter: &mut Interpreter<'py>) -> Result<(), PyErr> {
        let _ = interpreter;
        todo!("Primitive {} not yet implemented.", self.primitive.name)
    }
}

mod binary_primitives;
mod unary_primitives;
