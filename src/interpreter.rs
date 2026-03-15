use std::{collections::HashMap, iter::zip};

use enum_dispatch::enum_dispatch;
use ndarray::{ArrayD, ArrayViewD};
use numpy::PyArrayMethods;
use pyo3::{PyErr, exceptions::PyKeyError};

use crate::jaxpr::{Atom, Jaxpr, UnknownEqn, Var};

/// Values used in Jaxprs come in two variants: external values living in the Python interpreter,
/// and local values we have calculated ourselves. These need to be handled differently, because
/// they are external values are unowned while local values are owned.
#[derive(Debug)]
enum JaxprValue<'py> {
    External(ArrayViewD<'py, f64>),
    Local(ArrayD<f64>),
}

impl<'py> JaxprValue<'py> {
    /// Create a view over the inner value.
    pub fn view(&self) -> ArrayViewD<'_, f64> {
        match self {
            JaxprValue::External(view) => view.clone(),
            JaxprValue::Local(arr) => arr.view(),
        }
    }

    /// Retrieve an owned version of the inner value, either by copying or consuming it. self is
    /// consumed in the process.
    pub fn to_owned(self) -> ArrayD<f64> {
        match self {
            JaxprValue::External(view) => view.to_owned(),
            JaxprValue::Local(arr) => arr,
        }
    }
}

#[enum_dispatch(JaxprEqn)]
pub trait EvalJaxprEqn<'py> {
    /// Evaluate a JaxprEqn, taking and storing values using the interpreter.
    fn eval_eqn(&'py self, interpreter: &mut Interpreter<'py>) -> Result<(), PyErr>;
}

/// A simple Jaxpr interpreter. Stores in- and output values in a HashMap.
#[derive(Debug)]
pub struct Interpreter<'py> {
    env: HashMap<Var, JaxprValue<'py>>,
}

impl<'py> Interpreter<'py> {
    /// Create a new interpreter with empty environment.
    pub fn new() -> Interpreter<'py> {
        Interpreter {
            env: HashMap::new(),
        }
    }

    /// Read a single value from the interpreter environment and return a read-only reference.
    fn read_one(&self, var: &Var) -> Result<&JaxprValue<'py>, PyErr> {
        self.env
            .get(var)
            .ok_or(PyKeyError::new_err("Tried to read unknown variable"))
    }

    /// Depending on the value of `atom`, either read the associated value from the interpreter
    /// environment or return a view on the literal value stored within.
    fn read_or_resolve_one(&self, atom: &'py Atom<'py>) -> Result<ArrayViewD<'_, f64>, PyErr> {
        match atom {
            Atom::Var(var) => Ok(self.read_one(var)?.view()),
            Atom::Literal(lit) => Ok(lit.val.as_array()),
        }
    }

    /// Read/resolve multiple atoms (see `read_or_resolve_one` for details).
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

    /// Take a single value from the interpreter environment (with ownership).
    fn take_one(&mut self, var: &Var) -> Result<JaxprValue<'py>, PyErr> {
        self.env
            .remove(var)
            .ok_or(PyKeyError::new_err("Tried to take unknown variable"))
    }

    /// Depending on the value of `atom`, either take the associated value from the interpreter
    /// environment or return an owned copy of the literal value stored within.
    fn take_or_resolve_one(&mut self, atom: &'py Atom<'py>) -> Result<ArrayD<f64>, PyErr> {
        match atom {
            Atom::Var(var) => Ok(self.take_one(var)?.to_owned()),
            Atom::Literal(lit) => Ok(lit.val.to_owned_array()),
        }
    }

    /// Take/resolve multiple atoms (see `take_or_resolve_one` for details).
    fn take_or_resolve(&mut self, atoms: &'py [Atom<'py>]) -> Result<Box<[ArrayD<f64>]>, PyErr> {
        Ok(atoms
            .iter()
            .map(|atom| self.take_or_resolve_one(atom))
            .collect::<Result<Vec<_>, _>>()?
            .into_boxed_slice())
    }

    /// Write a single value into the interpreter environment, giving up ownership.
    fn write_one(&mut self, var: &Var, val: JaxprValue<'py>) {
        self.env.insert(var.clone(), val);
    }

    /// Write multiple values into the interpreter environment, giving up ownership.
    fn write(&mut self, vars: &[Var], vals: impl Iterator<Item = JaxprValue<'py>>) {
        for (var, val) in zip(vars, vals) {
            self.write_one(var, val);
        }
    }

    /// Evaluate a Jaxpr using a new interpreter, and return the output values.
    pub fn eval_jaxpr(
        jaxpr: &'py Jaxpr<'py>,
        consts: impl Iterator<Item = ArrayViewD<'py, f64>>,
        args: impl Iterator<Item = ArrayViewD<'py, f64>>,
    ) -> Result<Box<[ArrayD<f64>]>, PyErr> {
        let mut interpreter = Interpreter::new();

        // Write args and consts into the env.
        interpreter.write(&jaxpr.constvars, consts.map(JaxprValue::External));
        interpreter.write(&jaxpr.invars, args.map(JaxprValue::External));

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

pub mod binary_primitives;
pub mod unary_primitives;
