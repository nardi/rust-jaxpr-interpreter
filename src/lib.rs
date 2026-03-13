use pyo3::prelude::*;

mod boxed_slice;
mod interpreter;
mod jaxpr;

#[pymodule]
mod rust_jaxpr_interpreter {
    use super::interpreter::Interpreter;
    use super::jaxpr::Jaxpr;
    use numpy::{PyArrayDyn, PyArrayLikeDyn};
    use pyo3::prelude::*;
    use pyo3::types::PyList;

    #[pyfunction]
    #[pyo3(signature = (jaxpr, consts, *args))]
    fn eval_jaxpr<'py>(
        py: Python<'py>,
        jaxpr: Jaxpr<'py>,
        consts: Vec<PyArrayLikeDyn<f64>>,
        args: Vec<PyArrayLikeDyn<f64>>,
    ) -> PyResult<Bound<'py, PyList>> {
        // TEMP: print the input arguments to check they are correct.
        println!("jaxpr: {:?}", jaxpr);
        println!("consts: {:?}", consts);
        println!("args: {:?}", args);

        // Convert the input arguments to array views and evaluate the Jaxpr.
        let out_arrs = Interpreter::eval_jaxpr(
            &jaxpr,
            consts.iter().map(|arr| arr.as_array()),
            args.iter().map(|arr| arr.as_array()),
        )?;

        // Convert the outputs to a list of Numpy arrays to match the `eval_jaxpr` API.
        PyList::new(
            py,
            out_arrs
                .into_iter()
                .map(|arr| PyArrayDyn::from_owned_array(py, arr)),
        )
    }
}
