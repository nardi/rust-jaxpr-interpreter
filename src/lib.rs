use pyo3::prelude::*;

#[pymodule]
mod rust_jaxpr_interpreter {
    use numpy::{PyArrayDyn, PyArrayLikeDyn};
    use pyo3::prelude::*;
    use pyo3::types::PyList;

    #[pyfunction]
    #[pyo3(signature = (jaxpr, consts, *args))]
    fn eval_jaxpr<'py>(
        py: Python<'py>,
        jaxpr: &Bound<'py, PyAny>,
        consts: Vec<PyArrayLikeDyn<f64>>,
        args: Vec<PyArrayLikeDyn<f64>>,
    ) -> PyResult<Bound<'py, PyList>> {
        // TEMP: print the input arguments to check they are correct.
        println!("jaxpr: {}", jaxpr);
        println!("consts: {:?}", consts);
        println!("args: {:?}", args);

        // TEMP: calculate the output, knowing the function is `x -> x**2`.
        let out_arr = args[0].as_array().pow2();

        // Convert the output into a Numpy array, and wrap it in a list to match the `eval_jaxpr`
        // API.
        PyList::new(py, [PyArrayDyn::from_owned_array(py, out_arr)])
    }
}
