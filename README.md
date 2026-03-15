# `rust-jaxpr-interpreter`

`rust-jaxpr-interpreter` is a package that provides an alternative implementation of JAX's [`jax.core.eval_jaxpr`](https://github.com/jax-ml/jax/blob/8304fec218864ee24e5beae80cc5d095b5efad0a/jax/_src/core.py#L721), which evaluates an arbitrary [Jaxpr](https://docs.jax.dev/en/latest/jaxpr.html) by translating to operations from the Rust [`ndarray`](https://docs.rs/ndarray/latest/ndarray/) crate. The goal is to be able to evaluate any arbitrary Jaxpr without custom primitives produced by `jax.make_jaxpr`.

I have mostly been developing this as a learning exercise on Rust and Rust/Python interfacing. Right now, it is not usable as only a few primitives have been implemented. When developed further, it could be useful to have a straightforward way to execute a JAX function on CPU (without the limitations and intransparency of XLA compilation) that is more performant than just executing op-by-op from Python. If you want to mostly make use of the tracing transformations provided by JAX, such as autodiff, but the overhead of compilation is not worth it, this mode of execution could be useful.

In the future, it would also be interesting to explore execution of [shape-polymorphic traces](https://docs.jax.dev/en/latest/export/shape_poly.html) (which JAX cannot do without recompilation for each concrete shape), or execution based on sparse arrays (without having to modify the traced code).

## Currently implemented

- The basic Jaxpr data model and parsing from Python data structures via PyO3.
- A basic interpreter in the vein of `jax.core.eval_jaxpr`.
- A few test primitives: `integer_pow`, `sin`, `add`, `mul`.

Not much, but having the basic structure already gives a good starting point for further extension :)

Current limitations are:
- All input and literal arrays are upcast to `f64`.

## To-do list (or roadmap if you want to be fancy)

1. Collect a few interesting test cases (maybe simple examples from higher level JAX-based libraries like [Flax](https://flax.readthedocs.io/en/v0.8.1/examples/index.html)).
2. Implement all unary and binary primitives that have straightforward equivalents in `ndarray`.
3. Implement control logic primitives like `cond` and `select` (with downcasts from `f64` to `bool`).
4. Implement indexing primitives like `gather` (with downcasts from `f64` to `isize`).
5. Implement type-generic arrays and operations to natively support integer and boolean arrays (and possibly `f32`).
6. Keep going until all the test cases run!
