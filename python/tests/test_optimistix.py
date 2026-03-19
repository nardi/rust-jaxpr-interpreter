"""
This is an example for the Optimistix library.
Source: https://github.com/patrick-kidger/optimistix/blob/0fbf383cf14d246661ccd25df3c6b3b4aa005030/docs/examples/root_find.ipynb
"""


def test_optimistix_root_find():
    import jax
    import jax.numpy as jnp
    import optimistix as optx

    import rust_jaxpr_interpreter as rji

    # Often import when doing scientific work
    jax.config.update("jax_enable_x64", True)

    def fn(y, args):
        a, b = y
        c = jnp.tanh(jnp.sum(b)) - a
        d = a**2 - jnp.sinh(b + 1)
        return c, d

    solver = optx.Newton(rtol=1e-8, atol=1e-8)
    y0 = (jnp.array(0.0), jnp.zeros((2, 2)))
    # sol = jax.jit(lambda y0: optx.root_find(fn, solver, y0))(y0)
    sol = rji.jit(lambda y0: optx.root_find(fn, solver, y0))(y0)

    print(sol)
