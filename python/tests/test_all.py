import numpy as np
import jax
import jax.core
import jax.extend.core
import pytest

import rust_jaxpr_interpreter

JaxprAndArgs = tuple[jax.extend.core.ClosedJaxpr, tuple[np.ndarray, ...]]


@pytest.fixture
def closed_jaxpr_and_args() -> JaxprAndArgs:
    def f(x):
        return (np.array([2.0, 4, 6]) * x) ** 2 + 5

    x = np.array([1.0, 2, 3])

    f_closed_jaxpr = jax.make_jaxpr(f)(x)

    return f_closed_jaxpr, (x,)


def test_eval_jaxpr(closed_jaxpr_and_args: JaxprAndArgs):
    closed_jaxpr, args = closed_jaxpr_and_args

    y_ref = jax.core.eval_jaxpr(closed_jaxpr.jaxpr, closed_jaxpr.consts, *args)
    y = rust_jaxpr_interpreter.eval_jaxpr(
        closed_jaxpr.jaxpr, closed_jaxpr.consts, *args
    )

    assert all(np.allclose(a, b) for a, b in zip(y_ref, y, strict=True))


def test_execute():
    def f(d):
        return {"y": d["x"] ** 2}

    x = np.array([1.0, 2, 3])

    y = rust_jaxpr_interpreter.execute(f, {"x": x})["y"]

    assert np.allclose(y, x**2)
