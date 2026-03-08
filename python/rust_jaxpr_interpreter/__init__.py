from collections.abc import Callable
from typing import Any

import jax
import jax.core

from .rust_jaxpr_interpreter import eval_jaxpr

__all__ = [
    "eval_jaxpr",
    "execute",
]

USE_JAX = False
"""Constant, can be set to true to fall back to JAX eval_jaxpr for testing."""


def execute(f: Callable, *args, **kwargs) -> Any:
    # Trace f and create a Jaxpr.
    f_closed_jaxpr, out_info = jax.make_jaxpr(f, return_shape=True)(*args, **kwargs)

    # Determine the structure of the function output.
    out_tree = jax.tree.structure(
        out_info, is_leaf=lambda _: isinstance(_, jax.ShapeDtypeStruct)
    )

    # Convert the arguments to a flat list.
    flat_args = jax.tree.leaves((args, kwargs))

    # Optionally, choose the standard JAX implementation as reference.
    eval_jaxpr_fun = eval_jaxpr
    if USE_JAX:
        eval_jaxpr_fun = jax.core.eval_jaxpr

    # Evaluate the Jaxpr.
    flat_result = eval_jaxpr_fun(
        f_closed_jaxpr.jaxpr, f_closed_jaxpr.consts, *flat_args
    )

    # Convert the results to the proper structure and return.
    return jax.tree.unflatten(out_tree, flat_result)
