import jax.extend.core as jex
import numpy as np

def parse_jaxpr(jaxpr: jex.Jaxpr) -> str: ...
def eval_jaxpr(
    jaxpr: jex.Jaxpr, consts: list[np.ndarray], *args: np.ndarray
) -> list[np.ndarray]: ...
