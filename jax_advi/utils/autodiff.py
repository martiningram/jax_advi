from jax import jvp, grad, jit
from functools import partial


# forward-over-reverse
@partial(jit, static_argnums=0)
def hvp(f, primals, tangents):
    # Taken (and slightly modified) from:
    # https://jax.readthedocs.io/en/latest/notebooks/autodiff_cookbook.html
    return jvp(grad(f), (primals,), (tangents,))[1]
