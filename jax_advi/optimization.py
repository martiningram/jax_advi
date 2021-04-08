from .utils.autodiff import hvp
from scipy.optimize import minimize
from jax import jit, value_and_grad
from .utils.misc import convert_decorator, print_decorator, count_decorator
from functools import partial


def optimize_with_jac(to_minimize, start_params, method_name="L-BFGS-B", verbose=False):

    with_grad = partial(convert_decorator, verbose=verbose)(
        jit(value_and_grad(to_minimize))
    )

    result = minimize(with_grad, start_params, method=method_name, jac=True)

    return result


def optimize_with_hvp(
    to_minimize,
    start_params,
    method_name="trust-ncg",
    verbose=False,
    minimize_kwargs={},
):

    val_grad_fun = jit(value_and_grad(to_minimize))

    decorated = count_decorator(partial(print_decorator, verbose=verbose)(val_grad_fun))

    hvp_fun = lambda x, p: hvp(to_minimize, x, p)
    hvp_fun = count_decorator(jit(hvp_fun))

    result = minimize(
        decorated,
        start_params,
        method=method_name,
        hessp=hvp_fun,
        jac=True,
        **minimize_kwargs
    )

    n_hvp_calls = hvp_fun.calls
    n_val_and_grad_calls = decorated.calls

    return (
        result,
        hvp_fun,
        val_grad_fun,
        {"n_hvp_calls": n_hvp_calls, "n_val_and_grad_calls": n_val_and_grad_calls},
    )
