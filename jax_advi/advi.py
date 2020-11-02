import numpy as np
import jax.numpy as jnp
from .utils.flattening import flatten_and_summarise, reconstruct
from jax import vmap, jit, value_and_grad
from .constraints import apply_constraints
from .utils.misc import convert_decorator
from functools import partial
from scipy.optimize import minimize
from typing import Tuple, Dict, Callable, Any


@jit
def make_draws(z, mean, log_sd):

    draw = z * jnp.exp(log_sd) + mean

    return draw


@jit
def calculate_entropy(log_sds):

    return -jnp.sum(log_sds)


def optimize_advi_mean_field(
    theta_shape_dict: Dict[str, Tuple],
    log_prior_fun: Callable[[Dict[str, jnp.ndarray]], float],
    log_lik_fun: Callable[[Dict[str, jnp.ndarray]], float],
    M: int = 25,
    constrain_fun_dict: Dict[
        str, Callable[[jnp.ndarray], Tuple[jnp.ndarray, float]]
    ] = {},
    verbose: bool = False,
    seed: int = 2,
    n_draws: int = 1000,
) -> Dict[str, Any]:
    """Minimizes the KL divergence between the posterior defined by the
    `log_prior_fun` and `log_lik_fun` and a mean-field variational Bayes
    approximation using a factorised normal distribution. Based on the variant
    of ADVI detailed in "Covariances, Robustness, and Variational Bayes" by
    Giordano, Broderick and Jordan
    [https://www.jmlr.org/papers/v19/17-670.html].

    Args:
        theta_shape_dict: This is a dictionary containing the shapes of the
            parameters to perform inference on.
        log_prior_fun: A function taking in the dictionary of parameter settings and
            returning their log prior density.
        log_lik_fun: A function taking in the dictionary of parameter settings and
            returning their log likelihood.
        M: The number of Monte Carlo samples to use to optimize ADVI. Defaults to
            25. More is better, but yields diminishing returns. Giordano et
            al. suggests 10 may be enough, so 25 should be fairly conservative.
        constrain_fun_dict: A dictionary of the constraints the parameters have to
            satisfy. For example, variance parameters are constrained to be positive
            and can be constrained with the `constrain_positive` function. The
            constraint function must return the constrained parameter as well as the
            log determinant of the Jacobian of the transformation.
        verbose: If True, prints the value of the objective and the norm of its
            gradient at each iteration of the optimisation.
        seed: A random seed for reproducibility.
        n_draws: The number of draws to return. Can be set to None if no draws
            are desired.

    Returns:
    A dictionary of results. It contains the fields "free_means" and "free_sds"
    listing the unconstrained variational parameters, "opt_result" containing
    the results of the optimisation, and "draws" if n_draws is not None, which
    is a dictionary of draws from the constrained parameters. In general, these
    should be used rather than the free means and sds since they have been
    appropriately transformed.
    """

    # First, create placeholders for the theta dict so we know how to
    # reconstruct it [TODO: could avoid this step]
    theta = {x: jnp.empty(y) for x, y in theta_shape_dict.items()}

    flat_theta, summary = flatten_and_summarise(**theta)

    # Initialise variational parameters
    # First row is means, second is log_sd
    var_params = np.zeros((2, flat_theta.shape[0]))

    np.random.seed(seed)

    # Fixed draws from standard normal a la Giordano et al:
    # https://www.jmlr.org/papers/v19/17-670.html
    # TODO: Could use JAX here to draw things (but probably not essential)
    zs = np.random.randn(M, flat_theta.shape[0])

    def to_minimize(var_params_flat):

        var_params = var_params_flat.reshape(2, -1)

        cur_entropy = calculate_entropy(var_params[1])

        def calculate_log_posterior(cur_z):

            cur_flat_theta = make_draws(cur_z, var_params[0], var_params[1])
            cur_theta = reconstruct(cur_flat_theta, summary, jnp.reshape)

            # Compute the log determinant of the constraints
            cur_theta, cur_log_det = apply_constraints(cur_theta, constrain_fun_dict)

            cur_likelihood = log_lik_fun(cur_theta)
            cur_prior = log_prior_fun(cur_theta)

            return cur_likelihood + cur_prior + cur_log_det

        individual_log_posteriors = vmap(calculate_log_posterior)(zs)

        return -jnp.mean(individual_log_posteriors) + cur_entropy

    with_grad = partial(convert_decorator, verbose=verbose)(
        jit(value_and_grad(to_minimize))
    )

    result = minimize(with_grad, var_params.reshape(-1), method="L-BFGS-B", jac=True)

    means_flat, log_sds_flat = result.x.reshape(2, -1)

    means = reconstruct(means_flat, summary, jnp.reshape)
    sds = reconstruct(jnp.exp(log_sds_flat), summary, jnp.reshape)

    to_return = {
        "free_means": means,
        "free_sds": sds,
        "opt_result": result,
    }

    if n_draws is not None:
        # Make draws from the parameters and constrain them

        to_return["draws"] = get_posterior_draws(
            means, sds, constrain_fun_dict, n_draws
        )

    return to_return


def get_posterior_draws(
    free_means, free_sds, constrain_fun_dict, n_draws=1000, fun_to_apply=lambda x: x
):
    """Makes draws from the variational posterior.

    Args:
    free_means: The dictionary of free means returned by
        optimize_advi_mean_field.
    free_sds: The dictionary of free sds returned by optimize_advi_mean_field.
        constrain_fun_dict: The constraint functions to apply to the parameters
        [see optimize_advi_mean_field for information].
    n_draws: The number of draws to make.
    fun_to_apply: An optional function to make draws of instead. By default,
        this function is the identity, but it could be used to e.g. draw the value
        of linear predictors in a regression.
    """

    # Make the draws
    draws = {
        x: np.random.normal(
            loc=free_means[x], scale=free_sds[x], size=(n_draws, *free_means[x].shape)
        )
        for x in free_means
    }

    def to_vmap(cur_draw):

        cur_constrained, _ = apply_constraints(cur_draw, constrain_fun_dict)
        function_result = fun_to_apply(cur_constrained)

        return function_result

    constrained_draws = vmap(to_vmap)(draws)

    return constrained_draws
