import numpy as np
import jax.numpy as jnp
from ml_tools.flattening import flatten_and_summarise, reconstruct
from jax import vmap, jit, value_and_grad
from .constraints import apply_constraints
from ml_tools.jax import convert_decorator
from functools import partial
from scipy.optimize import minimize


@jit
def make_draws(z, mean, log_sd):

    draw = z * jnp.exp(log_sd) + mean

    return draw


@jit
def calculate_entropy(log_sds):

    return -jnp.sum(log_sds)


def optimize_advi_mean_field(
    theta_shape_dict,
    log_prior_fun,
    log_lik_fun,
    M=25,
    constrain_fun_dict={},
    verbose=False,
    seed=2,
    n_draws=None,
):

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
        "constrained_means": apply_constraints(means, constrain_fun_dict)[0],
    }

    if n_draws is not None:
        # Make draws from the parameters and constrain them
        draws = np.random.normal(
            loc=means_flat,
            scale=jnp.exp(log_sds_flat),
            size=(n_draws, means_flat.shape[0]),
        )

        def to_vmap(cur_draw):

            cur_unconstrained = reconstruct(cur_draw, summary, jnp.reshape)
            cur_constrained, _ = apply_constraints(
                cur_unconstrained, constrain_fun_dict
            )

            return cur_constrained

        constrained_draws = vmap(to_vmap)(draws)

        to_return["draws"] = constrained_draws

    return to_return
