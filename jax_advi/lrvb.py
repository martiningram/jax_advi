from jax import jvp, grad, jit, vmap
from functools import partial
from jax.scipy.linalg import block_diag
from jax.scipy.sparse.linalg import cg
import jax.numpy as jnp
from tqdm import tqdm
from itertools import islice
from math import ceil
import numpy as np
from .utils.flattening import reconstruct
from .constraints import apply_constraints


# forward-over-reverse
@partial(jit, static_argnums=0)
def hvp(f, primals, tangents):
    # Taken (and slightly modified) from:
    # https://jax.readthedocs.io/en/latest/notebooks/autodiff_cookbook.html
    return jvp(grad(f), (primals,), (tangents,))[1]


def column_generator(n_params):

    for i in range(n_params):

        cur_column = (jnp.arange(2 * n_params) == i).astype(float)

        yield cur_column


def draw_from_mvn_cholesky(mean, cov, size=1):

    L = jnp.linalg.cholesky(cov)
    z = np.random.randn(size, mean.shape[0])

    draws = vmap(lambda cur_z: jnp.matmul(L, cur_z) + mean)(z)

    return draws


def split_every(n, iterable):
    # Credit to:
    # https://stackoverflow.com/questions/1915170/split-a-generator-iterable-every-n-items-in-python-splitevery
    i = iter(iterable)
    piece = list(islice(i, n))
    while piece:
        yield piece
        piece = list(islice(i, n))


def compute_lrvb_covariance(
    final_var_params_flat, objective_fun, shape_summary, batch_size=1
):

    n_params = final_var_params_flat.shape[0] // 2

    fun_to_use = lambda x: hvp(objective_fun, final_var_params_flat, x)

    @jit
    def preconditioner(x):

        # We know that the top left corner should be roughly given by the
        # diagonal variance
        # We don't know about the rest; use identity.
        var_ests = jnp.square(jnp.exp(final_var_params_flat[n_params:]))
        rest = jnp.ones(n_params)
        combined = jnp.concatenate([var_ests, rest])

        return x * combined

    @jit
    def solve_cg_single(vector):

        return cg(fun_to_use, vector, M=preconditioner)[0]

    vmapped = jit(vmap(solve_cg_single))

    result = list()

    # Split into sub-arrays
    split_relevant = split_every(batch_size, column_generator(n_params))

    # Use a for loop to conserve memory
    for cur_relevant in tqdm(split_relevant, total=ceil(n_params / batch_size)):

        cur_array = jnp.array(list(cur_relevant))

        result.append(vmapped(cur_array))

    result = jnp.concatenate(result)

    # Extract the posterior sds for convenience
    sds = jnp.sqrt(jnp.diag(result))
    sds = reconstruct(sds, shape_summary, jnp.reshape)

    return sds, result[:n_params, :n_params]


def get_posterior_draws_lrvb(
    free_means_flat,
    lrvb_cov_mat,
    shape_summary,
    constrain_fun_dict,
    n_draws=100,
    fun_to_apply=lambda x: x,
):

    # Make the draws
    draws = draw_from_mvn_cholesky(free_means_flat, lrvb_cov_mat, size=n_draws)

    def to_vmap(cur_draw):

        cur_draw = reconstruct(cur_draw, shape_summary, jnp.reshape)
        cur_constrained, _ = apply_constraints(cur_draw, constrain_fun_dict)
        function_result = fun_to_apply(cur_constrained)

        return function_result

    constrained_draws = vmap(to_vmap)(draws)

    return constrained_draws
