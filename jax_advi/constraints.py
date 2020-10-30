import jax.numpy as jnp


def constrain_exp(theta):
    # Computes the value and log determinant of the transformation y =
    # exp(theta). theta can be any shape.
    # TODO: Make docstring better.

    value = jnp.exp(theta)
    log_det = jnp.sum(theta)

    return value, log_det


def apply_constraints(theta_dict, constraint_dict):
    # theta_dict is a dictionary of "var_name -> unconstrained_value"
    # constraint_dict is a dictionary of "var_name -> constrain_fun", where the
    # constrain_fun has to take in the unconstrained value and return a tuple of
    # the constrained value and the log determinant of the Jacobian of the
    # transformation (see constrain_exp for an example).

    new_dict = {x: y for x, y in theta_dict.items()}

    log_det = 0.0

    for cur_var_name, cur_constrain_fun in constraint_dict.items():

        new_dict[cur_var_name], cur_log_det = cur_constrain_fun(
            theta_dict[cur_var_name]
        )

        log_det = log_det + cur_log_det

    return new_dict, log_det


constrain_positive = constrain_exp
