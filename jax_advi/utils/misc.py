import numpy as np


def convert_decorator(fun, verbose=True):
    # A decorator that makes sure we return float64 dtypes, and optionally
    # prints the evaluation of the function.
    def result(x):

        value, grad = fun(x)

        if verbose:
            print(value, np.linalg.norm(grad))

        return (
            np.array(value).astype(np.float64),
            np.array(grad).astype(np.float64),
        )

    return result
