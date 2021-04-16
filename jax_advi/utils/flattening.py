import numpy as np
from functools import partial
from collections import namedtuple, OrderedDict


def extract_info(array):

    return array.shape


def flatten_and_summarise(**input_arrays):

    input_arrays = OrderedDict(input_arrays)

    summaries = OrderedDict({x: extract_info(y) for x, y in input_arrays.items()})

    flattened = np.concatenate([y.reshape(-1) for y in input_arrays.values()])

    return flattened, summaries


def reconstruct(flat_array, summaries, reshape_fun):

    # Base case
    if len(summaries) == 0:

        return {}

    cur_name, cur_summary = list(summaries.items())[0]

    # Cast to int is there to have this definitely work with TF
    cur_elements = int(np.prod(cur_summary))

    cur_result = {cur_name: reshape_fun(flat_array[:cur_elements], cur_summary)}

    remaining_summaries = OrderedDict(
        {x: y for x, y in summaries.items() if x != cur_name}
    )

    return {
        **cur_result,
        **reconstruct(
            flat_array[cur_elements:], remaining_summaries, reshape_fun=reshape_fun
        ),
    }
