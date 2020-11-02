import os
import pystan
import pickle
from datetime import datetime

def load_pickle_safely(path_to_pickle):

    assert os.path.isfile(path_to_pickle)

    with open(path_to_pickle, 'rb') as f:
        return pickle.load(f)


def save_pickle_safely(data_to_pickle, path_to_save_to):

    with open(path_to_save_to, 'wb') as f:
        pickle.dump(data_to_pickle, f)

        
def load_stan_model_cached(model_path):

    model_last_modified = os.path.getmtime(model_path)

    assert(os.path.isfile(model_path))
    cache_path = os.path.splitext(model_path)[0] + '.pkl'

    if (os.path.isfile(cache_path) and os.path.getmtime(cache_path) >
            model_last_modified):

        return load_pickle_safely(cache_path)

    else:

        model = pystan.StanModel(model_path)
        save_pickle_safely(model, cache_path)
        return model