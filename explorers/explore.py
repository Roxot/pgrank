import numpy as np

from helper import sample_without_replacement

def uniform(policy):
    uniform_policy = np.full(policy.shape, (1. / policy.shape[1]))
    return sample_without_replacement(uniform_policy)
