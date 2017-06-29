import numpy as np

from helper import sample_without_replacement

class Uniform:

    def rank(self, policy):
        uniform_policy = np.full(policy.shape, (1. / policy.shape[1]))
        return sample_without_replacement(uniform_policy)

    def explore_dist(self, k):
        exploration_probs = np.arange(k)
        exploration_probs = k - exploration_probs
        exploration_probs = 1.0 / exploration_probs
        return exploration_probs

