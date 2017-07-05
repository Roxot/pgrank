import numpy as np

from helper import sample_without_replacement

# Explores independent of the policy, queries or labels using a uniform distribution over actions.
class Uniform:

    def rank(self, policy, labels):
        uniform_policy = np.full(policy.shape, (1. / policy.shape[1]))
        return sample_without_replacement(uniform_policy)

    def explore_dist(self, ranking, labels):
        k = ranking.shape[1]
        exploration_probs = np.arange(k)
        exploration_probs = k - exploration_probs
        exploration_probs = 1.0 / exploration_probs
        return exploration_probs

# Explores using the best possible list according to the true labels of the documents and the queries.
class Oracle:

    # TODO deal with permutations, important.
    def rank(self, policy, labels):
        return np.argsort(labels, axis=1)[:, ::-1]

    def explore_dist(self, ranking, labels):
        oracle = self.rank(None, labels)
        explore_probs = (np.sum(np.abs(oracle - ranking), axis=1, keepdims=True) == 0).astype(float)
        return np.tile(explore_probs, (1, ranking.shape[1]))
