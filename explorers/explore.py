import numpy as np
import itertools

from scipy.special import factorial

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

    def rank(self, policy, labels):
        ranking = np.zeros(labels.shape, dtype=int)

        for i, row in enumerate(labels):
            ones = np.where(row == 1)[0]
            zeros = np.where(row == 0)[0]
            ranking[i] = np.concatenate([np.random.permutation(ones), np.random.permutation(zeros)])

        return ranking

    def explore_dist(self, ranking, labels):
        explore_probs = np.zeros(labels.shape, dtype=float)

        for i in range(labels.shape[0]):
            ranked_labels = labels[i][ranking[i]]
            if self._is_oracle(ranked_labels):
                num_ones = np.sum(labels[i])
                explore_probs[i] = 1.0 / (np.concatenate([np.arange(num_ones)[::-1] + 1, \
                        np.arange(labels.shape[1] - num_ones)[::-1] + 1]))

        return explore_probs

    def _is_oracle(self, ranking):
        return all(a >= b for a, b in zip(ranking[:-1], ranking[1:]))
