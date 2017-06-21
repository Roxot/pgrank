import numpy as np

import exploit, explore

# TODO: Decay, how?
class EpsGreedy:

    def __init__(self, epsilon, greedy_action=exploit.sample, explore_action=explore.uniform):
        assert epsilon <= 1. and epsilon >= 0.
        self.epsilon = epsilon
        self.greedy_action = greedy_action
        self.explore_action = explore_action

    def rank_docs(self, policy):

        # Decide which batch elements we explore and exploit.
        rand_unif = np.random.rand(policy.shape[0], 1)
        should_explore = np.tile((rand_unif < self.epsilon).astype(int), policy.shape[1])
        should_exploit = np.tile((rand_unif >= self.epsilon).astype(int), policy.shape[1])

        # Perform both the greedy as wel as the exploratory actions.
        exploratory_batch = self.explore_action(policy)
        greedy_batch = self.greedy_action(policy)

        # Return the batch balanced between exploration and exploitation.
        return exploratory_batch * should_explore + greedy_batch * should_exploit
