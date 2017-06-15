import numpy as np

import exploit, explore

# TODO: Decay, how?
class EpsGreedy:

    def __init__(self, epsilon, greedy_action=exploit.sample, explore_action=explore.uniform):
        assert epsilon <= 1. and epsilon >= 0.
        self.epsilon = epsilon
        self.greedy_action = greedy_action
        self.explore_action = explore_action

    # TODO should we explore/exploit on the entire batch?
    def rank_docs(self, policy):
        if np.random.rand() < self.epsilon:
            return self.explore_action(policy)
        else:
            return self.greedy_action(policy)
