import numpy as np

import exploit, explore

# TODO: Decay, how: linearly from 1.0 to e.g. 0.1 over x (e.g. 1 million) steps, then fix at 0.1.
class EpsGreedy:

    def __init__(self, epsilon, greedy_action=exploit.sample, explore_action=explore.Uniform()):
        assert epsilon <= 1. and epsilon >= 0.
        self.epsilon = epsilon
        self.greedy_action = greedy_action
        self.explore_action = explore_action

    def rank_docs(self, policy, labels):

        # Decide which batch elements we explore and exploit.
        rand_unif = np.random.rand(policy.shape[0], 1)
        should_explore = np.tile((rand_unif < self.epsilon).astype(int), policy.shape[1])
        should_exploit = np.tile((rand_unif >= self.epsilon).astype(int), policy.shape[1])

        # Perform both the greedy as wel as the exploratory actions.
        exploratory_batch = self.explore_action.rank(policy, labels)
        greedy_batch = self.greedy_action(policy)

        # Return the batch balanced between exploration and exploitation.
        return exploratory_batch * should_explore + greedy_batch * should_exploit

    # Calculates the weights that we need to give to samples in order to correct for exploration.
    def sample_weight(self, action_probs, ranking, labels):
        batch_size = action_probs.shape[0]
        k = action_probs.shape[1]
        corr_numerator = np.zeros(batch_size)
        corr_denominator = np.zeros(batch_size)

        # Get the probabilities of doing exploration for each document sampled.
        exploration_probs = self.explore_action.explore_dist(ranking, labels)

        # The actual probabilities we are using in sampling.
        actual_probs = (1.0 - self.epsilon) * action_probs + self.epsilon * exploration_probs

        # Calculate the weights for the samples as p_true / p_actual.
        correction = np.log(action_probs + 1e-12)
        correction = np.sum(correction, axis=1, keepdims=True)
        correction -= np.sum(np.log(actual_probs + 1e-12), axis=1, keepdims=True)
        correction = np.exp(correction)

        return correction
