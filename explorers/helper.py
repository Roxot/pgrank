import numpy as np

def sample_without_replacement(policy):
    rankings = np.zeros(policy.shape, dtype=np.int)
    for batch_id in range(policy.shape[0]):

        # Sample according to the policy for each batch, add some small value to prevent 0 entries
        # from crashing the sampler.
        rankings[batch_id][:] = np.random.choice(policy.shape[1], size=policy.shape[1], replace=False, \
            p=(policy[batch_id] + 1e-12))

    return rankings
