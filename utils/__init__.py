import numpy as np

from scipy.misc import logsumexp

def softmax(a):
    log_Z = logsumexp(a, axis=1, keepdims=True)
    log_softmax = a - log_Z
    return np.exp(log_softmax)
