import numpy as np

def ndcg_full(serp, rel_labels):
    rel_serp = np.array([rel_labels[i, serp[i]] for i in range(serp.shape[0])])
    ideal_serp = np.fliplr(np.sort(rel_serp, axis=1))
    return _dcg(rel_serp) / (_dcg(ideal_serp) + 1e-12)

def _dcg(rel_list):
    cg = np.power(2, rel_list) - 1
    discount = np.tile(np.log2(np.arange(rel_list.shape[1]) + 2), (rel_list.shape[0], 1))
    return np.sum(np.divide(cg, discount), axis=1, keepdims=True)
