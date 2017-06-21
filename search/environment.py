import numpy as np

from query import random_from_docs
from reward import ndcg_full

class Environment:

    def __init__(self, dataset, k, batch_size, query_fn=random_from_docs, reward_fn=ndcg_full, use_baseline=True):
        self.dataset = dataset
        self.epoch = self.dataset.train.epochs_completed
        self.k = k
        self.batch_size = batch_size
        self.reward_fn = reward_fn
        self.query_fn = query_fn
        self.use_baseline = use_baseline
        self.episodes = 0
        self.total_reward = 0
        self.baseline = 0.

    def reward(self, serp):
        reward = self.reward_fn(serp, self.rel_labels)
        if self.use_baseline:
            self._update_baseline(reward)
        return reward, self.baseline

    def next_epoch(self):
        self.dataset.train._index_in_epoch = 0
        while self.dataset.train.epochs_completed < (self.epoch + self.k):
            docs, doc_labels = self.dataset.train.next_batch(self.k * self.batch_size)
            docs = np.reshape(docs, (self.batch_size, self.k, docs.shape[1]))
            doc_labels = np.reshape(doc_labels, (self.batch_size, self.k))
            queries = self.query_fn(doc_labels)
            self.rel_labels = np.zeros(doc_labels.shape)
            self.rel_labels[np.where(doc_labels == queries)] = 1.
            yield (docs, queries)
        self.epoch = self.dataset.train.epochs_completed

    # As a baseline use the average reward.
    def _update_baseline(self, reward):
        self.total_reward += reward
        self.episodes += 1
        self.baseline = self.total_reward / self.episodes
