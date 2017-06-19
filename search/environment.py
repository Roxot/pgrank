import numpy as np

from query import random_from_docs
from reward import ndcg_full

class Environment:

    def __init__(self, dataset, k, batch_size, query_fn=random_from_docs, reward_fn=ndcg_full):
        self.dataset = dataset
        self.epoch = self.dataset.train.epochs_completed
        self.k = k
        self.batch_size = batch_size
        self.reward_fn = reward_fn
        self.query_fn = query_fn

    def reward(self, serp):
        return self.reward_fn(serp, self.rel_labels)

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
