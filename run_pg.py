import numpy as np

from tensorflow.contrib.learn.python.learn.datasets import mnist

from search.query import random_digit, random_from_docs
from search.reward import ndcg_full
from search import Environment

k = 3
batch_size = 2

# Print hyperparameters.
print("k = %d" % k)
print("Batch size = %d" % batch_size)

print("Loading dataset.")
dataset = mnist.load_mnist("data/")
env = Environment(dataset, k, batch_size, query_fn=random_digit, reward_fn=ndcg_full)
for docs, queries in env.next_epoch():
    print(docs)
    print(queries)
    break
