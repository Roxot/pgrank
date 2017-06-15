import numpy as np
import tensorflow as tf

import explorers

from tensorflow.contrib.learn.python.learn.datasets import mnist

from search.query import random_digit, random_from_docs
from search.reward import ndcg_full
from search import Environment
from models import PGRank

k = 3
batch_size = 10

# Print hyperparameters.
print("k = %d" % k)
print("Batch size = %d" % batch_size)

print("Loading dataset.")
dataset = mnist.load_mnist("data/")
env = Environment(dataset, k, batch_size, query_fn=random_digit, reward_fn=ndcg_full)
model = PGRank(784, 10, 200)
explorer = explorers.EpsGreedy(0.5, greedy_action=explorers.exploit.greedy)

with tf.Session() as sess:
    sess.run(tf.initialize_all_variables())
    for docs, queries in env.next_epoch():
        policy = sess.run(model.policy, feed_dict={model.x: docs, model.q: queries})
        print(policy)
        print explorer.rank_docs(policy)
        break
