import numpy as np
import tensorflow as tf

import explorers

from tensorflow.contrib.learn.python.learn.datasets import mnist

from search.query import random_digit, random_from_docs
from search.reward import ndcg_full
from search import Environment
from models import PGRank

k = 2
batch_size = 1024
num_epochs = 5
learning_rate = 1e-4
epsilon = 0.05

# Print hyperparameters.
print("Hyperparameters:")
print("k = %d" % k)
print("Batch size = %d" % batch_size)
print("Num epochs = %d" % num_epochs)
print("Learning rate = %f" % learning_rate)
print("Epsilon = %f" % epsilon)
print("")

print("Loading dataset.")
dataset = mnist.load_mnist("data/")
print("")

env = Environment(dataset, k, batch_size, query_fn=random_from_docs, reward_fn=ndcg_full)
model = PGRank(784, 10, 200)
explorer = explorers.EpsGreedy(epsilon, greedy_action=explorers.exploit.greedy)
optimizer = tf.train.AdamOptimizer(learning_rate)
train_step = optimizer.minimize(model.loss)

with tf.Session() as sess:
    sess.run(tf.initialize_all_variables())

    for epoch_id in range(num_epochs):
        print("epoch %d" % (epoch_id + 1))

        batch_id = 1
        for docs, queries in env.next_epoch():

            feed_dict = {model.x: docs, model.q: queries}
            doc_scores, policy = sess.run([model.doc_scores, model.policy], feed_dict=feed_dict)

            ranking = explorer.rank_docs(policy)
            reward = env.reward(ranking)
            batch_reward = np.average(reward)

            feed_dict[model.reward] = reward
            feed_dict[model.deriv_weights] = model.derivative_weights(doc_scores, ranking)
            _, loss = sess.run([train_step, model.loss], feed_dict=feed_dict)

            print("batch %02d: loss = %.3f    \tavg_reward = %.3f" % (batch_id, loss, batch_reward))

            batch_id += 1

        print("")
