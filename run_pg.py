import numpy as np
import tensorflow as tf

import explorers

from tensorflow.contrib.learn.python.learn.datasets import mnist

from search.query import random_digit, random_from_docs
from search.reward import ndcg_full
from search import Environment
from models import PGRank
from utils import evaluate_model

# Hyperparameters.
k = 2
batch_size = 5012
num_epochs = 2
learning_rate = 1e-4
epsilon = 0.05
h_dim = 256

# Print hyperparameters.
print("Hyperparameters:")
print("k = %d" % k)
print("Batch size = %d" % batch_size)
print("Num epochs = %d" % num_epochs)
print("Learning rate = %f" % learning_rate)
print("Epsilon = %f" % epsilon)
print("Hidden layer dimension = %d" % h_dim)
print("")

# Load the dataset.
print("Loading dataset.")
dataset = mnist.load_mnist("data/")
data_dim = 784
num_queries = 10
print("")

# Create the environment and explorer.
env = Environment(dataset, k, batch_size, query_fn=random_from_docs, reward_fn=ndcg_full)
explorer = explorers.EpsGreedy(epsilon, greedy_action=explorers.exploit.greedy)

# Create model and optimizer.
model = PGRank(data_dim, num_queries, h_dim)
optimizer = tf.train.AdamOptimizer(learning_rate)
train_step = optimizer.minimize(model.loss)

# Train the model.
with tf.Session() as sess:
    sess.run(tf.initialize_all_variables())

    for epoch_id in range(num_epochs):
        print("epoch %d" % (epoch_id + 1))

        batch_id = 1
        for docs, queries in env.next_epoch():

            # Train on the batch.
            batch_reward, loss = model.train_on_batch(sess, train_step, docs, queries, \
                    env, explorer)
            print("batch %02d: loss = %.3f    \taverage batch reward = %.3f" % (batch_id, loss, batch_reward))

            # Administration.
            batch_id += 1

        # Print some evaluation metrics on the validation set.
        val_accuracy, val_ndcgs = evaluate_model(model, dataset.validation, sess, num_queries)
        avg_val_ndcg = np.average(val_ndcgs)
        print("validation accuracy = %.3f\taverage validation NDCG = %.3f" % (val_accuracy, avg_val_ndcg))

        print("")

    test_accuracy, test_ndcgs = evaluate_model(model, dataset.test, sess, num_queries)
    avg_test_ndcg = np.average(test_ndcgs)
    print("test accuracy = %.3f     \taverage test NDCG = %.3f" % (test_accuracy, avg_test_ndcg))

