import numpy as np
import tensorflow as tf
import argparse
import itertools

import explorers

from tensorflow.contrib.learn.python.learn.datasets import mnist

from search.query import random_digit, random_from_docs
from search.reward import ndcg_full
from search import Environment
from models import PGRank
from utils import evaluate_model, mnist_image_summary, write_train_summaries, write_evaluation_summaries

def run_model(k, learning_rate, batch_size):
    query_fn = random_from_docs
    reward_fn = ndcg_full
    greedy_action = explorers.exploit.sample
    h_dim = 256
    num_epochs = 5 # TODO
    epsilon = 0.

    # Print hyperparameters.
    print("Hyperparameters:")
    print("k = %d" % k)
    print("Batch size = %d" % batch_size)
    print("Num epochs = %d" % num_epochs)
    print("Learning rate = %f" % learning_rate)
    print("Epsilon = %f" % epsilon)
    print("Hidden layer dimension = %d" % h_dim)
    print("Query function = %s" % query_fn)
    print("Reward function = %s" % reward_fn)
    print("Greedy action = %s" % greedy_action)
    print("")

    # Load the dataset.
    print("Loading dataset.")
    dataset = mnist.load_mnist("data/")
    data_dim = 784
    num_queries = 10
    print("")

    # Create the environment and explorer.
    env = Environment(dataset, k, batch_size, query_fn=query_fn, reward_fn=reward_fn)
    explorer = explorers.EpsGreedy(epsilon, greedy_action=greedy_action)

    # Create model and optimizer.
    model = PGRank(data_dim, num_queries, h_dim)
    optimizer = tf.train.AdamOptimizer(learning_rate)
    train_step = optimizer.minimize(model.loss)
    val_ndcgs = np.zeros(num_epochs)

    # Train the model.
    with tf.Session() as sess:
        sess.run(tf.initialize_all_variables())

        for epoch_id in range(num_epochs):
            print("epoch %d" % (epoch_id + 1))

            train_losses = []
            batch_rewards = []
            for docs, queries in env.next_epoch():

                # Train on the batch.
                batch_reward, loss = model.train_on_batch(sess, train_step, docs, queries, \
                        env, explorer)
                batch_rewards.append(batch_reward)
                train_losses.append(loss)

            # Print some evaluation metrics on the validation set.
            val_accuracy, val_ndcgs = evaluate_model(model, dataset.validation, sess, num_queries)
            avg_val_ndcg = np.average(val_ndcgs)
            val_ndcgs[epoch_id] = avg_val_ndcg
            print("average train loss = %.3f    \taverage train batch reward = %.3f" % \
                    (np.average(train_losses), np.average(batch_rewards)))
            print("validation accuracy = %.3f\taverage validation NDCG = %.3f" % (val_accuracy, avg_val_ndcg))
            print("")

        return val_ndcgs

learning_rates = np.power(10., np.arange(-10, 2))
batch_sizes = [512, 1024]
#[1, 32, 128, 256, 512, 1024, 2056, 5012, 1024, 2048, 4096]

parser = argparse.ArgumentParser()
parser.add_argument("--k", type=int, default=2)
args = parser.parse_args()
print("Finding best parameters for k = %d" % args.k)
print("=========")

cur_best_val_ndcgs = [-1, -1, -1]
cur_best_settings = [(), (), ()]

for learning_rate, batch_size in itertools.product(learning_rates, batch_sizes):
    val_ndcgs = run_model(args.k, learning_rate, batch_size)
    best_val_ndcg = np.max(val_ndcgs)

    # Save the best 3 scores to see the variety.
    for i, cur_best_val_ndcg in enumerate(cur_best_val_ndcgs):
        if best_val_ndcg > cur_best_val_ndcg:
            if i < 2:
                cur_best_val_ndcgs[i+1] = cur_best_val_ndcgs[i]
                cur_best_settings[i+1] = cur_best_settings[i]

            cur_best_val_ndcgs[i] = best_val_ndcg
            cur_best_settings[i] = (learning_rate, batch_size)
            break

    # Print the current best setups.
    print("========")
    print("Currently the best setups are %s, which got scores of %s" % (cur_best_settings, cur_best_val_ndcgs))
    print("========")
