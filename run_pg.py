import numpy as np
import tensorflow as tf
import argparse

import explorers

from tensorflow.contrib.learn.python.learn.datasets import mnist

from search.query import random_digit, random_from_docs
from search.reward import ndcg_full
from search import Environment
from models import PGRank
from utils import evaluate_model, mnist_image_summary, write_train_summaries, write_evaluation_summaries

# Arguments
parser = argparse.ArgumentParser()
parser.add_argument("--k", type=int, default=2)
parser.add_argument("--learning_rate", type=float, default=1e-4)
parser.add_argument("--log_dir", type=str, default=None)
parser.add_argument("--num_epochs", type=int, default=10)
parser.add_argument("--batch_size", type=int, default=1024)
parser.add_argument("--h_dim", type=int, default=256)
parser.add_argument("--epsilon", type=float, default=0.)
args = parser.parse_args()

# Hyperparameters
query_fn = random_from_docs
reward_fn = ndcg_full
greedy_action = explorers.exploit.sample

# Print hyperparameters.
print("Hyperparameters:")
print("k = %d" % args.k)
print("Batch size = %d" % args.batch_size)
print("Num epochs = %d" % args.num_epochs)
print("Learning rate = %f" % args.learning_rate)
print("Epsilon = %f" % args.epsilon)
print("Hidden layer dimension = %d" % args.h_dim)
print("Log dir = %s" % args.log_dir)
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
env = Environment(dataset, args.k, args.batch_size, query_fn=query_fn, reward_fn=reward_fn)
explorer = explorers.EpsGreedy(args.epsilon, greedy_action=greedy_action)

# Create model and optimizer.
model = PGRank(data_dim, num_queries, args.h_dim)
optimizer = tf.train.AdamOptimizer(args.learning_rate)
train_step = optimizer.minimize(model.loss)

# Train the model.
with tf.Session() as sess:
    sess.run(tf.initialize_all_variables())

    if args.log_dir is not None:
        summ_writer = tf.train.SummaryWriter("logs/%s" % args.log_dir, sess.graph)

    iteration = 0
    for epoch_id in range(args.num_epochs):
        print("epoch %d" % (epoch_id + 1))

        batch_id = 1
        for docs, queries in env.next_epoch():

            # Train on the batch.
            batch_reward, loss = model.train_on_batch(sess, train_step, docs, queries, \
                    env, explorer)
            print("batch %02d: loss = %.3f    \taverage batch reward = %.3f" % (batch_id, loss, batch_reward))

            # Administration.
            if args.log_dir is not None:
                write_train_summaries(summ_writer, iteration, batch_reward, loss)
            batch_id += 1
            iteration += 1

        # Print some evaluation metrics on the validation set.
        val_accuracy, val_ndcgs = evaluate_model(model, dataset.validation, sess, num_queries)
        avg_val_ndcg = np.average(val_ndcgs)
        print("validation accuracy = %.3f\taverage validation NDCG = %.3f" % (val_accuracy, avg_val_ndcg))

        # Write evaluation summaries.
        if args.log_dir is not None:
            write_evaluation_summaries(summ_writer, epoch_id, val_accuracy, avg_val_ndcg)
        print("")

    test_accuracy, test_ndcgs = evaluate_model(model, dataset.test, sess, num_queries)
    avg_test_ndcg = np.average(test_ndcgs)
    print("test accuracy = %.3f     \taverage test NDCG = %.3f" % (test_accuracy, avg_test_ndcg))

    if args.log_dir is not None:
        img_summary = mnist_image_summary(model, dataset.test, sess)
        summ_writer.add_summary(sess.run(img_summary))
