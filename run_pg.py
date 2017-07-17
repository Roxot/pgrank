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
parser.add_argument("--batch_size", type=int, default=64)
parser.add_argument("--h_dim", type=int, default=256)
parser.add_argument("--epsilon", type=float, default=0.)
parser.add_argument("--baseline", type=float, default=0.)
parser.add_argument("--weight_reg_str", type=float, default=0.)
parser.add_argument("--exploration_type", type=str, default="uniform")
parser.add_argument("--grad_clip_norm", type=float, default=5.0)
parser.add_argument("--sample_weight_lower", type=float, default=0.01)
parser.add_argument("--sample_weight_upper", type=float, default=5.0)
parser.add_argument("--batch_normalization", type=str, default="True")
args = parser.parse_args()

# Hyperparameters
query_fn = random_from_docs
reward_fn = ndcg_full
greedy_action = explorers.exploit.sample
explore_action = explorers.explore.Oracle() if args.exploration_type == "oracle" else explorers.explore.Uniform()
batch_normalization = True if args.batch_normalization.lower() == "true" else False

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
print("Explore action = %s" % explore_action)
print("Baseline = %s" % args.baseline)
print("Weight regularization strength = %s" % args.weight_reg_str)
print("Sample weight lower bound = %s" % args.sample_weight_lower)
print("Sample weight upper bound = %s" % args.sample_weight_upper)
print("Gradient clipping norm = %s" % args.grad_clip_norm)
print("Batch normalization = %s" % batch_normalization)
print("")

# Load the dataset.
print("Loading dataset.")
dataset = mnist.load_mnist("data/")
data_dim = 784
num_queries = 10
print("")

# Create the environment and explorer.
env = Environment(dataset, args.k, args.batch_size, query_fn=query_fn, reward_fn=reward_fn)
explorer = explorers.EpsGreedy(args.epsilon, greedy_action=greedy_action, explore_action=explore_action)

# Create model and optimizer.
optimizer = tf.train.AdamOptimizer(args.learning_rate)
model = PGRank(data_dim, num_queries, args.h_dim, optimizer, reg_str=args.weight_reg_str, \
        grad_clip_norm=args.grad_clip_norm, batch_normalization=batch_normalization)

# Train the model.
with tf.Session() as sess:
    sess.run(tf.initialize_all_variables())

    if args.log_dir is not None:
        summ_writer = tf.train.SummaryWriter("logs/%s" % args.log_dir, sess.graph)

    val_accuracy, val_ndcgs = evaluate_model(model, dataset.validation, sess, num_queries)
    avg_val_ndcg = np.average(val_ndcgs)
    if args.log_dir is not None:
        write_evaluation_summaries(summ_writer, 0, val_accuracy, avg_val_ndcg)
    print("before training")
    print("validation accuracy = %.3f\t\taverage validation NDCG = %.3f\n" % (val_accuracy, avg_val_ndcg))

    iteration = 0
    for epoch_id in range(1, args.num_epochs + 1):
        print("epoch %d" % epoch_id)

        batch_id = 1
        train_losses = []
        batch_rewards = []
        for docs, queries in env.next_epoch():

            # Train on the batch.
            batch_reward, loss = model.train_on_batch(sess, docs, queries, \
                    env, explorer, env.rel_labels, args.baseline, \
                    sample_weight_lower=args.sample_weight_lower, \
                    sample_weight_upper=args.sample_weight_upper)
            batch_rewards.append(batch_reward)
            train_losses.append(loss)

            # Administration.
            if args.log_dir is not None:
                write_train_summaries(summ_writer, iteration, batch_reward, loss)
            batch_id += 1
            iteration += 1

        # Print some evaluation metrics on the validation set.
        val_accuracy, val_ndcgs = evaluate_model(model, dataset.validation, sess, num_queries)
        avg_val_ndcg = np.average(val_ndcgs)
        print("average train loss = %.6f    \taverage train batch reward = %.3f" % \
                (np.average(train_losses), np.average(batch_rewards)))
        print("validation accuracy = %.3f\t\taverage validation NDCG = %.3f" % (val_accuracy, avg_val_ndcg))

        # Write evaluation summaries.
        if args.log_dir is not None:
            write_evaluation_summaries(summ_writer, epoch_id, val_accuracy, avg_val_ndcg)
        print("")

    test_accuracy, test_ndcgs = evaluate_model(model, dataset.test, sess, num_queries)
    avg_test_ndcg = np.average(test_ndcgs)
    print("test accuracy = %.3f     \t\taverage test NDCG = %.3f" % (test_accuracy, avg_test_ndcg))

    if args.log_dir is not None:
        img_summary = mnist_image_summary(model, dataset.test, sess)
        summ_writer.add_summary(sess.run(img_summary))
