import numpy as np
import tensorflow as tf

import explorers

from tensorflow.contrib.learn.python.learn.datasets import mnist

from search.query import random_digit, random_from_docs
from search.reward import ndcg_full
from search import Environment
from models import PGRank

# Calculates the NDCG @ list for all possible queries and the
# classification accuracy of the model.
def evaluate_model(model, evaluation_set, sess, num_queries):
        images = evaluation_set.images
        labels = evaluation_set.labels

        # Prepare the input.
        x = np.reshape(images, (1, images.shape[0], images.shape[1]))
        labels = np.reshape(labels, (1, len(labels)))

        # Calculate the NDCG for each possible query.
        ndcgs = np.zeros(num_queries)
        for query_id in range(num_queries):

            # Create the query input.
            queries = np.array([[query_id]])

            # Retrieve deterministic ranking for the query, sorted by document score.
            feed_dict = { model.x: x, model.q: queries }
            ranking = sess.run(model.det_ranking, feed_dict=feed_dict)

            # Calculate the ndcg for this list.
            rel_labels = np.zeros(labels.shape)
            rel_labels[np.where(labels == queries)] = 1.
            ndcgs[query_id] = ndcg_full(ranking, rel_labels)[0, 0]

        # Calculate the accuracy of the model by classifying documents as the output
        # with the highest document score.
        feed_dict = { model.x: x, model.true_labels: labels }
        accuracy = sess.run(model.accuracy, feed_dict=feed_dict)[0]

        return accuracy, ndcgs

def run_model(k, average_over=10, learning_rate=1e-4, num_epochs=5, batch_size=64, h_dim=256, epsilon=0., baseline=0., weight_reg_str=0., exploration_type="uniform", grad_clip_norm=None, sample_weight_lower=None, sample_weight_upper=None, batch_normalization=True):

    # Hyperparameters
    query_fn = random_from_docs
    reward_fn = ndcg_full
    greedy_action = explorers.exploit.sample
    explore_action = explorers.explore.Oracle() if exploration_type == "oracle" else explorers.explore.Uniform()

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
    print("Explore action = %s" % explore_action)
    print("Baseline = %s" % baseline)
    print("Weight regularization strength = %s" % weight_reg_str)
    print("Sample weight lower bound = %s" % sample_weight_lower)
    print("Sample weight upper bound = %s" % sample_weight_upper)
    print("Gradient clipping norm = %s" % grad_clip_norm)
    print("Batch normalization = %s" % batch_normalization)
    print("")

    all_val_ndcgs = np.zeros(average_over)
    for iter_num in range(1, average_over+1):

        # Load the dataset.
        print("Loading dataset.")
        dataset = mnist.load_mnist("data/")
        data_dim = 784
        num_queries = 10
        print("")

        # Create the environment and explorer.
        env = Environment(dataset, k, batch_size, query_fn=query_fn, reward_fn=reward_fn)
        explorer = explorers.EpsGreedy(epsilon, greedy_action=greedy_action, explore_action=explore_action)

        # Create model and optimizer.
        optimizer = tf.train.AdamOptimizer(learning_rate)
        model = PGRank(data_dim, num_queries, h_dim, optimizer, reg_str=weight_reg_str, \
                grad_clip_norm=grad_clip_norm, batch_normalization=batch_normalization)
        val_ndcgs = np.zeros(num_epochs)

        # Train the model.
        with tf.Session() as sess:
            sess.run(tf.initialize_all_variables())

            val_accuracy, val_ndcg = evaluate_model(model, dataset.validation, sess, num_queries)
            avg_val_ndcg = np.average(val_ndcg)
            print("before training")
            print("validation accuracy = %.3f\t\taverage validation NDCG = %.3f\n" % (val_accuracy, avg_val_ndcg))

            for epoch_id in range(num_epochs):
                print("epoch %d" % (epoch_id + 1))

                train_losses = []
                batch_rewards = []
                for docs, queries in env.next_epoch():

                    # Train on the batch.
                    batch_reward, loss = model.train_on_batch(sess, docs, queries, \
                            env, explorer, env.rel_labels, baseline, \
                            sample_weight_lower=sample_weight_lower, \
                            sample_weight_upper=sample_weight_upper)
                    batch_rewards.append(batch_reward)
                    train_losses.append(loss)

                # Print some evaluation metrics on the validation set.
                val_accuracy, val_ndcg = evaluate_model(model, dataset.validation, sess, num_queries)
                avg_val_ndcg = np.average(val_ndcg)
                val_ndcgs[epoch_id] = avg_val_ndcg
                print("average train loss = %.6f    \taverage train batch reward = %.3f" % \
                        (np.average(train_losses), np.average(batch_rewards)))
                print("validation accuracy = %.3f\t\taverage validation NDCG = %.3f" % (val_accuracy, avg_val_ndcg))
                print("")

        # Save the maximum as if we were early stopping.
        all_val_ndcgs[iter_num-1] = np.max(val_ndcgs)

    # Return statistics over all runs.
    max_val_ndcg = np.max(all_val_ndcgs)
    min_val_ndcg = np.min(all_val_ndcgs)
    avg = np.average(all_val_ndcgs)
    std_dev = np.std(all_val_ndcgs)

    return avg, std_dev, min_val_ndcg, max_val_ndcg
