from tensorflow.contrib.layers import initializers
from tensorflow.contrib.layers import regularizers
from sklearn.metrics import confusion_matrix from mnistsearchenv import MNISTSearchEnvironment
from scipy.misc import logsumexp

import io
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import argparse

# Arguments
parser = argparse.ArgumentParser()
parser.add_argument("--k", type=int, default=2)
parser.add_argument("--lr", type=float, default=1e-3)
parser.add_argument("--log_dir", type=str, default=None)
parser.add_argument("--num_iter", type=int, default=10000)
args = parser.parse_args()

# Dataset parameters
classes = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]
input_dim = 784
num_queries = 10

# Run settings
print_freq = 20
test_freq = 100
k = args.k

# Hyperparameters
learning_rate = args.lr
batch_size = 512
max_steps = args.num_iter
epsilon = 0
epsilon_decay = epsilon / max_steps
num_hidden = 256
weight_reg_strength = 0.
bias_init = 0.1
decay = 0.9
optimizer = tf.train.AdamOptimizer(learning_rate)

# Setup graph
epx = tf.placeholder(tf.float32, [None, input_dim], name="epx")
epq = tf.placeholder(tf.float32, [num_queries], name="epq")
epr = tf.placeholder(tf.float32, name="epr")
epy = tf.placeholder(tf.float32, [k], name="epy")

# Create a single hidden layer neural net
input_dim = epx.get_shape()[1]
W = tf.get_variable("weights_1", shape=([input_dim, num_hidden]),
                  initializer=initializers.xavier_initializer(),
                  regularizer=regularizers.l2_regularizer(weight_reg_strength))
b = tf.get_variable("biases_1", shape=([num_hidden]), \
                  initializer=tf.constant_initializer(bias_init),
                  regularizer=regularizers.l2_regularizer(weight_reg_strength))
W2 = tf.get_variable("weights_2", shape=([num_hidden, num_queries]),
                  initializer=initializers.xavier_initializer(),
                  regularizer=regularizers.l2_regularizer(weight_reg_strength))
b2 = tf.get_variable("biases_2", shape=([num_queries]), \
                  initializer=tf.constant_initializer(bias_init),
                  regularizer=regularizers.l2_regularizer(weight_reg_strength))
h = tf.matmul(epx, W) + b
h = tf.nn.relu(h)
logits = tf.matmul(h, W2) + b2
# scores = tf.reduce_sum(tf.nn.softmax(logits) * epq, reduction_indices=[1])
scores = tf.reduce_sum(logits * epq, reduction_indices=[1])
action_probs = tf.nn.softmax(scores)

# Loss
# loss = -tf.reduce_sum(tf.mul(epy, action_probs)) * epr
loss = -tf.reduce_sum(tf.mul(epy, scores)) * epr
train_step = optimizer.minimize(loss)

# Accuracy
true_labels = tf.placeholder(tf.int64, [None], name="true_labels")
y_pred = tf.argmax(logits, 1)
accuracy = tf.reduce_mean(tf.cast(tf.equal(y_pred, true_labels), tf.float32))

# Summaries
loss_summary = tf.scalar_summary("loss", loss)
merged = tf.merge_all_summaries()

def softmax(a):
    log_Z = logsumexp(a)
    log_softmax = a - log_Z
    return np.exp(log_softmax)

def sample_ranking(output_scores):
    k = len(output_scores)
    P = np.zeros((k, k))
    ranking = np.zeros(k, dtype=np.int)
    docs = np.arange(k)
    temp_outputs = np.copy(output_scores)
    for i in range(k):
        probs = softmax(temp_outputs)
        P[i, i:k] = probs
        action = np.random.choice(len(probs), p=probs)
        ranking[i] = docs[action]
        temp_outputs = np.delete(temp_outputs, action)
        docs = np.delete(docs, action)
    return ranking, P

def calculate_derivatives(ranking, P):
    k = len(ranking)
    derivatives = np.zeros(k)
    p_list = np.prod(np.diag(P))
    for i in range(k):
        M = np.zeros((k, k))
        M[:i, i] = -1.
        M[i, :] = 1.
        M[i, i] = 0.
        derivatives[i] = np.sum(M * P) * p_list
    return derivatives[ranking]

# Run the session
total_reward = 0
running_reward = None
average_reward = 0
with tf.Session() as sess:
    if args.log_dir is not None:
        train_writer = tf.train.SummaryWriter("logs/%s" % args.log_dir, sess.graph)

    sess.run(tf.initialize_all_variables())
    env = MNISTSearchEnvironment(k)
    images, query = env.reset()

    # Train for max_steps batches
    for iteration in range(max_steps):

        # Calculate action probabilities, sample an action
        query_onehot = np.zeros(num_queries)
        query_onehot[query] = 1
        net_out = sess.run(scores, {epx: images, epq: query_onehot})

        ranking, P = sample_ranking(net_out)
        # print("Ranking: %s,\nP =\n%s" % (ranking, P))
        derivatives = calculate_derivatives(ranking, P)

        # Observe the reward.
        observation, reward = env.step(ranking)
        old_images = images
        images, query = observation

        # Update statistics
        total_reward += reward
        running_reward = reward if running_reward is None else 0.99 * running_reward + 0.01 * reward
        average_reward = total_reward / (iteration + 1.0)

        sess.run(train_step, {epx: old_images, epq: query_onehot, epy: derivatives, epr: reward - average_reward})

        if iteration % print_freq == 0 or iteration == max_steps - 1:
            train_loss, summary = sess.run([loss, loss_summary], \
                    {epx: old_images, epq: query_onehot, epy: derivatives, epr: reward - average_reward})
            if args.log_dir is not None:
                train_writer.add_summary(summary, iteration)

            # Reward summaries
            if iteration != 0:
                avg_reward_summary = tf.Summary(value=[tf.Summary.Value(tag="average_reward", \
                        simple_value=average_reward)])
                running_reward_summary = tf.Summary(value=[tf.Summary.Value(tag="running_reward", \
                        simple_value=running_reward)])
                if args.log_dir is not None:
                    train_writer.add_summary(avg_reward_summary, iteration)
                    train_writer.add_summary(running_reward_summary, iteration)

            print("Iteration %d/%d: training loss = %.6f, average reward = %f, running reward = %f" % (iteration, \
                    max_steps, train_loss, average_reward, running_reward))

        if iteration % test_freq == 0 or iteration == max_steps - 1:
            test_images, test_labels = env.test_set()

            # Calculate the ndcg for all possible queries on the entire
            # test set
            sum_ndcg = 0
            for q in range(num_queries):
                query_onehot = np.zeros(num_queries)
                query_onehot[q] = 1
                rel = (test_labels == q).astype(int)
                s = sess.run(scores, {epx: test_images, epq: query_onehot})
                ranking = rel[np.argsort(s)[::-1]]
                idcg = env._get_idcg(rel, len(test_labels))
                ndcg = env._ndcg_at_k(ranking, len(test_labels), idcg)
                sum_ndcg += ndcg

            avg_ndcg = sum_ndcg / float(num_queries)
            avg_ndcg_summary = tf.Summary(value=[tf.Summary.Value(tag="average_ndcg", \
                    simple_value=avg_ndcg)])
            if args.log_dir is not None:
                train_writer.add_summary(avg_ndcg_summary, iteration)

            test_acc = sess.run(accuracy, {epx: test_images, true_labels: test_labels})
            test_acc_summary = tf.Summary(value=[tf.Summary.Value(tag="accuracy", \
                    simple_value=float(test_acc))])
            if args.log_dir is not None:
                train_writer.add_summary(test_acc_summary, iteration)

            print("Iteration %d/%d: test ndcg = %.6f test accuracy = %.2f" % (iteration, max_steps, avg_ndcg, test_acc))

    # Show top 10 images
    test_images, test_labels = env.test_set()
    for q in range(num_queries):
        query_onehot = np.zeros(num_queries)
        query_onehot[q] = 1
        s = sess.run(scores, {epx: test_images, epq: query_onehot})
        ranking = np.argsort(s)[::-1]
        images = test_images[ranking[:10]]
        for i, image in enumerate(images):
            plt.subplot(10, num_queries, (num_queries * i) + 1 + q)
            if i == 0:
                plt.title("%d" % q)
            plt.axis('off')
            plt.imshow(image.reshape(28, 28), cmap="gray")

    plt.suptitle('Top 10 images for different queries')

    # Write the plot to tensorboard
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    image = tf.image.decode_png(buf.getvalue(), channels=4)
    image = tf.expand_dims(image, 0)
    image_summary = tf.image_summary("top_10_images", image)
    if args.log_dir is not None:
        train_writer.add_summary(sess.run(image_summary))
