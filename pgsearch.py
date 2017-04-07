from tensorflow.contrib.layers import initializers
from tensorflow.contrib.layers import regularizers
from sklearn.metrics import confusion_matrix

from mnistsearchenv import MNISTSearchEnvironment

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# Dataset parameters
classes = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]
input_dim = 784
num_queries = 10

# Run settings
print_freq = 20
test_freq = 100
k = 2

# Hyperparameters
learning_rate = 3e-3
batch_size = 512
max_steps = 5000
epsilon = 0
epsilon_decay = epsilon / max_steps
num_hidden = 256
weight_reg_strength = 0.
bias_init = 0.1
decay = 0.9
optimizer = tf.train.AdamOptimizer(learning_rate)

# Setup graph
epx = tf.placeholder(tf.float32, [k, input_dim], name="epx")
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
loss = -tf.reduce_sum(tf.mul(epy, action_probs)) * epr
train_step = optimizer.minimize(loss)

# Summaries
loss_summary = tf.scalar_summary("loss", loss)
merged = tf.merge_all_summaries()

# Run the session
total_reward = 0
running_reward = None
average_reward = 0
with tf.Session() as sess:
    # train_writer = tf.train.SummaryWriter("logs/pgsearch/" + "/train", sess.graph)

    sess.run(tf.initialize_all_variables())
    env = MNISTSearchEnvironment(k)
    images, query = env.reset()

    # Train for max_steps batches
    for iteration in range(max_steps):

        # Calculate action probabilities, sample an action
        query_onehot = np.zeros(num_queries)
        query_onehot[query] = 1
        aprobs = sess.run(action_probs, {epx: images, epq: query_onehot})
        action = np.random.choice(k, p=aprobs)
        action_onehot = np.zeros(k)
        action_onehot[action] = 1

        # Observe the reward.
        observation, reward = env.step(action)
        old_images = images
        images, query = observation

        # Update statistics
        total_reward += reward
        running_reward = reward if running_reward is None else 0.99 * running_reward + 0.01 * reward
        average_reward = total_reward / (iteration + 1.0)

        sess.run(train_step, {epx: old_images, epq: query_onehot, epy: action_onehot, epr: reward - average_reward})

        if iteration % print_freq == 0 or iteration == max_steps - 1:
            train_loss, summary = sess.run([loss, loss_summary], \
                    {epx: old_images, epq: query_onehot, epy: action_onehot, epr: reward - average_reward})
            # train_writer.add_summary(summary, iteration)

            # Reward summaries
            if iteration != 0:
                avg_reward_summary = tf.Summary(value=[tf.Summary.Value(tag="average_reward", simple_value=average_reward)])
                running_reward_summary = tf.Summary(value=[tf.Summary.Value(tag="running_reward", \
                        simple_value=running_reward)])
                # train_writer.add_summary(avg_reward_summary, iteration)
                # train_writer.add_summary(running_reward_summary, iteration)

            print("Iteration %d/%d: training loss = %.6f, average reward = %f, running reward = %f" % (iteration, max_steps, train_loss, average_reward, running_reward))
