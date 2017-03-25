from tensorflow.contrib.layers import initializers
from tensorflow.contrib.layers import regularizers
from sklearn.metrics import confusion_matrix

from plot import plot_confusion_matrix
from mnistenv import MNISTEnvironment

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# Dataset parameters
classes = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]
input_dim = 784     # number of observations
num_actions = 10

# Run settings
print_freq = 100
test_freq = 1000

# Hyperparameters
learning_rate = 1e-3
decay = 0.9
max_steps = 1000
epsilon = 0
epsilon_decay = epsilon / max_steps
num_hidden = 128
weight_reg_strength = 1e-3
bias_init = 0.1
optimizer = tf.train.RMSPropOptimizer(learning_rate, decay=decay);

# Setup graph
epx = tf.placeholder(tf.float32, [None, input_dim], name="epx")
epy = tf.placeholder(tf.float32, [None, num_actions], name="epy")
epr = tf.placeholder(tf.float32, name="epr")

# Create a single hidden layer neural net
input_dim = epx.get_shape()[1]
W = tf.get_variable("weights_1", shape=([input_dim, num_hidden]),
                  initializer=initializers.xavier_initializer(),
                  regularizer=regularizers.l2_regularizer(weight_reg_strength))
b = tf.get_variable("biases_1", shape=([num_hidden]), \
                  initializer=tf.constant_initializer(bias_init),
                  regularizer=regularizers.l2_regularizer(weight_reg_strength))
W2 = tf.get_variable("weights_2", shape=([num_hidden, num_actions]),
                  initializer=initializers.xavier_initializer(),
                  regularizer=regularizers.l2_regularizer(weight_reg_strength))
b2 = tf.get_variable("biases_2", shape=([num_actions]), \
                  initializer=tf.constant_initializer(bias_init),
                  regularizer=regularizers.l2_regularizer(weight_reg_strength))
h = tf.matmul(epx, W) + b
h = tf.nn.relu(h)
logits = tf.matmul(h, W2) + b2
action_probs = tf.nn.softmax(logits)

# Promote actions we already took, then multiply it with the rewards,
# such that we only really promote actions that yielded good reward.
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits, epy)) * epr
train_step = optimizer.minimize(loss)

# Accuracy
true_labels = tf.placeholder(tf.float32, [None, num_actions], name="true_labels")
y_pred = tf.argmax(logits, 1)
y_true = tf.argmax(true_labels, 1)
accuracy = tf.reduce_mean(tf.cast(tf.equal(y_pred, y_true), tf.float32))

# Run the session
total_reward = 0
running_reward = None
with tf.Session() as sess:
    sess.run(tf.initialize_all_variables())
    env = MNISTEnvironment()
    observation = env.reset()

    # Train for max_steps batches
    for iteration in range(max_steps):

        # With probability epsilon take a random action, otherwise sample an action
        # from the action probabilities
        action = None
        aprobs = sess.run(action_probs, feed_dict={epx: observation})[0]
        if np.random.uniform() < epsilon:
            action = np.random.randint(num_actions)
        else:
            action = np.random.choice(num_actions, p=aprobs)
        fake_label = np.zeros_like(aprobs)
        fake_label[action] = 1

        # Use the old observation and the fake label as training data
        x = observation
        y = np.vstack([fake_label])

        # Take the action, observe the direct reward and a new observation
        observation, reward = env.step(action)

        # Train on the sample, also take into account the reward, prepare
        # the data for being fed to the network
        _ = sess.run([train_step], feed_dict={epx: x, epy: y, epr: reward})

        # Update epsilon and some statistics
        epsilon -= epsilon_decay
        total_reward += reward
        running_reward = reward if running_reward is None else running_reward * 0.999 + reward * 0.001

        if iteration % print_freq == 0 or iteration == max_steps - 1:
            train_loss, logs = sess.run([loss, logits], feed_dict={epx: x, epy: y, epr: reward})
            average_reward = total_reward / float(iteration + 1)
            print("Iteration %s/%s: Train Loss = %.6f, Running Reward = %.2f, Average Reward = %.2f" % \
                    (iteration, max_steps, train_loss, running_reward, average_reward))

        if iteration % test_freq == 0 or iteration == max_steps - 1:
            test_images, test_labels = env.test_set()
            test_acc = sess.run([accuracy], feed_dict={epx: test_images, true_labels: test_labels})
            print("Iteration %s/%s: Test Accuracy = %s" % (iteration, max_steps, test_acc))

        # Plot the confusion matrix
        if iteration == max_steps - 1:
            test_images, test_labels = env.test_set()
            true_labels, predictions = sess.run([y_true, y_pred], feed_dict={epx: test_images, \
                    true_labels: test_labels})
            cnf_matrix = confusion_matrix(true_labels, predictions)
            plt.figure()
            plot_confusion_matrix(cnf_matrix, classes=classes,
                title="confusion matrix on test dataset", cmap=plt.cm.Oranges)
            plt.show()
