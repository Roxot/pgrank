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
print_freq = 20
test_freq = 100

# Hyperparameters
learning_rate = 3e-3
batch_size = 512
max_steps = 1000
epsilon = 0
epsilon_decay = epsilon / max_steps
num_hidden = 256
weight_reg_strength = 0.
bias_init = 0.1
decay = 0.9
optimizer = tf.train.AdamOptimizer(learning_rate)

# Setup graph
epx = tf.placeholder(tf.float32, [None, input_dim], name="epx")
epy = tf.placeholder(tf.float32, [None, num_actions], name="epy")
epr = tf.placeholder(tf.float32, [None, 1], name="epr")

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
# loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits, epy) * tf.transpose(epr))
# loss = tf.reduce_mean(-tf.reduce_sum(epy * tf.log(action_probs + 1e-10), reduction_indices=[1]) * tf.transpose(epr))
loss = tf.reduce_mean(-tf.reduce_sum(epy * action_probs, reduction_indices=[1]) * tf.transpose(epr))
train_step = optimizer.minimize(loss)

# Accuracy
true_labels = tf.placeholder(tf.int64, [None], name="true_labels")
y_pred = tf.argmax(logits, 1)
accuracy = tf.reduce_mean(tf.cast(tf.equal(y_pred, true_labels), tf.float32))

# Summaries
acc_summary = tf.scalar_summary("accuracy", accuracy)
loss_summary = tf.scalar_summary("loss", loss)

# Run the session
total_reward = 0
running_reward = None
with tf.Session() as sess:
    train_writer = tf.train.SummaryWriter("logs/pg" + "/train", sess.graph)
    val_writer = tf.train.SummaryWriter("logs/pg" + "/validation")
    test_writer = tf.train.SummaryWriter("logs/pg" + "/test")

    sess.run(tf.initialize_all_variables())
    env = MNISTEnvironment(batch_size)
    observation = env.reset()

    # Train for max_steps batches
    for iteration in range(max_steps):

        # Sample random actions
        aprobs = sess.run(action_probs, feed_dict={epx: observation})
        actions = [np.random.choice(num_actions, p=aprob_i) for aprob_i in aprobs]

        # Create fake labels
        ys = np.zeros((batch_size, num_actions))
        ys[np.arange(batch_size), actions] = 1.

        # Observe rewards, save the last observation for training
        xs = observation
        old_labels = env.current_labels
        observation, rs = env.step(actions)

        # Prepare training data
        rs = np.vstack(rs)
        # baseline = total_reward / (batch_size * float(iteration + 1)) # baseline as average
        # rs -= baseline

        # Train on the batch
        summary1, summary2, _ = sess.run([acc_summary, loss_summary, train_step], feed_dict={epx: xs, epy: ys, epr: rs, true_labels: old_labels})
        train_writer.add_summary(summary1, iteration)
        train_writer.add_summary(summary2, iteration)

        # Update epsilon and some statistics
        epsilon -= epsilon_decay
        total_reward += np.sum(rs)
        running_reward = np.sum(rs) / batch_size if running_reward is None else running_reward * 0.999 + np.sum(rs) / batch_size * 0.001

        if iteration % print_freq == 0 or iteration == max_steps - 1:
            train_loss, logs = sess.run([loss, logits], feed_dict={epx: xs, epy: ys, epr: rs})
            average_reward = total_reward / (batch_size * float(iteration + 1))
            print("Iteration %s/%s: Train Loss = %.6f, Running Reward = %.2f, Average Reward = %.2f" % \
                    (iteration, max_steps, train_loss, running_reward, average_reward))

        if iteration % test_freq == 0 or iteration == max_steps - 1:

            # Log the validation accuracy.
            val_images, val_labels = env.test_set()
            summary, val_acc = sess.run([acc_summary, accuracy], feed_dict={epx: val_images, true_labels: val_labels})
            val_writer.add_summary(summary, iteration)

            # Log and print the test accuracy
            test_images, test_labels = env.test_set()
            summary, test_acc = sess.run([acc_summary, accuracy], feed_dict={epx: test_images, true_labels: test_labels})
            test_writer.add_summary(summary, iteration)
            print("Iteration %s/%s: Test Accuracy = %s" % (iteration, max_steps, test_acc))

        # Plot the confusion matrix
        if iteration == max_steps - 1:
            test_images, test_labels = env.test_set()
            true_labels, predictions = sess.run([true_labels, y_pred], feed_dict={epx: test_images, \
                    true_labels: test_labels})
            cnf_matrix = confusion_matrix(true_labels, predictions)
            plt.figure()
            plot_confusion_matrix(cnf_matrix, classes=classes,
                title="confusion matrix on test dataset", cmap=plt.cm.Oranges)
            plt.show()
