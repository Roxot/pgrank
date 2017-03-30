from tensorflow.contrib.layers import initializers
from tensorflow.contrib.layers import regularizers
from sklearn.metrics import confusion_matrix

from plot import plot_confusion_matrix
from mnistenv import MNISTEnvironment

import itertools
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

# Grid search
learning_rates = [1e-1, 1e-2, 1e-3, 1e-4, 1e-5]
batch_sizes = [64, 128, 256, 512, 1024]
num_hiddens = [32, 64, 128, 256, 512]
weight_reg_strengths = [1e-1, 1e-2, 1e-3, 1e-4, 1e-5]
optimizers = ["rms", "sgd", "adam"]

# Hyperparameters
# learning_rate = 0.1 # 1e-3
# batch_size = 128
max_steps = 500
# epsilon = 0
# epsilon_decay = epsilon / max_steps
# num_hidden = 128
# weight_reg_strength = 1e-3
bias_init = 0.1
# decay = 0.9
# # optimizer = tf.train.RMSPropOptimizer(learning_rate, decay=decay);
# optimizer = tf.train.GradientDescentOptimizer(learning_rate)

best_val_acc = 0.
best_params = ()

i = 0
for opt_name, learning_rate, batch_size, num_hidden, weight_reg_strength in itertools.product(optimizers, learning_rates, batch_sizes, num_hiddens, weight_reg_strengths):
    i += 1

    optimizer = None
    if opt_name == "sgd":
        optimizer = tf.train.GradientDescentOptimizer(learning_rate)
    elif opt_name == "rms":
        optimizer = tf.train.RMSPropOptimizer(learning_rate)
    elif opt_name == "adam":
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)

    # Setup graph
    epx = tf.placeholder(tf.float32, [None, input_dim], name="epx")
    epy = tf.placeholder(tf.float32, [None, num_actions], name="epy")
    epr = tf.placeholder(tf.float32, [None, 1], name="epr")

    # Create a single hidden layer neural net
    input_dim = epx.get_shape()[1]
    W = tf.get_variable("weights_1_%s" % i, shape=([input_dim, num_hidden]),
                      initializer=initializers.xavier_initializer(),
                      regularizer=regularizers.l2_regularizer(weight_reg_strength))
    b = tf.get_variable("biases_1_%s" % i, shape=([num_hidden]), \
                      initializer=tf.constant_initializer(bias_init),
                      regularizer=regularizers.l2_regularizer(weight_reg_strength))
    W2 = tf.get_variable("weights_2_%s" % i, shape=([num_hidden, num_actions]),
                      initializer=initializers.xavier_initializer(),
                      regularizer=regularizers.l2_regularizer(weight_reg_strength))
    b2 = tf.get_variable("biases_2_%s" % i, shape=([num_actions]), \
                      initializer=tf.constant_initializer(bias_init),
                      regularizer=regularizers.l2_regularizer(weight_reg_strength))
    h = tf.matmul(epx, W) + b
    h = tf.nn.relu(h)
    logits = tf.matmul(h, W2) + b2
    action_probs = tf.nn.softmax(logits)

    # Promote actions we already took, then multiply it with the rewards,
    # such that we only really promote actions that yielded good reward.
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits, epy) * tf.transpose(epr))
    train_step = optimizer.minimize(loss)

    # Accuracy
    true_labels = tf.placeholder(tf.int64, [None], name="true_labels")
    y_pred = tf.argmax(logits, 1)
    accuracy = tf.reduce_mean(tf.cast(tf.equal(y_pred, true_labels), tf.float32))

    # Run the session
    total_reward = 0
    running_reward = None
    with tf.Session() as sess:
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
            observation, rs = env.step(actions)

            # Prepare training data
            rs = np.vstack(rs)

            # Train on the batch
            _ = sess.run([train_step], feed_dict={epx: xs, epy: ys, epr: rs})

            # Update epsilon and some statistics
            # epsilon -= epsilon_decay
            total_reward += np.sum(rs)
            running_reward = np.sum(rs) / batch_size if running_reward is None else running_reward * 0.999 + np.sum(rs) / batch_size * 0.001

            if iteration % print_freq == 0 or iteration == max_steps - 1:
                train_loss, logs = sess.run([loss, logits], feed_dict={epx: xs, epy: ys, epr: rs})
                average_reward = total_reward / (batch_size * float(iteration + 1))
                print("Iteration %s/%s: Train Loss = %.6f, Running Reward = %.2f, Average Reward = %.2f" % \
                        (iteration, max_steps, train_loss, running_reward, average_reward))

            # if iteration % test_freq == 0 or iteration == max_steps - 1:
            #     test_images, test_labels = env.test_set()
            #     test_acc = sess.run([accuracy], feed_dict={epx: test_images, true_labels: test_labels})
            #     print("Iteration %s/%s: Test Accuracy = %s" % (iteration, max_steps, test_acc))

            # Plot the confusion matrix
            # if iteration == max_steps - 1:
            #     test_images, test_labels = env.test_set()
            #     true_labels, predictions = sess.run([true_labels, y_pred], feed_dict={epx: test_images, \
            #             true_labels: test_labels})
            #     cnf_matrix = confusion_matrix(true_labels, predictions)
            #     plt.figure()
            #     plot_confusion_matrix(cnf_matrix, classes=classes,
            #         title="confusion matrix on test dataset", cmap=plt.cm.Oranges)
            #     plt.show()


        val_images, val_labels = env.validation_set()
        val_acc = sess.run([accuracy], feed_dict={epx: val_images, true_labels: val_labels})
        print("Validation Accuracy = %s for (%s, %s, %s, %s, %s)" % (val_acc, opt_name, learning_rate, batch_size, weight_reg_strength, num_hidden))
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_params = (opt_name, learning_rate, batch_size, weight_reg_strength, num_hidden)
        print("Best params so far: %s with val accuracy %s in order (opt_name, learning_rate, batch_size, weight_reg_strength, num_hidden)" % (best_params, best_val_acc))

print("Best params: %s with val accuracy %s in order (opt_name, learning_rate, batch_size, weight_reg_strength, num_hidden)" % (best_params, best_val_acc))
