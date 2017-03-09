from tensorflow.examples.tutorials.mnist import input_data
from sklearn.metrics import confusion_matrix

from plot import plot_confusion_matrix

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

class MNISTEnvironment:

    POS_REWARD = 1
    NEG_REWARD = -1

    def __init__(self):
        self.dataset = input_data.read_data_sets("MNIST_data/", one_hot=True)
        # self.action_space = spaces.Discrete(10)
        # self.observation_space = spaces.Box(low=0, high=255, shape=(784,))

    def _observe(self):
        batch = self.dataset.train.next_batch(1)
        observation = batch[0]
        label_onehot = batch[1][0]
        label = np.where(label_onehot == 1)[0][0]
        return observation, label

    def step(self, label):
        reward = self.POS_REWARD if label == self.current_label else self.NEG_REWARD
        new_observation, self.current_label = self._observe()
        return new_observation, reward

    def reset(self):
        observation, self.current_label = self._observe()
        return observation

    def test_set(self):
        return self.dataset.test.images, self.dataset.test.labels

# Dataset parameters
# classes = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]
input_dim = 784     # number of observations
num_actions = 10

# Run settings
print_freq = 5000
test_freq = 10000

# Hyperparameters
learning_rate = 1e-2
decay = 0.9
max_steps = 50000
num_hidden = 128
batch_size = 256
optimizer = tf.train.RMSPropOptimizer(learning_rate, decay=decay);

# Setup graph
epx = tf.placeholder(tf.float32, [None, input_dim])
epy = tf.placeholder(tf.float32, [None, num_actions])
epr = tf.placeholder(tf.float32)

# Create a single hidden layer neural net
input_dim = epx.get_shape()[1]
W = tf.get_variable("weights_1", shape=([input_dim, num_hidden]),
                  initializer=tf.random_normal_initializer(1e-4),
                  regularizer=tf.no_regularizer(None))
b = tf.get_variable("biases_1", shape=([num_hidden]), \
                  initializer=tf.constant_initializer(0.1))
W2 = tf.get_variable("weights_2", shape=([num_hidden, num_actions]),
                  initializer=tf.random_normal_initializer(1e-4),
                  regularizer=tf.no_regularizer(None))
b2 = tf.get_variable("biases_2", shape=([num_actions]), \
                  initializer=tf.constant_initializer(0.1))
h = tf.matmul(epx, W) + b
h = tf.nn.relu(h)
logits = tf.matmul(h, W2) + b2
action_probs = tf.nn.softmax(logits)

# Promote actions we already took, then multiply the gradient with the rewards,
# such that we only really promote actions that yielded good reward.
loss = tf.nn.l2_loss(epy - action_probs)
#loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits, epy))
grads = optimizer.compute_gradients(loss)#, grad_loss=epr)
train_step = optimizer.apply_gradients(grads)

# Accuracy
true_labels = tf.placeholder(tf.float32, [None, num_actions])
y_pred = tf.argmax(logits, 1)
y_true = tf.argmax(true_labels, 1)
accuracy = tf.reduce_mean(tf.cast(tf.equal(y_pred, y_true), tf.float32))

# Run the session
total_reward = 0
with tf.Session() as sess:
    sess.run(tf.initialize_all_variables())
    env = MNISTEnvironment()
    observation = env.reset()

    # Train for max_steps batches
    for iteration in range(max_steps):

        # Calculate the action probabilities, sample an action
        aprobs = sess.run(action_probs, feed_dict={epx: observation})[0]
        action = np.random.choice(num_actions, p=aprobs)
        fake_label = None

        # Use the old observation and the fake label as training data
        x = observation
        # y = np.vstack([fake_label])

        # Take the action, observe the direct reward and a new observation
        observation, reward = env.step(action)
        total_reward += reward
        if reward == 1:
            fake_label = np.zeros_like(aprobs)
            fake_label[action] = 1
        else:
            fake_label = np.ones_like(aprobs)
            fake_label[np.where(fake_label == 1)] = 1./9
            fake_label[action] = 0
        y = np.vstack([fake_label])

        # Train on the sample, also take into account the reward, prepare
        # the data for being fed to the network
        _ = sess.run([train_step], feed_dict={epx: x, epy: y, epr: reward})

        if iteration % print_freq == 0 or iteration == max_steps - 1:
            train_loss = sess.run([loss], feed_dict={epx: x, epy: y, epr: reward})
            average_reward = total_reward / float(iteration + 1)
            print("Iteration %s/%s: Train Loss = %s, Average Reward = %s" % \
                    (iteration, max_steps, train_loss, average_reward))

        if iteration % test_freq == 0 or iteration == max_steps - 1:
            test_images, test_labels = env.test_set()
            test_acc = sess.run([accuracy], feed_dict={epx: test_images, true_labels: test_labels})
            print("Iteration %s/%s: Test Accuracy = %s" % (iteration, max_steps, test_acc))
