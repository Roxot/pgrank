from tensorflow.examples.tutorials.mnist import input_data
from sklearn.metrics import confusion_matrix

from mlp import MLP
from plot import plot_confusion_matrix

import tensorflow as tf
import matplotlib.pyplot as plt

# Dataset
dataset = input_data.read_data_sets("MNIST_data/", one_hot=True)
classes = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]
input_dim = 784     # number of observations
num_classes = 10    # number of actions

# Hyperparameters
learning_rate = 0.1
max_steps = 1000
batch_size = 256
optimizer = tf.train.GradientDescentOptimizer(learning_rate);

# Setup graph
epx = tf.placeholder(tf.float32, [None, input_dim])
epy = tf.placeholder(tf.float32, [None, num_classes])
epr = tf.placeholder(tf.float32, [None, 1])


# Create a single layer neural net
input_dim = epx.get_shape()[1]
W = tf.get_variable("weights", shape=([input_dim, num_classes]),
                  initializer=tf.random_normal_initializer(1e-4),
                  regularizer=tf.no_regularizer(None))
b = tf.get_variable("biases", shape=([num_classes]), \
                  initializer=tf.constant_initializer(0.1))
logits = tf.matmul(epx, W) + b

# Define the loss, TODO
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits, epy), name="cross-entropy")

# Optimize the loss
train_step = optimizer.minimize(loss)

# Metrics
correct_preds = tf.equal(tf.argmax(logits, 1, name="predictions"), \
  tf.argmax(epy, 1, name="true_labels"), name="correct_preds")
accuracy = tf.reduce_mean(tf.cast(correct_preds, tf.float32), name="accuracy")

# Run the session
with tf.Session() as sess:
    sess.run(tf.initialize_all_variables())
    # env = MNISTEnvironment()
    # observation = env.reset()

    # Train for max_steps batches
    for iteration in range(max_steps):
        batch = dataset.train.next_batch(batch_size)
        _ = sess.run([train_step], feed_dict={epx: batch[0], epy: batch[1]})

        if iteration % 100 == 0 or iteration == max_steps - 1:
            train_acc, train_loss = sess.run([accuracy, loss], \
                    feed_dict={epx: batch[0], epy: batch[1]})
            print("Iteration %s/%s: Train Loss = %s, Train Accuracy = %s" % \
                    (iteration, max_steps, train_loss, train_acc))

    # Report the test accuracy
    y_pred = tf.argmax(logits, 1)
    y_true = tf.argmax(epy, 1)
    test_acc, y_pred, y_true = sess.run([accuracy, y_pred, y_true], \
            feed_dict={epx: dataset.test.images, epy: dataset.test.labels})
    print("Test accuracy: %s" % test_acc)

    # Plot the confusion matrix
    cnf_matrix = confusion_matrix(y_true, y_pred)
    plt.figure()
    plot_confusion_matrix(cnf_matrix, classes=classes,
            title="confusion matrix on test dataset", cmap=plt.cm.Oranges)
    plt.show()
