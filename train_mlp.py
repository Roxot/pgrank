from tensorflow.examples.tutorials.mnist import input_data
from sklearn.metrics import confusion_matrix
from tensorflow.contrib.layers import initializers
from tensorflow.contrib.layers import regularizers

from mlp import MLP
from plot import plot_confusion_matrix

import tensorflow as tf
import matplotlib.pyplot as plt

if __name__ == '__main__':

    # Dataset
    dataset = input_data.read_data_sets("MNIST_data/", one_hot=True)
    classes = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]
    input_dim = 784
    num_classes = 10
    test_freq = 100
    print_freq = 20

    # Hyperparameters
    num_hidden = [256]
    learning_rate = 3e-3
    max_steps = 5000
    batch_size = 512
    dropout_rate = 0.
    activation_fn = tf.nn.relu
    weight_initializer = initializers.xavier_initializer()
    weight_reg_strength = 0.
    weight_regularizer = regularizers.l2_regularizer(weight_reg_strength)
    optimizer = tf.train.AdamOptimizer(learning_rate)

    # Setup graph
    is_training = tf.placeholder(tf.bool)
    x = tf.placeholder(tf.float32, [None, input_dim])
    y = tf.placeholder(tf.float32, [None, num_classes])
    mlp = MLP(n_classes=num_classes, is_training=is_training, \
            n_hidden=num_hidden, dropout_rate=dropout_rate, \
            activation_fn=activation_fn, \
            weight_regularizer=weight_regularizer)
    logits = mlp.inference(x)
    loss = mlp.loss(logits, y)
    train_step = optimizer.minimize(loss)
    accuracy = mlp.accuracy(logits, y)
    merged = tf.merge_all_summaries()

    # Run the session
    with tf.Session() as sess:
        train_writer = tf.train.SummaryWriter("logs/mlp" + "/train", sess.graph)
        val_writer = tf.train.SummaryWriter("logs/mlp" + "/validation")
        test_writer = tf.train.SummaryWriter("logs/mlp" + "/test")
        sess.run(tf.initialize_all_variables())

        # Train for max_steps batches
        for iteration in range(max_steps):
            batch = dataset.train.next_batch(batch_size)
            summary, _ = sess.run([merged, train_step], \
                    feed_dict={x: batch[0], y: batch[1], is_training: True})
            train_writer.add_summary(summary, iteration)

            if iteration % print_freq == 0 or iteration == max_steps - 1:
                train_acc, train_loss = sess.run([accuracy, loss], \
                        feed_dict={x: batch[0], y: batch[1], is_training: False})
                print("Iteration %s/%s: Train Loss = %s, Train Accuracy = %s" % \
                        (iteration, max_steps, train_loss, train_acc))

            if iteration % test_freq == 0 or iteration == max_steps - 1:
                summary = sess.run(merged, \
                        feed_dict={x: dataset.validation.images, y: dataset.validation.labels, \
                        is_training: False})
                val_writer.add_summary(summary, iteration)

                test_acc, summary = sess.run([accuracy, merged], \
                        feed_dict={x: dataset.test.images, y: dataset.test.labels, \
                        is_training: False})
                print("Test accuracy: %s" % test_acc)
                test_writer.add_summary(summary, iteration)

        # Report the test accuracy
        y_pred = tf.argmax(logits, 1)
        y_true = tf.argmax(y, 1)
        test_acc, y_pred, y_true = sess.run([accuracy, y_pred, y_true], \
                feed_dict={x: dataset.test.images, y: dataset.test.labels, \
                is_training: False})
        # print("Test accuracy: %s" % test_acc)

        # Plot the confusion matrix
        cnf_matrix = confusion_matrix(y_true, y_pred)
        plt.figure()
        plot_confusion_matrix(cnf_matrix, classes=classes,
                title="confusion matrix on test dataset", cmap=plt.cm.Oranges)
        plt.show()
