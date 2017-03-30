from tensorflow.examples.tutorials.mnist import input_data
from sklearn.metrics import confusion_matrix
from tensorflow.contrib.layers import initializers
from tensorflow.contrib.layers import regularizers

from mlp import MLP
from plot import plot_confusion_matrix

import itertools
import tensorflow as tf
import matplotlib.pyplot as plt

if __name__ == '__main__':

    # Dataset
    # dataset = input_data.read_data_sets("MNIST_data/", one_hot=True)
    classes = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]
    input_dim = 784
    num_classes = 10

    # Grid search
    learning_rates = [1e-1, 1e-2, 1e-3, 1e-4, 1e-5]
    batch_sizes = [64, 128, 256, 512, 1024]
    num_hiddens = [32, 64, 128, 256, 512]
    weight_reg_strengths = [1e-1, 1e-2, 1e-3, 1e-4, 1e-5]
    optimizers = ["rms", "sgd", "adam"]

    # Hyperparameters
    # num_hidden = []
    # learning_rate = 0.1
    max_steps = 500
    # batch_size = 256
    dropout_rate = 0.
    activation_fn = tf.nn.relu
    weight_initializer = initializers.xavier_initializer()
    # weight_reg_strength = 1e-3
    # weight_regularizer = regularizers.l2_regularizer(weight_reg_strength)
    # optimizer = tf.train.GradientDescentOptimizer(learning_rate)

    best_val_acc = 0.
    best_params = ()

    i = 0
    for learning_rate, batch_size, num_hidden, weight_reg_strength, opt_name in itertools.product(learning_rates, batch_sizes, num_hiddens, weight_reg_strengths, optimizers):
        i += 1

        dataset = input_data.read_data_sets("MNIST_data/", one_hot=True)
        optimizer = None
        if opt_name == "sgd":
            optimizer = tf.train.GradientDescentOptimizer(learning_rate)
        elif opt_name == "rms":
            optimizer = tf.train.RMSPropOptimizer(learning_rate)
        elif opt_name == "adam":
            optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
        weight_regularizer = regularizers.l2_regularizer(weight_reg_strength)

        # Setup graph
        is_training = tf.placeholder(tf.bool)
        x = tf.placeholder(tf.float32, [None, input_dim])
        y = tf.placeholder(tf.float32, [None, num_classes])
        mlp = MLP(n_classes=num_classes, is_training=is_training, \
                n_hidden=[num_hidden], dropout_rate=dropout_rate, \
                activation_fn=activation_fn, \
                weight_regularizer=weight_regularizer)
        logits = mlp.inference(x, i)
        loss = mlp.loss(logits, y)
        train_step = optimizer.minimize(loss)
        accuracy = mlp.accuracy(logits, y)
        merged = tf.merge_all_summaries()

        # Run the session
        with tf.Session() as sess:
            sess.run(tf.initialize_all_variables())
# Train for max_steps batches
            for iteration in range(max_steps):
                batch = dataset.train.next_batch(batch_size)
                _ = sess.run([train_step], \
                        feed_dict={x: batch[0], y: batch[1], is_training: True})

                if iteration % 100 == 0 or iteration == max_steps - 1:
                    train_acc, train_loss = sess.run([accuracy, loss], \
                            feed_dict={x: batch[0], y: batch[1], is_training: False})
                    print("Iteration %s/%s: Train Loss = %s, Train Accuracy = %s" % \
                            (iteration, max_steps, train_loss, train_acc))

            # Report the test accuracy
            # y_pred = tf.argmax(logits, 1)
            # y_true = tf.argmax(y, 1)
            # test_acc, y_pred, y_true = sess.run([accuracy, y_pred, y_true], \
            #         feed_dict={x: dataset.test.images, y: dataset.test.labels, \
            #         is_training: False})
            # print("Test accuracy: %s" % test_acc)

            # # Plot the confusion matrix
            # cnf_matrix = confusion_matrix(y_true, y_pred)
            # plt.figure()
            # plot_confusion_matrix(cnf_matrix, classes=classes,
            #         title="confusion matrix on test dataset", cmap=plt.cm.Oranges)
            # plt.show()

            val_images, val_labels = (dataset.validation.images, dataset.validation.labels)
            val_acc = sess.run([accuracy], feed_dict={x: val_images, y: val_labels, is_training: False})
            print("Validation Accuracy = %s for (%s, %s, %s, %s, %s)" % (val_acc, opt_name, learning_rate, batch_size, weight_reg_strength, num_hidden))
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_params = (opt_name, learning_rate, batch_size, weight_reg_strength, num_hidden)
            print("Best params so far: %s with val accuracy %s in order (opt_name, learning_rate, batch_size, weight_reg_strength, num_hidden)" % (best_params, best_val_acc))

    print("Best params: %s with val accuracy %s in order (opt_name, learning_rate, batch_size, weight_reg_strength, num_hidden)" % (best_params, best_val_acc))
