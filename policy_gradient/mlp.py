from sklearn.metrics import confusion_matrix
from tensorflow.contrib.layers import initializers
from tensorflow.contrib.layers import regularizers

import tensorflow as tf

class MLP:

    def __init__(self, n_hidden = [100], n_classes = 10, is_training = tf.constant(True),
                 activation_fn = tf.nn.relu, dropout_rate = 0.0,
                 weight_initializer = initializers.xavier_initializer(),
                 weight_regularizer = regularizers.l2_regularizer(0.001)):
        self.n_hidden = n_hidden
        self.n_classes = n_classes
        self.is_training = is_training
        self.activation_fn = activation_fn
        self.dropout_rate = dropout_rate
        self.weight_initializer = weight_initializer
        self.weight_regularizer = weight_regularizer

    def inference(self, x, i):
        inp = x
        for layer_num, hidden_dim in enumerate(self.n_hidden):
          layer_name = "hidden_%d_%d" % (layer_num, i)
          with tf.name_scope(layer_name):
            pre_activations = self._layer(inp, hidden_dim, layer_name)
            activations = self.activation_fn(pre_activations)
            tf.histogram_summary(layer_name + "/activations", activations)
            inp = activations

          with tf.name_scope("dropout_%d_%d" % (layer_num, i)):
            inp = tf.cond(self.is_training, lambda: tf.nn.dropout(inp, 1. - self.dropout_rate), lambda: inp)

        with tf.name_scope("final_linear"):
          logits = self._layer(inp, self.n_classes, "final_linear_%d" % i)

        return logits

    def _layer(self, x, output_dim, layer_name):
        input_dim = x.get_shape()[1]
        with tf.variable_scope(layer_name):
          W = tf.get_variable("weights", shape=([input_dim, output_dim]), \
                              initializer=self.weight_initializer,
                              regularizer=self.weight_regularizer)
          tf.histogram_summary(layer_name + "/weights", W)
          b = tf.get_variable("biases", shape=([output_dim]), \
                              initializer=tf.constant_initializer(0.))
          tf.histogram_summary(layer_name + "/biases", b)
        pre_activations = tf.matmul(x, W) + b
        tf.histogram_summary(layer_name + "/pre_activations", pre_activations)
        return pre_activations

    def loss(self, logits, labels):
        with tf.name_scope("cross_entropy_loss"):
          cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits, labels), name="cross-entropy")
          reg_loss = tf.reduce_sum(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))
          loss = cross_entropy + reg_loss
          tf.scalar_summary("cross_entropy", cross_entropy)
          tf.scalar_summary("reg_loss", reg_loss)
          tf.scalar_summary("total_loss", loss)
        return loss

    def accuracy(self, logits, labels):
        with tf.name_scope("accuracy"):
          correct_preds = tf.equal(tf.argmax(logits, 1, name="predictions"), \
              tf.argmax(labels, 1, name="true_labels"), name="correct_preds")
          accuracy = tf.reduce_mean(tf.cast(correct_preds, tf.float32), name="accuracy")
          tf.scalar_summary("accuracy", accuracy)

        return accuracy
