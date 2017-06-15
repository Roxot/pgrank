from tensorflow.contrib.layers import initializers
from tensorflow.contrib.layers import regularizers

import tensorflow as tf

class PGRank:

    def __init__(self, input_dim, output_dim, h_dim, reg_str=0.):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.h_dim = h_dim
        self.reg_str = reg_str

        self._create_placeholders()
        self._build_model()

    def _create_placeholders(self):
        self.x = tf.placeholder(tf.float32, shape=[None, None, self.input_dim])  # batch_size x k x input_dim
        self.q = tf.placeholder(tf.int32, shape=[None, 1])                  # batch_size x 1
        self.r = tf.placeholder(tf.float32, shape=[None, 1])                # batch_size x 1
        self.serp = tf.placeholder(tf.int32, shape=[None, None])            # batch_size x k

    def _build_model(self):

        batch_size = tf.shape(self.x)[0]
        k = tf.shape(self.x)[1]

        nn_input = tf.reshape(self.x, (-1, self.input_dim)) # k * batch_size x input_dim

        with tf.name_scope("policy_network"):

            # First layer: (batch_size * k x input_dim) -> (batch_size * k x h_dim)
            h = tf.contrib.layers.fully_connected(nn_input, self.h_dim, activation_fn=tf.nn.relu, \
                weights_initializer=initializers.xavier_initializer(), \
                weights_regularizer=regularizers.l2_regularizer(self.reg_str), \
                biases_initializer=tf.constant_initializer(0.), \
                biases_regularizer=regularizers.l2_regularizer(self.reg_str))

            # Second layer: (batch_size * k x h_dim) -> (batch_size * k x output_dim)
            logits = tf.contrib.layers.fully_connected(h, self.output_dim, activation_fn=tf.nn.relu, \
                weights_initializer=initializers.xavier_initializer(), \
                weights_regularizer=regularizers.l2_regularizer(self.reg_str), \
                biases_initializer=tf.constant_initializer(0.), \
                biases_regularizer=regularizers.l2_regularizer(self.reg_str))

            # Select only the output for the current queries.
            query = tf.one_hot(self.q, self.output_dim, axis=-1)            # batch_size x 1 x output_dim
            logits = tf.reshape(logits, (batch_size, k, self.output_dim))   # batch_size x k x output_dim
            doc_scores = tf.reduce_sum(tf.mul(logits, query), 2)            # batch_size x k
            self.policy = tf.nn.softmax(doc_scores * 1000)                         # batch_size x k

