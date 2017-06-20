from helper import softmax

from tensorflow.contrib.layers import initializers
from tensorflow.contrib.layers import regularizers

import tensorflow as tf
import numpy as np

class PGRank:

    def __init__(self, input_dim, output_dim, h_dim, reg_str=0.):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.h_dim = h_dim
        self.reg_str = reg_str

        self._create_placeholders()
        self._build_model()

    def _create_placeholders(self):
        self.x = tf.placeholder(tf.float32, shape=[None, None, self.input_dim]) # batch_size x k x input_dim
        self.q = tf.placeholder(tf.int32, shape=[None, 1])                      # batch_size x 1
        self.reward = tf.placeholder(tf.float32, shape=[None, 1])               # batch_size x 1
        self.serp = tf.placeholder(tf.int32, shape=[None, None])                # batch_size x k
        self.deriv_weights = tf.placeholder(tf.float32, shape=[None, None])     # batch_size x k
        self.true_labels = tf.placeholder(tf.int64, [None, None])               # num_data_sets x data_size

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
            policy = tf.nn.softmax(doc_scores)                              # batch_size x k

            # Calculate the surrogate loss using an input placeholder weights vector and a rewards vector.
            J = tf.mul(doc_scores, self.deriv_weights)
            J = tf.reduce_sum(J, reduction_indices=[1], keep_dims=True)
            J = tf.reduce_mean(tf.mul(J, self.reward))
            self.loss = -J

            # Some evaluation metrics on some left-out test or validation set.
            predictions = tf.argmax(logits, 2)
            accuracy = tf.reduce_mean(tf.cast(tf.equal(predictions, self.true_labels), tf.float32), \
                    reduction_indices=[1])
            self.det_ranking = tf.nn.top_k(doc_scores, k=k).indices

            # Make some tensors available for fetching.
            self.logits = logits
            self.doc_scores = doc_scores
            self.policy = policy
            self.accuracy = accuracy

    """
        derivative_weights(doc_scores, ranking):

        Calculates the weights for each network output we can use in a surrogate TensorFlow objective that
        yield the correct gradients. It uses two matrices P and M to calculate these for a batch of doc_scores
        and rankings.

        P is a matrix that looks as follows:
        P = [[P(R_1 = D_1) P(R_1 = D_2) P(R_1 = D_3)]
             [0.           P(R_2 = D_2) P(R_2 = D_3)]
             [0.0          0.0          1.0]]

        It contains the probabilities of all documents D_1, ..., D_k where D_i is the document ranked
        at the ith position, being ranked at some rank R_1, ..., R_k. E.g. P(R_1 = D_1) is the probability
        that the first document was ranked 1st, which did actually happen. And P(R_2 = D_3) is the
        probability that the document ranked third was actually ranked second.

        M is a matrix of 1s, 0s and -1s that is created for each ith ranked document so that the derivatives
        for the ith ranked document is calculated correctly by sum(M * P).

        Returns a (batch_size x k) matrix with the derivative weights. deriv_weights[b, i] represents the
        derivative for the ith output score document (not the ith ranked document).
    """
    def derivative_weights(self, doc_scores, ranking):

        # Calculate matrix P.
        batch_size, k = ranking.shape
        P = np.zeros((batch_size, k, k))
        doc_scores_copy = np.copy(doc_scores)
        for i in range(k):
            num_docs_left = k - i
            doc_scores_copy = np.reshape(doc_scores_copy, (batch_size, num_docs_left))

            # Calculate the probability of each document being chosen.
            P[:, i, i:k] = softmax(doc_scores_copy)

            # Remove the documents selected at this iteration from the scores array.
            actions = (ranking[:, i] - i) % num_docs_left
            delete_ids = actions + (np.arange(batch_size) * num_docs_left)
            doc_scores_copy = np.delete(doc_scores_copy, delete_ids, axis=None)

        # Calulcate the derivative weights.
        deriv_weights = np.zeros((batch_size, k))
        for i in range(k):

            # We calculate M for each ith ranked document.
            M = np.zeros((k, k))
            M[:i, i] = -1.
            M[i, :] = 1.
            M[i, i] = 0.
            M = np.tile(M, (batch_size, 1, 1))

            # Calculate the derivative weights for all documents ranked on the ith position as
            # the summed product M * P for each batch element.
            deriv_weights[:, i] = np.sum(M * P, (1, 2))

        # Reorder the derivative weights so that deriv_weights[b, i] represents the output for document with the ith
        # output score instead of the ith ranked document.
        for b in range(batch_size):
            deriv_weights[b, :] = deriv_weights[b, ranking[b, :]]

        return deriv_weights

    def train_on_batch(self, sess, train_step, docs, queries, env, explorer):

        # Calculate the document scores andthe policy.
        feed_dict = { self.x: docs, self.q: queries }
        doc_scores, policy = sess.run([self.doc_scores, self.policy], feed_dict=feed_dict)

        # Rank the documents using the policy and observe the reward.
        ranking = explorer.rank_docs(policy)
        reward = env.reward(ranking)
        batch_reward = np.average(reward)

        # Train on the batch.
        feed_dict[self.reward] = reward
        feed_dict[self.deriv_weights] = self.derivative_weights(doc_scores, ranking)
        _, loss = sess.run([train_step, self.loss], feed_dict=feed_dict)

        return batch_reward, loss

    # # TODO remove this function, should be in run_pg.py
    # # Evaluates NDCG and accuracy on a set of evaluation images. This function is reasonably
    # # specific to the MNIST dataset. Could be more efficient but does the job.
    # def evaluate(self, sess, evaluation_set):
    #     images = evaluation_set.images
    #     labels = evaluation_set.labels
    #     x = np.reshape(images, (1, images.shape[0], images.shape[1]))
    #     labels = np.reshape(labels, (1, len(labels)))

    #     ndcgs = np.zeros((1, self.output_dim))
    #     for query_id in range(self.output_dim):

    #         # Prepare the network input.
    #         queries = np.array([[query_id]])

    #         # Retrieve the document deterministic ranking for the query, sorted by document score.
    #         feed_dict = { self.x: x, self.q: queries }
    #         ranking = sess.run(self.det_ranking, feed_dict=feed_dict)

    #         # Calculate the ndcg for this list.
    #         rel_labels = np.zeros(labels.shape)
    #         rel_labels[np.where(labels == queries)] = 1.
    #         from search.reward import ndcg_full
    #         ndcgs[0, query_id] = ndcg_full(ranking, rel_labels)[0, 0]

    #     print(ndcgs)
    #     print(np.average(ndcgs))

    #     # Retrieve the logits for all possible queries.
    #     feed_dict = { self.x: x, self.true_labels: labels }
    #     acc = sess.run(self.accuracy, feed_dict=feed_dict)
    #     print(acc)