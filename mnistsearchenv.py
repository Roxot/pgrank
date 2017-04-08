from gym import spaces
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data

class MNISTSearchEnvironment:

    POS_REWARD = 1
    NEG_REWARD = -1

    def __init__(self, k, random_query=False):
        self.dataset = input_data.read_data_sets("MNIST_data/", one_hot=False)
        self.k = k
        self.random_query = random_query

    def _observe(self):

        # Get k images from the batch.
        batch = self.dataset.train.next_batch(self.k)

        # Pick one of the labels randomly.
        if self.random_query:
            # Completely random query.
            query = np.random.randint(10)
        else:
            # Query as random label of one of the images.
            query = batch[1][np.random.randint(self.k)]
        labels = np.where(batch[1] == query)[0]

        # Return ([image1, ..., imagek], random_label) and the correct image.
        observation = (batch[0], query)
        return observation, labels

    def _reward(self, predicted_label):
        # return self.POS_REWARD if predicted_label in self.current_labels else self.NEG_REWARD
        data = np.zeros(self.k)
        data[self.current_labels] = 1.
        idcg = self._get_idcg(data, self.k)
        ranking = data[::-1] if predicted_label == 1 else data
        return 0 if idcg == 0 else self._ndcg_at_k(ranking, self.k, idcg)

    def step(self, label):
        reward = self._reward(label)
        new_observation, self.current_labels = self._observe()
        return new_observation, reward

    def reset(self):
        observation, self.current_labels = self._observe()
        return observation

    def test_set(self):
        return self.dataset.test.images, self.dataset.test.labels

    def validation_set(self):
        return self.dataset.validation.images, self.dataset.validation.labels

    # Calculates the NDCG@k at rank k, given the ideal DCG which
    # can be obtained using get_idcg(data, k).
    def _ndcg_at_k(self, ranking, k, idcg):
        return self._dcg_at_k(ranking, k) / idcg 

    # Calculate the ideal DCG@k given all data.
    def _get_idcg(self, data, k):
        ideal_ranking = sorted(data, reverse=True)
        return self._dcg_at_k(ideal_ranking, k)

    def _dcg_at_k(self, ranking, k):
        cg = np.power(2, ranking[:k]) - 1
        discount = np.log2(np.arange(k) + 2)
        return np.sum(np.divide(cg, discount))

