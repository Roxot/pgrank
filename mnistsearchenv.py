from gym import spaces
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data

class MNISTSearchEnvironment:

    POS_REWARD = 1
    NEG_REWARD = -1

    def __init__(self, k):
        self.dataset = input_data.read_data_sets("MNIST_data/", one_hot=False)
        self.k = k

    def _observe(self):

        # Get k images from the batch.
        batch = self.dataset.train.next_batch(self.k)

        # Pick one of the labels randomly.
        label = np.random.randint(self.k)
        query = batch[1][label]

        # Return ([image1, ..., imagek], random_label) and the correct image.
        observation = (batch[0], query)
        return observation, label

    def _reward(self, predicted_label):
        return self.POS_REWARD if predicted_label == self.current_label else self.NEG_REWARD

    def step(self, label):
        reward = self._reward(label)
        new_observation, self.current_label = self._observe()
        return new_observation, reward

    def reset(self):
        observation, self.current_label = self._observe()
        return observation

    def test_set(self):
        return self.dataset.test.images, self.dataset.test.labels

    def validation_set(self):
        return self.dataset.validation.images, self.dataset.validation.labels
