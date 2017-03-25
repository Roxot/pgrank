from gym import spaces
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data

class MNISTEnvironment:

    POS_REWARD = 1
    NEG_REWARD = -1

    def __init__(self, batch_size):
        self.dataset = input_data.read_data_sets("MNIST_data/", one_hot=False)
        self.batch_size = batch_size

    def _observe(self):
        return self.dataset.train.next_batch(self.batch_size)

    def _reward(self, predicted_labels):
        rewards = np.full(self.current_labels.shape, self.NEG_REWARD, dtype=np.float)
        rewards[self.current_labels == predicted_labels] = self.POS_REWARD
        return rewards

    def step(self, labels):
        rewards = self._reward(labels)
        new_observation, self.current_labels = self._observe()
        return new_observation, rewards

    def reset(self):
        observation, self.current_labels = self._observe()
        return observation

    def test_set(self):
        return self.dataset.test.images, self.dataset.test.labels
