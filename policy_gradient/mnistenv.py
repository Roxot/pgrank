from gym import spaces
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data

class MNISTEnvironment:

    POS_REWARD = 1
    NEG_REWARD = -1

    def __init__(self):
        self.dataset = input_data.read_data_sets("MNIST_data/", one_hot=True)
        self.action_space = spaces.Discrete(10)
        self.observation_space = spaces.Box(low=0, high=255, shape=(784,))

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
