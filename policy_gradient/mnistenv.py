from gym import spaces
from tensorflow.examples.tutorials.mnist import input_data

class MNISTEnvironment:

    POS_REWARD = 1
    NEG_REWARD = -1

    def __init__(self):
        self.dataset = input_data.read_data_sets("MNIST_data/", one_hot=True)
        self.action_space = spaces.Discrete(10)
        self.observation_space = spaces.Box(low=0, high=255, shape=(784,))

    def step(self, label):
        # TODO we need to work with tensors I think
        reward = self.POS_REWARD if label == self.current_label else self.NEG_REWARD
        new_observation = self.dataset.train.next_batch(1)
        self.current_label = new_observation[1]
        return new_observation[0], reward

    def reset(self):
        observation = dataset.train.next_batch(1)
        self.current_label = observation[1]
        return observation[0]
