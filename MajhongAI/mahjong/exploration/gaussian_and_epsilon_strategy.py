import random
from mahjong.exploration_strategies.base import RawExplorationStrategy
import numpy as np


class GaussianAndEpsilonStrategy(RawExplorationStrategy):
    """
    With probability epsilon, take a completely random action.
    with probability 1-epsilon, add Gaussian noise to the action taken by a deterministic policy.
    """
    def __init__(self, model, action_space, epsilon, max_sigma=1.0, min_sigma=None, decay_period=1000000):
        assert len(action_space.shape) == 1
        if min_sigma is None:
            min_sigma = max_sigma
        self.max_sigma = max_sigma
        self.epsilon = epsilon
        self.min_sigma = min_sigma
        self.decay_period = decay_period
        self.action_space = action_space
        self.model = model

    def get_action_from_raw_action(self, state, t=None, **kwargs):
        if random.random() < self.epsilon:
            return self.action_space.sample()
        else:
            sigma = self.max_sigma - (self.max_sigma - self.min_sigma) * min(1.0, t * 1.0 / self.decay_period)
            action = self.model(state)
            return np.clip(
                action + np.random.normal(size=len(action)) * sigma,
                self.action_space.low,
                self.action_space.high,
                )
