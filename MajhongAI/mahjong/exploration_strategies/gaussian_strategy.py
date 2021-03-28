from mahjong.exploration_strategies.base import RawExplorationStrategy
import numpy as np


class GaussianStrategy(RawExplorationStrategy):
    """
    This strategy adds Gaussian noise to the action taken by the deterministic policy.
    """
    def __init__(self, model, action_space, max_sigma=1.0, min_sigma=None, decay_period=1000000):
        assert len(action_space.shape) == 1
        self.max_sigma = max_sigma
        if min_sigma is None:
            min_sigma = max_sigma
        self.min_sigma = min_sigma
        self.decay_period = decay_period
        self.action_space = action_space
        self.model = model

    def get_action_from_raw_action(self, state, t=None, **kwargs):
        sigma = (
            self.max_sigma - (self.max_sigma - self.min_sigma) *
            min(1.0, t * 1.0 / self.decay_period)
        )
        action = self.model(state)
        return np.clip(
            action + np.random.normal(size=len(action)) * sigma,
            self.action_space.low,
            self.action_space.high,
        )