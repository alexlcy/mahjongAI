# -*- coding: utf-8 -*-
# @FileName : epsilon_method.py
# @Project  : MAHJONG AI
# @Author   : WANG Jianxing
# @Time     : 2021/4/18 14:19
import random
import numpy as np

from mahjong.exploration_strategies.base import RawExplorationStrategy


class EpsilonGreedy(RawExplorationStrategy):
    """
    Take a random discrete action with some probability.
    """
    def __init__(self, model, action_space, prob_random_action=0.1):
        self.decay_step = 0
        self.model = model
        self.epsilon = 1.0  # exploration probability at start
        self.epsilon_min = 0.01  # minimum exploration probability
        self.epsilon_decay = 0.0005  # exponential decay rate for exploration prob
        self.prob_random_action = prob_random_action
        self.action_space = action_space

    def select_action1(self, state, **kwargs):  # fixed probability for choosing a random action
        if random.random() <= self.prob_random_action:
            return self.action_space.sample()
            # return random.randrange(self.action_space)
        return self.model(state)

    def select_action2(self, state, **kwargs):  # probability with decay
        if self.epsilon > self.epsilon_min:
            self.epsilon *= (1 - self.epsilon_decay)
        explore_probability = self.epsilon
        self.decay_step += 1
        if explore_probability > np.random.rand():
            return random.randrange(self.action_space)
        else:
            return self.model(state)

    def select_action3(self, state, **kwargs):  # advanced probability with decay
        explore_probability = self.epsilon_min + (self.epsilon - self.epsilon_min) * np.exp(-self.epsilon_decay * self.decay_step)
        self.decay_step += 1
        if explore_probability > np.random.rand():
            return self.action_space.sample()
        else:
            return self.model(state)