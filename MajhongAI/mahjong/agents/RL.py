# -*- coding: utf-8 -*-
import math
import torch
from torch import nn
from torch.nn import init, Parameter
from torch.nn import functional as F
from torch.autograd import Variable
import random
from collections import deque
import numpy as np


# https://github.com/niravnb/Reinforcement-learning/blob/master/Bandits/Code/epsilonGreedy_Softmax_UCB1.py
# From https://pylessons.com/Epsilon-Greedy-DQN/
class ReinforceLearningAgent:
    def __init__(self, model, explore_method='epsilon_greedy'):
        self.explore_method = explore_method
        self.decay_step = 0
        self.model = model
        self.epsilon = 1.0  # exploration probability at start
        self.epsilon_min = 0.01  # minimum exploration probability
        self.epsilon_decay = 0.0005  # exponential decay rate for exploration prob

    def select_action(self, state, action_space):
        if self.explore_method == 'epsilon_greedy':
            # old version
            # if self.epsilon > self.epsilon_min:
            #     self.epsilon *= (1 - self.epsilon_decay)
            # explore_probability = self.epsilon
            explore_probability = self.epsilon_min + (self.epsilon - self.epsilon_min) * np.exp(-self.epsilon_decay * self.decay_step)
            self.decay_step += 1
            if explore_probability > np.random.rand():
                # Make a random action (exploration)
                return random.randrange(action_space)
            else:
                return self.model(state)
        # elif self.explore_method == 'noisy_net':


# Noisy linear layer with independent Gaussian noise
class NoisyLinear(nn.Linear):
    def __init__(self, in_features, out_features, sigma_init=0.017, bias=True):
        super(NoisyLinear, self).__init__(in_features, out_features, bias=True)  # TODO: Adapt for no bias
        # µ^w and µ^b reuse self.weight and self.bias
        self.sigma_init = sigma_init
        self.sigma_weight = Parameter(torch.Tensor(out_features, in_features))  # σ^w
        self.sigma_bias = Parameter(torch.Tensor(out_features))  # σ^b
        self.register_buffer('epsilon_weight', torch.zeros(out_features, in_features))
        self.register_buffer('epsilon_bias', torch.zeros(out_features))

    def reset_parameters(self):
        if hasattr(self, 'sigma_weight'):  # Only init after all params added (otherwise super().__init__() fails)
          init.uniform(self.weight, -math.sqrt(3 / self.in_features), math.sqrt(3 / self.in_features))
          init.uniform(self.bias, -math.sqrt(3 / self.in_features), math.sqrt(3 / self.in_features))
          init.constant(self.sigma_weight, self.sigma_init)
          init.constant(self.sigma_bias, self.sigma_init)

    def forward(self, input):
        return F.linear(input, self.weight + self.sigma_weight * Variable(self.epsilon_weight), self.bias + self.sigma_bias * Variable(self.epsilon_bias))

    def sample_noise(self):
        self.epsilon_weight = torch.randn(self.out_features, self.in_features)
        self.epsilon_bias = torch.randn(self.out_features)

    def remove_noise(self):
        self.epsilon_weight = torch.zeros(self.out_features, self.in_features)
        self.epsilon_bias = torch.zeros(self.out_features)