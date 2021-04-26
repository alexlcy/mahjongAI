# -*- coding: utf-8 -*-
# @FileName : NoiseNetDQN.py
# @Project  : MAHJONG AI
# @Author   : WANG Jianxing
# @Time     : 2021/4/24 15:54

import os
import copy
import ast
import json
import glob
from functools import partial
from collections import Counter
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import random
import numpy as np
from collections import deque
from tqdm import tqdm
from mahjong.ReinforcementLearning.experience import ExperienceBuffer
import math


class SamePadConv2d(nn.Conv2d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.padding = (self.kernel_size[0] // 2, self.kernel_size[1] // 2)


conv3x1 = partial(SamePadConv2d, kernel_size=(3,1))


class ResidualBlock(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.layer1 = self.make_layer(in_channels)
        self.layer2 = self.make_layer(in_channels)

    def make_layer(self, in_channels, dropout_prob=0.5):
        layer = nn.Sequential(
            conv3x1(in_channels, in_channels),
            nn.BatchNorm2d(256),
            nn.Dropout2d(dropout_prob),
            nn.LeakyReLU()
        )
        return layer

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out += x
        return out


class MJResNet50(nn.Module):
    def __init__(self, history_len=4, n_cls=34, n_residuals=50):
        super().__init__()
        self.net = self.create_model(2+(history_len+1)*21, n_residuals, n_cls)

    def forward(self, x):
        return self.net(x)

    def create_model(self, in_channels, n_residuals, n_cls):
        # First layer
        module_list = nn.ModuleList([
            conv3x1(in_channels, 256),
            nn.BatchNorm2d(256),
            nn.Dropout2d(0.5),
            nn.LeakyReLU()
        ])
        # Adding residual blocks
        for layer_i in range(n_residuals):
            module_list.append(ResidualBlock(256))

        # Flatten & then fc layers
        module_list.append(nn.Flatten())
        out_feat = 1024
        module_list += nn.ModuleList([
            *self.linear_block(256*34, 1024, dropout_prob=0.2),
            *self.linear_block(1024, 256, dropout_prob=0.2),
            nn.Linear(256, n_cls)
        ])

        return nn.Sequential(*module_list)

    def linear_block(self, n_feat, out_feat, dropout_prob=0.5):
        block = nn.ModuleList([
            NoisyFactorizedLinear(n_feat, out_feat),
            nn.BatchNorm1d(out_feat),
            nn.Dropout(dropout_prob),
            nn.LeakyReLU()
        ])
        return block


class NoisyFactorizedLinear(nn.Linear):
    """
    NoisyNet layer with factorized gaussian noise
    nn.Linear already initializes weight and bias to mu_w and mu_b
    """
    def __init__(self, in_features, out_features, sigma_zero=0.5, bias=True):
        super(NoisyFactorizedLinear, self).__init__(in_features, out_features, bias=bias)
        sigma_init = sigma_zero / math.sqrt(in_features)
        self.sigma_w = nn.Parameter(torch.Tensor(out_features, in_features).fill_(sigma_init))
        self.register_buffer("epsilon_input", torch.zeros(1, in_features))
        self.register_buffer("epsilon_output", torch.zeros(out_features, 1))
        if bias:
            self.sigma_b = nn.Parameter(torch.Tensor(out_features).fill_(sigma_init))

    def forward(self, input):
        bias = self.bias
        func = lambda x: torch.sign(x) * torch.sqrt(torch.abs(x))

        with torch.no_grad():
            torch.randn(self.epsilon_input.size(), out=self.epsilon_input)
            torch.randn(self.epsilon_output.size(), out=self.epsilon_output)
            eps_in = func(self.epsilon_input)
            eps_out = func(self.epsilon_output)  # eps_b
            noise = torch.mul(eps_in, eps_out)  # eps_w
        if bias is not None:
            bias = bias + self.sigma_b * eps_out.t()  # bias = mu_b + sigma_b * eps_b
        return F.linear(input, self.weight + self.sigma_w * noise, bias)  # weight: mu_w


class DQN:
    def __init__(self):
        # trained model
        self.model = self.create_model()

        # target model
        self.target_model = self.create_model()
        self.target_model.set_weights(self.model.get_weights())

        self.replay_buffer = deque(maxlen=BUFFER_SIZE)

        # Used to count when to update target network with main network's weights
        self.target_update_counter = 0

    def preprocess(self, states, rewards, actions, cnt):  # (n,195,34,1),(n,),(n,1,34)
        for i in range(len(states)-1):
            round_cnt = cnt[i]
            # For a same round, set done = False
            if cnt[i+1] == round_cnt:
                self.replay_buffer.append((states[i], actions[i], rewards[i], states[i+1], False))
            # For a different round, set done = True
            else:
                self.replay_buffer.append((states[i], actions[i], rewards[i], states[i+1], True))
        # return math.ceil(len(self.replay_buffer)/BATCH_SIZE)
        return self.replay_buffer

    def create_model(self):
        return MJResNet50().to(device)

    def get_estimated_Q(self, state):
        return self.model.predict(np.array(state))

    def train(self, terminal_state):
        mini_batch = random.sample(self.replay_buffer, BATCH_SIZE)

        all_current_state = np.array([transition[0] for transition in mini_batch])
        all_current_Q = self.model.predict(all_current_state)  # TODO: whether can predict directly

        all_next_state = np.array([transition[3] for transition in mini_batch])
        all_next_Q = self.target_model.predict(all_next_state)

        X,y = [],[]
        for i, (current_state, action, reward, new_state, done) in enumerate(mini_batch):
            if not done:
                max_next_Q = np.max(all_next_Q[i])
                new_Q = reward + GAMMA*max_next_Q
            else:
                new_Q = reward

            # Update Q value for given state
            current_qs = all_current_Q[i]  # current_qs: (1,34)
            for idx,val in enumerate(action):
                if val == 1:
                    current_qs[idx] = new_Q
                break

            # append to training data
            X.append(current_state)
            y.append(current_qs)

        # Train as one batch
        self.model.fit(np.array(X), np.array(y), shuffle=False, batch_size=BATCH_SIZE)

        # Update target network counter every ju
        if terminal_state:
            self.target_update_counter += 1

        # If counter reaches set value, update target network with weights of main network
        if self.target_update_counter == UPDATE_FREQUENCY:
            self.target_model.set_weights(self.model.get_weights())
            self.target_update_counter = 0
            return self.target_model


LR = 0.00025
GAMMA = 0.99
BATCH_SIZE = 32
UPDATE_FREQUENCY = 3  # 每三局更新一次
BUFFER_SIZE = 10000
EPSILON = 1.0
# 每次从buffer中读一次数据就训练一次，一共100次。训练一次从buffer中随机取32batch训练
EPISODE = 100

DQN_agent = DQN()
FILE = 'experiment_2021_04_21_12_13_25.h5'

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using {} device".format(device))

for episode in tqdm(range(1, EPISODE+1)):
    new_buffer = DQN_agent.preprocess(ExperienceBuffer(10).read_experience(FILE))

    new_model = False
    step = 0

    # When new_model changes from False to True, meaning finished 3 局
    while not new_model:
        done = new_buffer[step][4]
        new_model = DQN_agent.train(done)
        step += 1

    # Update target model

# TODO: Use new Q function as the discard model to play in env
# TODO: Replace discard model with new target model in env every 3 rounds (How?)
# TODO: New epsilon-greedy should be added into env
# TODO: Initialize replay buffer as zero

