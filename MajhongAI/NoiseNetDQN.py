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
from mahjong.ReinforcementLearning.experience import ReplayBuffer
import math
import logging
import time

from mahjong.env import Env
from mahjong.ReinforcementLearning.experience import ReplayBuffer
from mahjong.models.model import DiscardModel, KongModel, PongModel
from mahjong.stats_logger.calc_functions import calc_win_rates, calc_hu_scores, calc_win_times, calc_hu_score_each_game, calc_mean_loss_each_train
from mahjong.agents.DL import DeepLearningAgent
from mahjong.agents.RL import ReinforceLearningAgent
from mahjong.agents.rule import RuleAgent
from mahjong.DQN import DQNModel


# class SamePadConv2d(nn.Conv2d):
#     def __init__(self, *args, **kwargs):
#         super().__init__(*args, **kwargs)
#         self.padding = (self.kernel_size[0] // 2, self.kernel_size[1] // 2)
#
#
# conv3x1 = partial(SamePadConv2d, kernel_size=(3,1))
#
#
# class ResidualBlock(nn.Module):
#     def __init__(self, in_channels):
#         super().__init__()
#         self.layer1 = self.make_layer(in_channels)
#         self.layer2 = self.make_layer(in_channels)
#
#     def make_layer(self, in_channels, dropout_prob=0.5):
#         layer = nn.Sequential(
#             conv3x1(in_channels, in_channels),
#             nn.BatchNorm2d(256),
#             nn.Dropout2d(dropout_prob),
#             nn.LeakyReLU()
#         )
#         return layer
#
#     def forward(self, x):
#         out = self.layer1(x)
#         out = self.layer2(out)
#         out += x
#         return out
#
#
# class MJResNet50(nn.Module):
#     def __init__(self, history_len=4, n_cls=34, n_residuals=50):
#         super().__init__()
#         self.net = self.create_model(2+(history_len+1)*21, n_residuals, n_cls)
#
#     def forward(self, x):
#         return self.net(x)
#
#     def create_model(self, in_channels, n_residuals, n_cls):
#         # First layer
#         module_list = nn.ModuleList([
#             conv3x1(in_channels, 256),
#             nn.BatchNorm2d(256),
#             nn.Dropout2d(0.5),
#             nn.LeakyReLU()
#         ])
#         # Adding residual blocks
#         for layer_i in range(n_residuals):
#             module_list.append(ResidualBlock(256))
#
#         # Flatten & then fc layers
#         module_list.append(nn.Flatten())
#         out_feat = 1024
#         module_list += nn.ModuleList([
#             *self.linear_block(256*34, 1024, dropout_prob=0.2),
#             *self.linear_block(1024, 256, dropout_prob=0.2),
#             nn.Linear(256, n_cls)
#         ])
#
#         return nn.Sequential(*module_list)
#
#     def linear_block(self, n_feat, out_feat, dropout_prob=0.5):
#         block = nn.ModuleList([
#             nn.Linear(n_feat, out_feat),
#             nn.BatchNorm1d(out_feat),
#             nn.Dropout(dropout_prob),
#             nn.LeakyReLU()
#         ])
#         return block
#
#
# class NoisyFactorizedLinear(nn.Linear):
#     """
#     NoisyNet layer with factorized gaussian noise
#     nn.Linear already initializes weight and bias to mu_w and mu_b
#     """
#     def __init__(self, in_features, out_features, sigma_zero=0.5, bias=True):
#         super(NoisyFactorizedLinear, self).__init__(in_features, out_features, bias=bias)
#         sigma_init = sigma_zero / math.sqrt(in_features)
#         self.sigma_w = nn.Parameter(torch.Tensor(out_features, in_features).fill_(sigma_init))
#         self.register_buffer("epsilon_input", torch.zeros(1, in_features))
#         self.register_buffer("epsilon_output", torch.zeros(out_features, 1))
#         if bias:
#             self.sigma_b = nn.Parameter(torch.Tensor(out_features).fill_(sigma_init))
#
#     def forward(self, input):
#         bias = self.bias
#         func = lambda x: torch.sign(x) * torch.sqrt(torch.abs(x))
#
#         with torch.no_grad():
#             torch.randn(self.epsilon_input.size(), out=self.epsilon_input)
#             torch.randn(self.epsilon_output.size(), out=self.epsilon_output)
#             eps_in = func(self.epsilon_input)
#             eps_out = func(self.epsilon_output)  # eps_b
#             noise = torch.mul(eps_in, eps_out)  # eps_w
#         if bias is not None:
#             bias = bias + self.sigma_b * eps_out.t()  # bias = mu_b + sigma_b * eps_b
#         return F.linear(input, self.weight + self.sigma_w * noise, bias)  # weight: mu_w
#
#
# class BaseModel:
#     def __init__(self, weight_path=None, device=None):
#         if device is None:
#             self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
#         else:
#             self.device = device
#         self._load_models(weight_path)
#
#     def _load_models(self,weight_path):
#         self.model = MJResNet50()
#         if weight_path is not None:
#             self.model.load_state_dict(torch.load(weight_path, map_location=torch.device('cpu')))
#             print(f'Loaded checkpoint {weight_path.split("/")[-1]}')
#
#         self.model.to(self.device)
#
#     def predict(self, inp):
#         '''
#         Run forward propagation for all models
#
#         Args:
#         - inp (torch.float32): array of size [bs, (hist+1)*39, 34, 1]
#
#         Returns:
#         - preds (torch.float32): array of size [bs, n_cls]
#         '''
#         self.model.eval()
#         with torch.no_grad():
#             preds = self.model(inp)
#         return preds.cpu()
#
#
# class DQNModel(BaseModel):
#     def __init__(self, device=None):
#         super().__init__(device)


class DQNAgent:
    def __init__(self):
        self.DQN = DQNModel().model
        self.DQN_target = DQNModel().model
        self.DQN_target.load_state_dict(self.DQN.state_dict())

        self.optimizer = torch.optim.Adam(self.DQN.parameters(), lr=LR)
        self.loss_fn = nn.MSELoss()

    # def act(self, state):
    #     raw_prediction = self.DQN.predict(state)  # (1,34)
    #     softmax = torch.nn.Softmax(dim=1)
    #     softmax_prediction = softmax(raw_prediction)
    #     tile_priority = np.argsort(softmax_prediction.numpy())[0][::-1]
    #     tile_priority_list = [self.total_dict_revert[index] for index in tile_priority]
    #     tile_index_priority = [CARD_DICT[index] for index in tile_priority_list if index[0] not in ('J', 'F')]

    def preprocess(self, exps):
        game_no, states, rewards, actions = exps['game_no'], exps['states'], exps['rewards'], exps['actions']
        next_states = copy.deepcopy(states)
        dones = np.zeros(len(states))
        dones[-1] = 1
        for i in range(len(states)-1):
            next_states[i] = states[i+1]
        next_states[-1] = states[-1]
        return states, actions, rewards, next_states, dones

    def train(self, exps):
        losses = []
        for exp_i, exp in enumerate(exps):
            states, actions, rewards, next_states, dones = self.preprocess(exp)
            for i in tqdm(range(math.ceil(len(states) / BATCH_SIZE)), desc=f"Training on buffer {exp_i+1}: "):
                batch_s, batch_a, batch_r, batch_next_s, batch_d = states[i*BATCH_SIZE:(i+1)*BATCH_SIZE], \
                                                                   actions[i*BATCH_SIZE:(i+1)*BATCH_SIZE], \
                                                                   rewards[i*BATCH_SIZE:(i+1)*BATCH_SIZE], \
                                                                   next_states[i*BATCH_SIZE:(i+1)*BATCH_SIZE], \
                                                                   dones[i*BATCH_SIZE:(i+1)*BATCH_SIZE]
                # s_j, a_j, r_j, s_j+1
                batch_s = torch.tensor(batch_s, dtype=torch.float32, device=device)
                batch_a = torch.tensor(batch_a).long().to(device)
                batch_r = (batch_r - batch_r.mean()) / (batch_r.std() + 1e-7)
                batch_r = torch.tensor(batch_r, dtype=torch.float32, device=device)
                batch_next_s = torch.tensor(batch_next_s, dtype=torch.float32, device=device)
                batch_d = torch.tensor(batch_d, dtype=torch.bool, device=device)

                # get q-values for all actions in current states
                predicted_qvalues = self.DQN(batch_s)

                # select q-values for chosen actions: Q(s_j,a_j;theta)
                predicted_qvalues_for_actions = torch.gather(predicted_qvalues, 1, batch_a.unsqueeze(-1))

                # compute q-values for all actions in next states: Q_hat(s_j+1,a';theta')
                predicted_next_qvalues = self.DQN_target(batch_next_s)

                # compute max using predicted next q-values:
                max_next_qvalues = predicted_next_qvalues.max(-1)[0]

                # compute "target q-values" for loss y_j
                target_qvalues_for_actions = batch_r + GAMMA * max_next_qvalues

                # at the last state we shall use simplified formula: Q(s,a) = r(s,a) since s' doesn't exist
                target_qvalues_for_actions = torch.where(batch_d, batch_r, target_qvalues_for_actions)

                # loss = F.smooth_l1_loss(predicted_qvalues_for_actions.squeeze(), target_qvalues_for_actions.detach())
                loss = self.loss_fn(predicted_qvalues_for_actions.squeeze(), target_qvalues_for_actions)

                self.optimizer.zero_grad()
                # loss = loss.mean()
                loss.backward()
                for param in self.DQN.parameters():
                    param.grad.data.clamp_(-1, 1)
                self.optimizer.step()
                # print(f"loss:{loss}")
                losses.append(loss)
        return np.mean(losses)

    def update_tar_DQN(self):
        self.DQN_target.load_state_dict(self.DQN.state_dict())
        print('\n=============== Target DQN model is updated ===============\n')


# =========================== Training ===========================
# Hyper-parameters & settings
PLAY_TIMES = 100000
LR = 0.00001
BATCH_SIZE = 512
EXP_SAMPLE_SIZE = 100  # how many games to sample to train model each time
BEHAVIOR_POLICY_UPDATE_INTV = 100  # interval after which the behavior policy gets replaced by the newest target policy
SAVE_INTV = 1000
TRAIN_FREQUENCY = 100
GAMMA = 0.99
# MODEL_TO_TRAIN = 'discard'
device = 'cuda' if torch.cuda.is_available() else 'cpu'

DQNAgent = DQNAgent()

LOG_FORMAT = "%(message)s "
logging.basicConfig(level=logging.DEBUG, format=LOG_FORMAT)

start = time.time()
buffer = ReplayBuffer(PLAY_TIMES)
random.seed(0)

config = {
    'show_log': False,
    'player_num': 4,
    'seed': None  # to None for random run, if seed == None, will not save record
}
env = Env(config)
RL_agent = ReinforceLearningAgent(0)
env.set_agents([RL_agent, RuleAgent(1), RuleAgent(2), RuleAgent(3)])

hu_reward_statistics = {0: [], 1: [], 2: [], 3: []}

prev_weights = None

# model_init_name = f'RL-discard-NoisyDQN-init.pth'
# RL_SAVE_DIR = 'mahjong/models/weights/discard/'
# torch.save(model.state_dict(), os.path.join(RL_SAVE_DIR, model_init_name))

for i in range(PLAY_TIMES):
    print(f'No.{i + 1} Game ing~')

    """
    reset & run
    """
    env.reset()
    env.run(buffer)

    # tensor board
    # win_times
    calc_win_times(buffer.win_times, buffer.game_no)
    # win_rates
    calc_win_rates(buffer.win_times, buffer.game_no)
    # hu_score
    calc_hu_scores(buffer.hu_score, buffer.game_no)
    # hu_score each game
    calc_hu_score_each_game(buffer.hu_reward, buffer.game_no)

    # # TODO: checking, can delete
    # reward_sum = np.sum([replay_buffer.hu_reward[b_key] for b_key in hu_reward_statistics.keys()])
    # for h_key in hu_reward_statistics.keys():
    #     hu_reward_statistics[h_key].append(replay_buffer.hu_reward[h_key])

    # # TODO: checking the reward sum is zero, can delete
    # if reward_sum != 0:
    #     print(f'Cal buffer: {replay_buffer.hu_reward}, sum: {reward_sum}')

    buffer.update_buffer()

    # Update policy
    if i < 200 or len(buffer) < EXP_SAMPLE_SIZE:
        continue

    if i != 0 and i % TRAIN_FREQUENCY == 0:
        exps = buffer.sample(EXP_SAMPLE_SIZE)
        mean_loss = DQNAgent.train(exps)
        # mean loss each train
        calc_mean_loss_each_train(buffer.mean_loss, buffer.game_no)

    # Replace behavior policy
    if i != 0 and i % BEHAVIOR_POLICY_UPDATE_INTV == 0:
        DQNAgent.update_tar_DQN()

    # Save policy
    if i != 0 and i % SAVE_INTV == 0:
        RL_SAVE_DIR = 'mahjong/models/weights/discard/'
        if not os.path.exists(RL_SAVE_DIR):
            os.makedirs(RL_SAVE_DIR)

        model_name = f'DQN-RL-discard-playtime_{i}.pth'
        torch.save(DQNAgent.DQN.state_dict(), os.path.join(RL_SAVE_DIR, model_name))

end = time.time()
print(f'Recording: {(end - start) / 60} min played {PLAY_TIMES} games')
