# -*- coding: utf-8 -*-
# @FileName : settings.py
# @Project  : MAHJONG AI
# @Author   : WANG Jianxing
# @Time     : 2021/4/17
from collections import deque
import copy
from mahjong.Serialization import online_serialize
import torch


class FeatureTracer:

    def __init__(self, player_initial_hands: dict):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        self.discard = {i: [] for i in range(4)}
        self.steal = None
        self.tiles = player_initial_hands
        self.open_meld = {i: [] for i in range(4)}
        self.own_wind = {i: [] for i in range(4)}
        self.round_wind = {i: [] for i in range(4)}
        self.q_dict = {i: deque([[], [], [], [], []], maxlen=5) for i in range(4)}
        # TODO: debug
        # self.current_prediction = {i: None for i in range(4)}
        # self.epsilons = {i: None for i in range(4)}
        self.current_prediction = {i: None for i in range(4)}
        self.epsilons = {i: None for i in range(4)}
        self.is_trigger_by_rl = {i: None for i in range(4)}

    def set_current_prediction(self, player, prediction):
        self.current_prediction[player] = prediction

    def set_explore_probability(self, player, explore_probability):
        self.epsilons[player] = explore_probability

    def set_is_trigger_by_rl(self, player, is_trigger_by_rl):
        self.is_trigger_by_rl[player] = is_trigger_by_rl

    def update(self, line):
        line = str(line)
        cur = line.replace('\t', ' ').replace(':', ' ').split()  # ['玩家', '0', '行为', 'DRAW', 'B7', '奖励', '0']
        player = int(cur[1])
        new_act = cur[3]
        card = cur[4]

        if new_act == 'PLAY':
            self.tiles[player].remove(card)
            self.discard[player].append(card)
        elif new_act == 'DRAW':
            self.tiles[player].append(card)
        elif new_act == 'PENG':
            self.open_meld[player].extend([card] * 3)  # [card]*3 : ['B7', 'B7', 'B7']
            for time in range(2):
                self.tiles[player].remove(card)
        elif new_act == 'BU':
            self.tiles[player].remove(card)
            for value in self.open_meld.values():
                if card in value:
                    value.append(card)
        elif new_act == 'ZHI':
            self.open_meld[player].extend([card] * 4)  # [card]*3 : ['B7', 'B7', 'B7', 'B7']
            for time in range(3):
                self.tiles[player].remove(card)
        elif new_act == 'GANG':
            self.open_meld[player].extend([card] * 4)
            for time in range(4):
                self.tiles[player].remove(card)
        elif new_act == 'HU':
            self.open_meld[player].append(card)

        # update steal
        if new_act in ['PLAY', 'DRAW', 'PENG']:
            self.steal = [card]

        # Updating this player's all status after making this action, then extract features
        def get_features(player):
            features = copy.deepcopy([self.own_wind[player],
                                      self.round_wind[player],
                                      self.tiles[player],
                                      self.steal,
                                      self.discard[player],
                                      self.discard[player + 1 if player + 1 <= 3 else (player + 1) % 4],
                                      self.discard[player + 2 if player + 2 <= 3 else (player + 2) % 4],
                                      self.discard[player + 3 if player + 3 <= 3 else (player + 3) % 4],
                                      self.open_meld[player],
                                      self.open_meld[player + 1 if player + 1 <= 3 else (player + 1) % 4],
                                      self.open_meld[player + 2 if player + 2 <= 3 else (player + 2) % 4],
                                      self.open_meld[player + 3 if player + 3 <= 3 else (player + 3) % 4]
                                      ])
            return features

        self.q_dict[player].appendleft(get_features(player))
        self.q_dict[player + 1 if player + 1 <= 3 else (player + 1) % 4].appendleft(
            get_features(player + 1 if player + 1 <= 3 else (player + 1) % 4))
        self.q_dict[player + 2 if player + 2 <= 3 else (player + 2) % 4].appendleft(
            get_features(player + 2 if player + 2 <= 3 else (player + 2) % 4))
        self.q_dict[player + 3 if player + 3 <= 3 else (player + 3) % 4].appendleft(
            get_features(player + 3 if player + 3 <= 3 else (player + 3) % 4))

        # # For debug:
        # with open('test_file1', 'a') as file:
        #     for key,val in self.q_dict.items():
        #         file.write(str(key)+str(val))
        #         file.write('\n')
        #     file.write('\n')
        #     file.write('\n')

    def get_features(self, player):
        return torch.Tensor(online_serialize(self.q_dict[player])).unsqueeze(0).to(self.device)
