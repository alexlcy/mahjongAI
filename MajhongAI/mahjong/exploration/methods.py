# -*- coding: utf-8 -*-
# @FileName : methods.py
# @Project  : MAHJONG AI
# @Author   : WANG Jianxing
# @Time     : 2021/4/18 14:19
import random
import numpy as np
import torch
import math
from mahjong.consts import COMMAND, CARD_DICT,CARD
# from mahjong.agents.RL import ReinforceLearningAgent
# from mahjong.exploration.base import RawExplorationStrategy


class ExplorationMethods:
    """
    Take a random discrete action with some probability.
    """
    def __init__(self, model, prob_random_action=0.1, max_sigma=1.0, min_sigma=0, decay_period=200):
        # super().__init__(self,player_id)
        self.decay_step = 0
        self.model = model
        self.epsilon = 1.0  # exploration probability at start
        self.epsilon_min = 0.01  # minimum exploration probability
        self.epsilon_decay = 0.0005  # exponential decay rate for exploration prob
        self.prob_random_action = prob_random_action

        self.t = 0
        self.max_sigma = max_sigma
        if min_sigma is None:
            min_sigma = max_sigma
        self.min_sigma = min_sigma
        self.decay_period = decay_period

        w_dict = {'W' + str(i + 1): i for i in range(9)}  # 万
        b_dict = {'B' + str(i + 1): i + 9 for i in range(9)}  # 饼
        t_dict = {'T' + str(i + 1): i + 18 for i in range(9)}  # 条
        f_dict = {'F' + str(i + 1): i + 27 for i in range(4)}  # 风 东南西北
        j_dict = {'J' + str(i + 1): i + 31 for i in range(3)}  # （剑牌）中发白
        total_dict = {**w_dict, **b_dict, **t_dict, **f_dict, **j_dict}
        self.total_dict_revert = {index: value for value, index in total_dict.items()}

    def epsilon_1(self, feature, player, **kwargs):  # fixed probability for choosing a random action
        if random.random() <= self.prob_random_action:
            # return super().decide_discard_by_rule()
            return self.decide_discard_by_rule(player)
            # return random.randrange(self.action_space)
        else:
            return self.decide_discard_by_AI(feature, player)

    def epsilon_2(self, feature, player, **kwargs):  # probability with decay
        if self.epsilon > self.epsilon_min:
            self.epsilon *= (1 - self.epsilon_decay)
        explore_probability = self.epsilon
        self.decay_step += 1
        if explore_probability > np.random.rand():
            return self.decide_discard_by_rule(player)
        else:
            return self.decide_discard_by_AI(feature, player)

    def epsilon_3(self, feature, player, feature_tracer, **kwargs):  # advanced probability with decay
        explore_probability = self.epsilon_min + (self.epsilon - self.epsilon_min) * np.exp(-self.epsilon_decay * self.decay_step)
        feature_tracer.set_explore_probability(player['player_id'], explore_probability)
        self.decay_step += 1
        ai_discard_tile, raw_prediction = self.decide_discard_by_AI(feature, player)
        if raw_prediction is None:
            print('Error here~ type2: methods/epsilon_3')
        if explore_probability > np.random.rand():
            return self.decide_discard_by_rule(player), False, raw_prediction
        else:
            return ai_discard_tile, True, raw_prediction

    def gaussian(self, feature, player, action_space, **kwargs):
        self.t += 1
        sigma = (
            self.max_sigma - (self.max_sigma - self.min_sigma) *
            min(1.0, self.t * 1.0 / self.decay_period)
        )
        action = self.decide_discard_by_AI(feature, player)
        return np.clip(
            math.ceil(action + np.random.normal() * sigma),
            np.array(action_space).min(),
            np.array(action_space).max(),
        )

    def decide_discard_by_AI_help(self, feature):
        raw_prediction = self.model.predict(feature)  # (1,34)
        softmax = torch.nn.Softmax(dim=1)
        softmax_prediction = softmax(raw_prediction)
        tile_priority = np.argsort(softmax_prediction.numpy())[0][::-1]
        tile_priority_list = [self.total_dict_revert[index] for index in tile_priority]
        tile_index_priority = [CARD_DICT[index] for index in tile_priority_list if index[0] not in ('J', 'F')]
        return softmax_prediction, tile_index_priority, raw_prediction

    def decide_discard_by_AI(self, feature, player):
        """
        Call the discard model and return the tile that we should discard
        Returns:
        """
        softmax_prediction, ai_discard_tile_list, raw_prediction = self.decide_discard_by_AI_help(feature)
        for index, ai_discard_tile in enumerate(ai_discard_tile_list):
            if ai_discard_tile in player['hands']:
                return ai_discard_tile, raw_prediction

    def decide_discard_by_rule(self, player):
        """
        When there are no valid decision made by AI and no discard based on color.
        The discard will be conducted using the same naive rule as the rule agent.

        Args:
            player ():

        Returns:

        """
        # # Discard based on rule
        # cards = [0] * 30
        # for card in player['hands']:
        #     cards[card] += 1
        #
        # for card in range(30):
        #     if cards[card] == 1:
        #         return card
        # for card in range(30):
        #     if cards[card] == 2:
        #         return card

        return random.choice(player['hands'])
