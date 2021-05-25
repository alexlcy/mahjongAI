# -*- coding: utf-8 -*-
import math
import torch
from torch import nn
from torch.nn import init, Parameter
from torch.nn import functional as F
from torch.autograd import Variable
from collections import deque
import random
import numpy as np

from mahjong.snapshot import Snapshot
from mahjong.consts import COMMAND, CARD_DICT,CARD
from collections import Counter
from mahjong.models.model import DiscardModel, KongModel, PongModel
from mahjong.settings import FeatureTracer
from mahjong.exploration.methods import ExplorationMethods


class ReinforceLearningAgent:
    def __init__(self, player_id: int, play_times):
        self.play_times = play_times
        self.name = 'reinforcementlearning'
        self.__player_id = player_id
        self.state_tensor = np.zeros((9, 8, 10))
        # w_dict = {'W' + str(i + 1): i for i in range(9)}  # 万
        # b_dict = {'B' + str(i + 1): i + 9 for i in range(9)}  # 饼
        # t_dict = {'T' + str(i + 1): i + 18 for i in range(9)}  # 条
        # f_dict = {'F' + str(i + 1): i + 27 for i in range(4)}  # 风 东南西北
        # j_dict = {'J' + str(i + 1): i + 31 for i in range(3)}  # （剑牌）中发白
        # total_dict = {**w_dict, **b_dict, **t_dict, **f_dict, **j_dict}
        # self.total_dict_revert = {index: value for value, index in total_dict.items()}
        self.discard_model = DiscardModel()
        self.kong_model = KongModel()
        self.pong_model = PongModel()
        self.exploration_method = ExplorationMethods(self.discard_model)
        # self.decay_step = 0
        # self.epsilon = 0.1  # exploration probability at start
        # self.epsilon_min = 0.01  # minimum exploration probability
        # self.epsilon_decay = 0.0005  # exponential decay rate for exploration prob

    def decide_kong(self, feature):
        # Decide whether to make a Kong
        pred = self.kong_model.predict(feature)
        softmax = torch.nn.Softmax(dim=1)
        softmax_pred = softmax(pred)
        index = np.argmax(softmax_pred.numpy())
        if index == 0:
            whether_kong = False
        else:
            whether_kong = True
        score = softmax_pred.numpy()[0][index]
        # print(f'Kong: {whether_kong}, score: {softmax_pre}, index: {index}')
        return whether_kong, score

    def decide_pong(self, feature):
        # Decide whether to make a Pong
        pred = self.pong_model.predict(feature)
        softmax = torch.nn.Softmax(dim=1)
        softmax_pred = softmax(pred)
        index = np.argmax(softmax_pred.numpy())
        if index == 0:
            whether_pong = False
        else:
            whether_pong = True
        score = softmax_pred.numpy()[0][index]
        # print(f'Pong: {whether_pong}, score: {softmax_pre}, index: {index}')
        return whether_pong, score

    def decide_win(self):
        # Later will implement decide win model
        return True

    # TODO: Debug & optimize some repeat part - Koning
    def decide(self, snapshot: Snapshot, feature_tracer: FeatureTracer, trace: list, deck: list):
        """
        Decide which action to take based on the legal action and data available

        Args:
            snapshot ():
            trace ():
            deck ():

        Returns:
        """
        player = snapshot.players[self.__player_id]
        legal_actions = player['legal_actions']
        feature = feature_tracer.get_features(player['player_id'])

        # Exception handling
        if not legal_actions or len(legal_actions) == 0:
            return

        # This optimization may cause bug in new experience buffer
        # So we comment this and let it do the following procedure
        # if len(legal_actions) == 1:
        #     player['choice'] = legal_actions[0]
        #     return

        # For discard predict, no matter what action it takes
        discard_tile, is_trigger_by_rl, discard_probabilities = self.decide_discard(player, feature, player['hands'],
                                                                                    feature_tracer)
        feature_tracer.set_current_prediction(self.__player_id, discard_probabilities)
        feature_tracer.set_is_trigger_by_rl(self.__player_id, is_trigger_by_rl)

        # Step 1: 选缺 process (Only trigger once in a round)
        colors = [0] * 3
        for card in player['hands']:
            colors[math.floor(card / 10)] += 1

        if legal_actions[0] >= 600:
            player['choice'] = COMMAND.COLOR.value + colors.index(min(colors))
            return

        # Step 2: Choose which Action
        player['choice'] = max(legal_actions)
        pos = -1

        # Choose whether win
        if player['choice'] >= 500:
            whether_win = self.decide_win()
            if whether_win is True:
                return
            else:
                pos -= 1
                player['choice'] = legal_actions[pos]
                # TODO: If no bug, will remove below print
                # print(f"1 Checking: new choice - {player['choice']} should be >=100 & < 500~~")

        # Choose whether zhi kong
        if player['choice'] >= 400:
            # check whether pong
            # retrieve confidence score of pong and kong
            whether_pong, pong_score = self.decide_pong(feature)
            whether_kong, kong_score = self.decide_kong(feature)
            if all([whether_pong, whether_kong]):
                if pong_score > kong_score:
                    player['choice'] = legal_actions[pos - 1]
                    # TODO: If no bug, will remove below print
                    # print(f"2 Checking: new choice - {player['choice']} should be >=100 & < 200~~")
                    return
                else:
                    return
            elif whether_kong is True and whether_pong is False:
                return
            elif whether_kong is False and whether_pong is True:
                player['choice'] = legal_actions[pos - 1]
                # TODO: If no bug, will remove below print
                # print(f"3 Checking: new choice - {player['choice']} should be >=100 & < 200~~")
                return
            elif whether_kong is False and whether_pong is False:
                player['choice'] = legal_actions[pos - 2]
                # TODO: If no bug, will remove below print
                # print(f"4 Checking: new choice - {player['choice']} should be <= 0~~")
                return

        # Choose whether bu kong
        if player['choice'] >= 300:
            # check whether kong
            whether_kong, _ = self.decide_kong(feature)
            if whether_kong is True:
                return
            else:
                player['choice'] = legal_actions[pos - 1]
                # TODO: If no bug, will remove below print
                # If not bu kong, we need to discard a card
                # print(f"5 Checking: new choice - {player['choice']} should be < 100~~")
                return

        # Choose whether an kong
        if player['choice'] >= 200:
            # check whether kong
            whether_kong, _ = self.decide_kong(feature)
            if whether_kong is True:
                return
            else:
                player['choice'] = legal_actions[pos - 1]
                # TODO: If no bug, will remove below print
                # If not an kong, we need to discard a card
                # print(f"6 Checking: new choice - {player['choice']} should be < 100~~")
                return

        # Choose whether pong
        if player['choice'] >= 100:
            # check whether pong
            whether_pong, _ = self.decide_pong(feature)
            if whether_pong is True:
                return
            else:
                player['choice'] = legal_actions[pos - 1]
                # TODO: If no bug, will remove below print
                # print(f"7 Checking: new choice - {player['choice']} should be <= -1~~")
                return

        # Step 3: Choose which one to discard
        if player['choice'] < 100:
            # Done at the beginning
            # discard_tile, is_trigger_by_rl, discard_probabilities = self.decide_discard(player, feature, player['hands'], feature_tracer)
            #
            # # Save current prediction
            # if discard_probabilities is None:
            #     print('Checking nan pro??!!!')
            # feature_tracer.set_current_prediction(self.__player_id, discard_probabilities)
            # # Save the status that is_trigger_by_rl
            # if is_trigger_by_rl is None:
            #     print(f'Checking why??')
            # feature_tracer.set_is_trigger_by_rl(self.__player_id, is_trigger_by_rl)
            # # TODO: Checking, can delete later
            # if is_trigger_by_rl is None:
            #     print(f'Check unexpected error: player_id:{self.__player_id}, is_trigger_by_rl is {is_trigger_by_rl}')

            # Call discard function to discard a tile
            if discard_tile is not None:
                player['choice'] = discard_tile
                return

        player['choice'] = random.choice(legal_actions)

    def decide_discard(self, player, feature, hands, feature_tracer):
        """
        The tile is discarded based on the below sequence
        1. Discard color tile if there exist
        2. Discard based on model suggestion
        3. Discard based on naive rules

        Returns:
            discard_tile: The tile want to discard
        """
        # return self.exploration_method.epsilon_1(feature, player)  # total 10: 1,1,3,2
        # return self.exploration_method.epsilon_2(feature, player)  # total 10: 1,3,2,3
        # return self.exploration_method.epsilon_3(feature, player, feature_tracer)  # total 10: 3,3,2,2

        # TODO: Try different explore methods and see whether need change
        # return self.exploration_method.epsilon_second_of_softmax(feature, player, feature_tracer)  # total 10: 3,3,2,2


        # return self.exploration_method.epsilon_by_softmax(feature, player, feature_tracer)  # William train on this

        # return self.exploration_method.epsilon_random(feature, player, feature_tracer)  # Alex train on this

        return self.exploration_method.epsilon_rule(feature, player, feature_tracer, self.play_times)  # Koning train on this

        # Priority 1: Discard based on color
        # color_discard_tile = self.decide_discard_by_color(player)
        # if color_discard_tile is not None:
        #     return color_discard_tile

        # Using eplison greedy to decide whether to discard by rule or use model prediction
        # self.calculate_epsilon_decay()
        # if np.random.random() > self.epsilon:
        #     softmax_pred, ai_discard_tile_list = self.decide_discard_by_AI(feature)
        #     for index, ai_discard_tile in enumerate(ai_discard_tile_list):
        #         if ai_discard_tile in player['hands']:
        #             return ai_discard_tile
        # else:
        #     # Priority 3: Discard based on naive rule
        #     return self.decide_discard_by_rule(player)



    # def calculate_epsilon_decay(self):
    #     self.epsilon = self.epsilon * (1 - self.epsilon_decay) if self.epsilon > self.epsilon_min else self.epsilon

    # def decide_discard_by_AI(self, feature):
    #     """
    #     Call the discard model and return the tile that we shoudl discard
    #     Returns:
    #     """
    #
    #     pred = self.discard_model.predict(feature)
    #     softmax = torch.nn.Softmax(dim=1)
    #     softmax_pred = softmax(pred)
    #     tile_priority = np.argsort(softmax_pred.numpy())[0][::-1]
    #     tile_priority_list = [self.total_dict_revert[index] for index in tile_priority]
    #     tile_index_priority = [CARD_DICT[index] for index in tile_priority_list if index[0] not in ('J', 'F')]
    #
    #     return softmax_pred, tile_index_priority

    def decide_discard_by_color(self, player):
        """
        If there exist a tile that belongs to the color (缺), the player should
        discard it before it discard other tile.

        Args:
            player ():

        Returns:

        """
        # Discard based on color (缺)
        color = player['color']
        tile_counter = Counter(player['hands'])

        # Discard if the there are only one 缺 tile
        for tile_num in range(1, 10):
            if tile_counter[color + tile_num] == 1:
                return color + tile_num

        # Discard if the there are two 缺 tile
        for tile_num in range(1, 10):
            if tile_counter[color + tile_num] == 2:
                return color + tile_num

        # Discard if the there are only three tile
        for tile_num in range(1, 10):
            if tile_counter[color + tile_num] == 3:
                return color + tile_num

        # If no 缺 tile available, then skip the 缺 discard function
        return None

    # def decide_discard_by_rule(self, player):
    #     """
    #     When there are no valid decision made by AI and no discard based on color.
    #     The discard will be conducted using the same naive rule as the rule agent.
    #
    #     Args:
    #         player ():
    #
    #     Returns:
    #
    #     """
    #     # Discard based on rule
    #     cards = [0] * 30
    #     for card in player['hands']:
    #         cards[card] += 1
    #
    #     for card in range(30):
    #         if cards[card] == 1:
    #             return card
    #     for card in range(30):
    #         if cards[card] == 2:
    #             return card













# Code from William (Archive)
# # https://github.com/niravnb/Reinforcement-learning/blob/master/Bandits/Code/epsilonGreedy_Softmax_UCB1.py
# # From https://pylessons.com/Epsilon-Greedy-DQN/
# class ReinforceLearningAgent:
#     def __init__(self, model, explore_method='epsilon_greedy'):
#         self.explore_method = explore_method
#         self.decay_step = 0
#         self.model = model
#         self.epsilon = 1.0  # exploration probability at start
#         self.epsilon_min = 0.01  # minimum exploration probability
#         self.epsilon_decay = 0.0005  # exponential decay rate for exploration prob
#
#     def select_action(self, state, action_space):
#         if self.explore_method == 'epsilon_greedy':
#             # old version
#             # if self.epsilon > self.epsilon_min:
#             #     self.epsilon *= (1 - self.epsilon_decay)
#             # explore_probability = self.epsilon
#             explore_probability = self.epsilon_min + (self.epsilon - self.epsilon_min) * np.exp(-self.epsilon_decay * self.decay_step)
#             self.decay_step += 1
#             if explore_probability > np.random.rand():
#                 # Make a random action (exploration)
#                 return random.randrange(action_space)
#             else:
#                 return self.model(state)
#         # elif self.explore_method == 'noisy_net':
#
#
# # Noisy linear layer with independent Gaussian noise
# class NoisyLinear(nn.Linear):
#     def __init__(self, in_features, out_features, sigma_init=0.017, bias=True):
#         super(NoisyLinear, self).__init__(in_features, out_features, bias=True)  # TODO: Adapt for no bias
#         # µ^w and µ^b reuse self.weight and self.bias
#         self.sigma_init = sigma_init
#         self.sigma_weight = Parameter(torch.Tensor(out_features, in_features))  # σ^w
#         self.sigma_bias = Parameter(torch.Tensor(out_features))  # σ^b
#         self.register_buffer('epsilon_weight', torch.zeros(out_features, in_features))
#         self.register_buffer('epsilon_bias', torch.zeros(out_features))
#
#     def reset_parameters(self):
#         if hasattr(self, 'sigma_weight'):  # Only init after all params added (otherwise super().__init__() fails)
#           init.uniform(self.weight, -math.sqrt(3 / self.in_features), math.sqrt(3 / self.in_features))
#           init.uniform(self.bias, -math.sqrt(3 / self.in_features), math.sqrt(3 / self.in_features))
#           init.constant(self.sigma_weight, self.sigma_init)
#           init.constant(self.sigma_bias, self.sigma_init)
#
#     def forward(self, input):
#         return F.linear(input, self.weight + self.sigma_weight * Variable(self.epsilon_weight), self.bias + self.sigma_bias * Variable(self.epsilon_bias))
#
#     def sample_noise(self):
#         self.epsilon_weight = torch.randn(self.out_features, self.in_features)
#         self.epsilon_bias = torch.randn(self.out_features)
#
#     def remove_noise(self):
#         self.epsilon_weight = torch.zeros(self.out_features, self.in_features)
#         self.epsilon_bias = torch.zeros(self.out_features)