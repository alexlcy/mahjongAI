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
import copy
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
        self.epsilon_decay = 0.01  # exponential decay rate for exploration prob
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

    def new_epsilon(self, step, play_times):
        A = 0.5
        B = 0.1
        C = 0.3
        EPISODES = 15 * play_times
        standardized_time = (step - A * EPISODES) / (B * EPISODES)
        cosh = np.cosh(math.exp(-standardized_time))
        epsilon = 1 - (1 / cosh + (step * C / EPISODES))
        return epsilon


    def epsilon_random(self, feature, player, feature_tracer, **kwargs):  # fixed probability for choosing a random action
        explore_probability = self.epsilon_min + (self.epsilon - self.epsilon_min) * np.exp(-self.epsilon_decay * self.decay_step)
        self.decay_step += 1

        # # For a fixed epsilon
        # explore_probability = 0.1
        feature_tracer.set_explore_probability(player['player_id'], explore_probability)
        ai_discard_tile, discard_probabilities = self.decide_discard_by_AI(feature, player)
        # TODO: if no unexpected error, can delete below print
        if discard_probabilities is None:
            print('Error here~ type2: methods/epsilon_by_softmax')
        if explore_probability > np.random.rand():
            return self.decide_discard_by_random(player), False, discard_probabilities
        else:
            return ai_discard_tile, True, discard_probabilities


    def epsilon_rule(self, feature, player, feature_tracer, play_times, **kwargs):  # fixed probability for choosing a random action
        # explore_probability = self.epsilon_min + (self.epsilon - self.epsilon_min) * np.exp(-self.epsilon_decay * self.decay_step)
        explore_probability = self.new_epsilon(self.decay_step, play_times)
        self.decay_step += 1

        # # For a fixed epsilon
        # explore_probability = 0.1
        feature_tracer.set_explore_probability(player['player_id'], explore_probability)
        ai_discard_tile, discard_probabilities = self.decide_discard_by_AI(feature, player)
        # TODO: if no unexpected error, can delete below print
        if discard_probabilities is None:
            print('Error here~ type2: methods/epsilon_by_softmax')
        if explore_probability > np.random.rand():
            return self.decide_discard_by_rule(player), False, discard_probabilities
        else:
            return ai_discard_tile, True, discard_probabilities


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
        # # minimize the epsilon, (Koning) just for more attempt with the sl model
        # explore_probability = explore_probability / 10
        feature_tracer.set_explore_probability(player['player_id'], explore_probability)
        self.decay_step += 1
        ai_discard_tile, discard_probabilities = self.decide_discard_by_AI(feature, player)
        # TODO: if no unexpected error, can delete below print
        if discard_probabilities is None:
            print('Error here~ type2: methods/epsilon_3')
        if explore_probability > np.random.rand():
            return self.decide_discard_by_rule(player), False, discard_probabilities
        else:
            return ai_discard_tile, True, discard_probabilities

    def epsilon_second_of_softmax(self, feature, player, feature_tracer, **kwargs):  # Explore by our model with second max prob
        explore_probability = self.epsilon_min + (self.epsilon - self.epsilon_min) * np.exp(-self.epsilon_decay * self.decay_step)
        self.decay_step += 1

        # For a fixed epsilon
        explore_probability = 0.1

        feature_tracer.set_explore_probability(player['player_id'], explore_probability)
        ai_discard_tile, discard_probabilities = self.decide_discard_by_AI(feature, player)
        # TODO: if no unexpected error, can delete below print
        if discard_probabilities is None:
            print('Error here~ type2: methods/epsilon_second_of_softmax')
        if explore_probability > np.random.rand():
            return self.decide_card_with_second_possibility(feature, player), False, discard_probabilities
        else:
            return ai_discard_tile, True, discard_probabilities

    def epsilon_by_softmax(self, feature, player, feature_tracer, **kwargs):  # Explore by our model with softmax
        # explore_probability = self.epsilon_min + (self.epsilon - self.epsilon_min) * np.exp(-self.epsilon_decay * self.decay_step)
        # self.decay_step += 1

        # For a fixed epsilon
        explore_probability = 0.1
        feature_tracer.set_explore_probability(player['player_id'], explore_probability)
        ai_discard_tile, discard_probabilities = self.decide_discard_by_AI(feature, player)
        # TODO: if no unexpected error, can delete below print
        if discard_probabilities is None:
            print('Error here~ type2: methods/epsilon_by_softmax')
        if explore_probability > np.random.rand():
            return self.decide_discard_by_softmax_and_epsilon(feature, player), False, discard_probabilities
        else:
            return ai_discard_tile, True, discard_probabilities

    def decide_discard_by_AI_help(self, feature):
        raw_prediction = self.model.predict(feature)  # (1,34)
        softmax = torch.nn.Softmax(dim=1)
        softmax_prediction = softmax(raw_prediction)
        tile_priority = np.argsort(softmax_prediction.numpy())[0][::-1]
        tile_priority_list = [self.total_dict_revert[index] for index in tile_priority]
        tile_index_priority = [CARD_DICT[index] for index in tile_priority_list if index[0] not in ('J', 'F')]
        return softmax_prediction, tile_index_priority, softmax_prediction

    def decide_discard_by_AI(self, feature, player):
        """
        Call the discard model and return the tile that we should discard
        Returns:
        """
        softmax_prediction, ai_discard_tile_list, softmax_prediction = self.decide_discard_by_AI_help(feature)
        for index, ai_discard_tile in enumerate(ai_discard_tile_list):
            if ai_discard_tile in player['hands']:
                return ai_discard_tile, softmax_prediction

    def decide_discard_by_rule(self, player):
        """
        When there are no valid decision made by AI and no discard based on color.
        The discard will be conducted using the same naive rule as the rule agent.

        Args:
            player ():

        Returns:

        """
        # Discard based on rule
        cards = [0] * 30
        for card in player['hands']:
            cards[card] += 1

        for card in range(30):
            if cards[card] == 1:
                return card
        for card in range(30):
            if cards[card] == 2:
                return card

    # Choose the card in hands has the second max probability
    def decide_card_with_second_possibility(self, feature, player):
        softmax_prediction, ai_discard_tile_list, softmax_prediction = self.decide_discard_by_AI_help(feature)

        # Delete the original target card in order to not choose it anymore in exploration
        find = False
        for index, ai_discard_tile in enumerate(ai_discard_tile_list):
            if ai_discard_tile in player['hands'] and not find:
                find = True
            else:
                if ai_discard_tile in player['hands'] and find:
                    return ai_discard_tile

    def decide_discard_by_softmax_and_epsilon(self, feature, player):
        softmax_prediction, ai_discard_tile_list, raw_prediction = self.decide_discard_by_AI_help(feature)
        raw_prediction_27 = copy.deepcopy(raw_prediction[0][:27])
        softmax = torch.nn.Softmax(dim=0)
        softmax_prediction_27 = softmax(raw_prediction_27).numpy()

        # Set the original target card's possibility is 0 in order to not choose it anymore in exploration
        for index, ai_discard_tile in enumerate(ai_discard_tile_list):
            if ai_discard_tile in player['hands']:
                softmax_prediction_27[index] = 0
                break
        softmax_prediction_27 = softmax_prediction_27 / np.sum(softmax_prediction_27)
        size = sum(1 for item in softmax_prediction_27 if item != 0)
        try:
            sample_list = np.random.choice(ai_discard_tile_list, size=size, replace=False, p=softmax_prediction_27)
        except:
            with open('bug_data3', 'w') as f:
                f.write('softmax_prediction_27: ' + str(softmax_prediction_27) + '/n')
                f.write('raw_prediction: ' + str(raw_prediction))
        for ai_discard_tile in sample_list:
            if ai_discard_tile in player['hands']:
                return ai_discard_tile

        # softmax_prediction, ai_discard_tile_list, discard_probabilities = self.decide_discard_by_AI_help(feature)
        # discard_probabilities_27 = copy.deepcopy(discard_probabilities[0][:27])
        # softmax = torch.nn.Softmax(dim=0)
        # softmax_prediction_27 = softmax(discard_probabilities_27).numpy()
        # softmax_prediction_27 = softmax_prediction_27 / np.sum(softmax_prediction_27)
        # size = sum(1 for item in softmax_prediction_27 if item)
        # try:
        #     sample_list = np.random.choice(ai_discard_tile_list, size=size, replace=False, p=softmax_prediction_27)
        # except:
        #     with open('bug_data3','w') as f:
        #         f.write('softmax_prediction_27: ' + str(softmax_prediction_27) + '/n')
        #         f.write('discard_probabilities: ' + str(discard_probabilities))
        # for ai_discard_tile in sample_list:
        #     if ai_discard_tile in player['hands']:
        #         return ai_discard_tile

        # softmax_prediction, ai_discard_tile_list, raw_prediction = self.decide_discard_by_AI_help(feature)
        # # Set the original target card's possibility is 0 in order to not choose it anymore in exploration
        # temp_weight_list = copy.deepcopy(softmax_prediction)[0][:27].tolist()
        # for index, ai_discard_tile in enumerate(ai_discard_tile_list):
        #     if ai_discard_tile in player['hands']:
        #         temp_weight_list[index] = 0
        #         break
        # while True:
        #     target_card = random.choices(ai_discard_tile_list, weights=temp_weight_list)[0]
        #     if target_card in player['hands']:
        #         return target_card

    def decide_discard_by_random(self, player):
        return random.choice(player['hands'])




