# -*- coding: utf-8 -*-
# @FileName : experience.py
# @Project  : MAHJONG AI
# @Author   : Koning LIU
# @Time     : 2021/3/22 17:12

import pandas as pd
from copy import deepcopy
import torch
import numpy as np
import h5py
import datetime
import random
from collections import Counter, deque

from mahjong.Serialization import helper
from mahjong.ReinforcementLearning.calculation_rl import cal_probability_of_action, cal_probability_of_action_2

__all__ = [
    'ExperienceCollector',
    'ExperienceBuffer'
]


class ExperienceCollector:
    def __init__(self, player_id, is_rl=False):
        self.action_nums = []
        self.player_ids = []
        self.raw_states = []
        self.states = []
        self.discards = []
        self.open_melds = []
        self.steals = []
        self.actions = []
        self.rewards = []
        self.scores = []
        self.lack_color = None
        self.features = []
        self.lack_colors = []
        self.feature_tracers = []
        self.discard_cards = []
        self.win = False
        self.win_times = 0
        self.player_id = player_id
        self.hu_rewards = 0
        self.norm_hu_rewards = 0
        self.norm_rewards = []
        self.last_hu_record_index = 0
        self.is_rl_agent = int(is_rl)

    # def record_feature_reward(self, q_dict):

    def record_decision(self, action_num, raw_state, state, discard, open_meld, steal, action, reward, score,
                        lack_color, feature_tracer, hu_rewards, norm_reward):
        if action[0] == 'HU':
            r = deepcopy(reward)
            self.hu_rewards += hu_rewards
            self.norm_hu_rewards = norm_reward
            for i in range(self.last_hu_record_index, len(self.rewards)):
                self.rewards[i] += r
                self.norm_rewards[i] = norm_reward
            self.last_hu_record_index = len(self.rewards)
            if r > 0 and action[2] == action[3]:
                self.win = True
                self.win_times += 1

        elif action[0] == 'PLAY':
            # TODO: Checking is_trigger_by_rl, can delete later, if no bug
            if self.is_rl_agent and feature_tracer.is_trigger_by_rl[self.player_id] is None:
                print(f'player_id: {self.player_id}, is_trigger_by_rl: {feature_tracer.is_trigger_by_rl}, checking !!!')
            self.action_nums.append(deepcopy(action_num))
            self.states.append(deepcopy(raw_state))
            self.raw_states.append(deepcopy(state))
            self.discards.append(deepcopy(discard))
            self.open_melds.append(deepcopy(open_meld))
            self.steals.append(deepcopy(steal))
            self.actions.append(deepcopy(action))
            self.rewards.append(deepcopy(reward))
            self.scores.append(deepcopy(score))
            self.lack_colors.append(deepcopy(lack_color))
            self.feature_tracers.append(deepcopy(feature_tracer))
            self.discard_cards.append(deepcopy(action[1]))
            self.norm_rewards.append(deepcopy(norm_reward))


class ExperienceBuffer:
    def __init__(self, play_times):
        keys = ['player_ids', 'lack_color', 'action_nums', 'raw_states', 'states', 'discards',
                'open_melds', 'steals', 'actions', 'rewards', 'scores']
        self.buffer = {key: [] for key in keys}
        self.game_no_list = []
        self.x = []
        self.y = []
        self.discard = []
        self.is_rl_agent = []
        self.is_trigger_by_rl = []
        self.action_probabilities = []
        self.discard_argmax = []
        self.discard_probabilities = []
        # self.epsilons = []
        self.win_times = {0: 0, 1: 0, 2: 0, 3: 0}
        self.hu_score = {0: 0, 1: 0, 2: 0, 3: 0}
        self.hu_reward = {0: 0, 1: 0, 2: 0, 3: 0}
        self.play_times = play_times
        self.game_no = 0
        self.win_game = 0

    # TODO: if ok should be moved to calculation_rl.py
    # def cal_probability_of_action(self, is_trigger_by_rl, epsilon, discard_argmax, discard_probabilities):
    #     p_action = 0
    #     try:
    #         p_action = (1 - epsilon) * discard_probabilities.T[discard_argmax][0]
    #     except Exception:
    #         print('Error here, type 1: experience/cal_probability_of_action')
    #     if not is_trigger_by_rl:
    #         try:
    #             p_action += epsilon / 27
    #         except Exception:
    #             print('Error here, type 4: experience/cal_probability_of_action')
    #     return p_action

    def massage_experience(self, collectors, normed=True):
        self.game_no += 1
        for c_key in collectors.keys():
            for i in range(len(collectors[c_key].feature_tracers)):
                self.game_no_list.append(self.game_no)
                self.x.append(collectors[c_key].feature_tracers[i].get_features(c_key).cpu())
                self.discard.append(helper(1, [collectors[c_key].discard_cards[i]]))
                is_rl_agent = collectors[c_key].is_rl_agent
                self.is_rl_agent.append(is_rl_agent)
                self.discard_argmax.append(np.argmax(self.discard[-1]))
                # ### Discard model predictions ###
                tmp = collectors[c_key].feature_tracers[i].current_prediction[c_key]
                is_trigger_by_rl = collectors[c_key].feature_tracers[i].is_trigger_by_rl[c_key]
                if is_rl_agent and is_trigger_by_rl is None:
                    is_trigger_by_rl = -2
                    print(f'Checking None: ?? in experience')
                self.is_trigger_by_rl.append(int(is_trigger_by_rl) if is_rl_agent else -1)
                self.discard_probabilities.append(tmp if tmp is not None else torch.Tensor(np.zeros((1, 34))))
                # ### probability from model (rule & AI)
                epsilon = collectors[c_key].feature_tracers[i].epsilons[c_key]
                if is_rl_agent:
                    p = cal_probability_of_action(self.is_trigger_by_rl[-1], epsilon, self.discard_argmax[-1], tmp)
                    self.action_probabilities.append(p)
                    # self.epsilons.append(epsilon if epsilon is not None else -0.5)
                else:
                    # print(f'Checking cases~ type 3')
                    self.action_probabilities.append(1)
                    # self.epsilons.append(-1)

            if normed:
                self.y.extend(collectors[c_key].norm_rewards)
            else:
                self.y.extend(collectors[c_key].rewards)
            self.hu_score[c_key] += collectors[c_key].hu_rewards
            self.hu_reward[c_key] = collectors[c_key].hu_rewards
            if collectors[c_key].win:
                self.win_times[c_key] += collectors[c_key].win_times
        for c_key in collectors.keys():
            if collectors[c_key].win:
                self.win_game += 1
                break

    def save_experience(self, folder_path):
        if len(self.x) != 0:
            game_no = np.array(self.game_no_list)
            x = torch.cat(self.x, dim=0)
            y = np.array(self.y)
            # discard = np.stack(self.discard)
            is_rl_agent = np.array(self.is_rl_agent)
            print(f'is_trigger_by_rl: {Counter(self.is_trigger_by_rl)}')
            # is_trigger_by_rl = np.array(self.is_trigger_by_rl)
            p_action = np.array(self.action_probabilities)
            discard_argmax = np.array(self.discard_argmax)
            # discard_probabilities = torch.cat(self.discard_probabilities, dim=0)
            # epsilon = np.array(self.epsilons)
            date_string = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
            with h5py.File(folder_path + "experiment_" + date_string + r'.h5', 'w') as experience_outf:
                experience_outf.create_group('experience')
                # 1
                experience_outf['experience'].create_dataset('game_no', data=game_no)
                # 2
                experience_outf['experience'].create_dataset('states', data=x)
                # 3
                experience_outf['experience'].create_dataset('rewards', data=y)
                # 4
                experience_outf['experience'].create_dataset('actions', data=discard_argmax)
                # 5
                experience_outf['experience'].create_dataset('is_rl_agents', data=is_rl_agent)
                # 6
                # experience_outf['experience'].create_dataset('is_trigger_by_rl', data=is_trigger_by_rl)
                # 7
                experience_outf['experience'].create_dataset('p_action', data=p_action)
                # 8
                # experience_outf['experience'].create_dataset('discard_argmax', data=discard_argmax)
                # 9
                # experience_outf['experience'].create_dataset('discard_probabilities', data=discard_probabilities)
                # 10
                # experience_outf['experience'].create_dataset('epsilon', data=epsilon)
            for c_key in self.win_times.keys():
                print(f'Player {c_key} won {self.win_times[c_key]} times...')
            print(f'HU {sum(self.win_times.values())} times data generated...')
        else:
            print('No HU experience data...')

    def read_experience(self, file_name):
        h5file = h5py.File(file_name, 'r')
        headers = ['game_no', 'is_rl_agents', 'states', 'rewards', 'actions', 'p_action']
        buffer_dict = {header: np.array(h5file['experience'][header]) for header in headers}
        return buffer_dict

    # # ## deprecation
    # def combine_experience(self, collectors):
    #     # self.feature_buffer = []
    #     for c_key in collectors.keys():
    #         if collectors[c_key].win:
    #             self.buffer['action_nums'].extend(collectors[c_key].action_nums)
    #             self.buffer['raw_states'].extend(collectors[c_key].raw_states)
    #             self.buffer['states'].extend(collectors[c_key].states)
    #             self.buffer['discards'].extend(collectors[c_key].discards)
    #             self.buffer['open_melds'].extend(collectors[c_key].open_melds)
    #             self.buffer['steals'].extend(collectors[c_key].steals)
    #             self.buffer['actions'].extend(collectors[c_key].actions)
    #             self.buffer['rewards'].extend(collectors[c_key].rewards)
    #             self.buffer['scores'].extend(collectors[c_key].scores)
    #             self.buffer['lack_color'].extend(collectors[c_key].lack_colors)
    #             self.buffer['player_ids'].extend([c_key] * len(collectors[c_key].action_nums))
    #
    # def store_experience(self, folder_path, csv_file_name):
    #     dataframe = pd.DataFrame(self.buffer)
    #     dataframe.to_csv(folder_path + '/' + csv_file_name + '.csv', index=False, sep='|')
    #
    # def load_experience(self, folder_path, csv_file_name):
    #     self.buffer = pd.read_csv(folder_path + '/' + csv_file_name + '.csv', sep='|')
    #     return self.buffer


class ReplayBuffer:
    def __init__(self, play_times, buffer_capacity=100):
        self.buffer = deque([], maxlen=buffer_capacity)
        self.game_no_list = []
        self.x = []
        self.y = []
        self.discard = []
        self.is_rl_agent = []
        self.is_trigger_by_rl = []
        self.action_probabilities = []
        self.discard_argmax = []
        self.discard_probabilities = []
        # self.epsilons = []
        self.win_times = {0: 0, 1: 0, 2: 0, 3: 0}
        self.hu_score = {0: 0, 1: 0, 2: 0, 3: 0}
        self.hu_reward = {0: 0, 1: 0, 2: 0, 3: 0}
        self.play_times = play_times
        self.game_no = 0
        self.win_game = 0

    def massage_experience(self, collectors, normed=True):
        self.game_no += 1
        self.game_no_list = []
        self.x = []
        self.y = []
        self.discard = []
        self.is_rl_agent = []
        self.is_trigger_by_rl = []
        self.action_probabilities = []
        self.discard_argmax = []
        self.discard_probabilities = []
        for c_key in collectors.keys():
            for i in range(len(collectors[c_key].feature_tracers)):
                self.game_no_list.append(self.game_no)
                self.x.append(collectors[c_key].feature_tracers[i].get_features(c_key).cpu())
                self.discard.append(helper(1, [collectors[c_key].discard_cards[i]]))
                is_rl_agent = collectors[c_key].is_rl_agent
                self.is_rl_agent.append(is_rl_agent)
                self.discard_argmax.append(np.argmax(self.discard[-1]))
                # ### Discard model predictions ###
                tmp = collectors[c_key].feature_tracers[i].current_prediction[c_key]
                is_trigger_by_rl = collectors[c_key].feature_tracers[i].is_trigger_by_rl[c_key]
                if is_rl_agent and is_trigger_by_rl is None:
                    is_trigger_by_rl = -2
                    print(f'Checking None: ?? in experience')
                self.is_trigger_by_rl.append(int(is_trigger_by_rl) if is_rl_agent else -1)
                self.discard_probabilities.append(tmp if tmp is not None else torch.Tensor(np.zeros((1, 34))))
                # ### probability from model (rule & AI)
                epsilon = collectors[c_key].feature_tracers[i].epsilons[c_key]
                if is_rl_agent:
                    # p = cal_probability_of_action(self.is_trigger_by_rl[-1], epsilon, self.discard_argmax[-1], tmp)
                    p = cal_probability_of_action_2(self.discard_argmax[-1], tmp)
                    self.action_probabilities.append(p)
                else:
                    self.action_probabilities.append(1)

            if normed:
                self.y.extend(collectors[c_key].norm_rewards)
            else:
                self.y.extend(collectors[c_key].rewards)
            self.hu_score[c_key] += collectors[c_key].hu_rewards
            self.hu_reward[c_key] = collectors[c_key].hu_rewards
            if collectors[c_key].win:
                self.win_times[c_key] += collectors[c_key].win_times
        for c_key in collectors.keys():
            if collectors[c_key].win:
                self.win_game += 1
                break

    def update_buffer(self):
        if len(self.x) != 0:
            game_no = np.array(self.game_no_list)
            x = torch.cat(self.x, dim=0)
            y = np.array(self.y)
            is_rl_agent = np.array(self.is_rl_agent)
            print(f'is_trigger_by_rl: {Counter(self.is_trigger_by_rl)}')

            p_action = np.array(self.action_probabilities)
            discard_argmax = np.array(self.discard_argmax)

            game_data = {
                'game_no': game_no,
                'states': x,
                'rewards': y,
                'actions': discard_argmax,
                'is_rl_agents': is_rl_agent,
                'p_action': p_action
            }

            for c_key in self.win_times.keys():
                print(f'Player {c_key} won {self.win_times[c_key]} times...')
            print(f'HU {sum(self.win_times.values())} times data generated...')

            self.buffer.append(game_data)
        else:
            print('No HU experience data...')

    def sample(self, n_games):
        return random.sample(self.buffer, n_games)

    def __len__(self):
        return len(self.buffer)
