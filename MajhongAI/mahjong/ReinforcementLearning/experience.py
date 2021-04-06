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

from mahjong.Serialization import helper

__all__ = [
    'ExperienceCollector',
    'ExperienceBuffer'
]


class ExperienceCollector:
    def __init__(self, player_id):
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

    # def record_feature_reward(self, q_dict):

    def record_decision(self, action_num, raw_state, state, discard, open_meld, steal, action, reward, score,
                        lack_color, feature_tracer):
        if action[0] == 'HU':
            r = deepcopy(reward)
            for i in range(len(self.rewards)):
                self.rewards[i] += r
            if r > 0:
                self.win = True
                self.win_times += 1

        elif action[0] == 'PLAY':
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


class ExperienceBuffer:
    def __init__(self):
        keys = ['player_ids', 'lack_color', 'action_nums', 'raw_states', 'states', 'discards',
                'open_melds', 'steals', 'actions', 'rewards', 'scores']
        self.buffer = {key: [] for key in keys}
        self.x = []
        self.y = []
        self.discard = []
        self.win_times = {0: 0, 1: 0, 2: 0, 3: 0}

    def massage_experience(self, collectors):
        for c_key in collectors.keys():
            for i in range(len(collectors[c_key].feature_tracers)):
                self.x.append(collectors[c_key].feature_tracers[i].get_features(c_key))
                self.discard.append(helper(1, [collectors[c_key].discard_cards[i]]))
            self.y.extend(collectors[c_key].rewards)
            if collectors[c_key].win:
                self.win_times[c_key] += collectors[c_key].win_times

    def save_experience(self, folder_path):
        if len(self.x) != 0:
            x = torch.cat(self.x, dim=0)
            y = np.array(self.y)
            discard = np.stack(self.discard)
            date_string = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
            with h5py.File(folder_path + "experiment_" + date_string + r'.h5', 'w') as experience_outf:
                experience_outf.create_group('experience')
                experience_outf['experience'].create_dataset('x', data=x)
                experience_outf['experience'].create_dataset('y', data=y)
                experience_outf['experience'].create_dataset('discard', data=discard)
            for c_key in self.win_times.keys():
                print(f'Player {c_key} won {self.win_times[c_key]} times...')
            print(f'HU {sum(self.win_times.values())} times data generated...')
        else:
            print('No HU experience data...')

    def read_experience(self, file_name):
        h5file = h5py.File(file_name, 'r')
        self.x = np.array(h5file['experience']['x'])
        self.y = np.array(h5file['experience']['y'])
        self.discard = np.array(h5file['experience']['discard'])
        return self.x, self.y, self.discard

    def combine_experience(self, collectors):
        # self.feature_buffer = []
        for c_key in collectors.keys():
            if collectors[c_key].win:
                self.buffer['action_nums'].extend(collectors[c_key].action_nums)
                self.buffer['raw_states'].extend(collectors[c_key].raw_states)
                self.buffer['states'].extend(collectors[c_key].states)
                self.buffer['discards'].extend(collectors[c_key].discards)
                self.buffer['open_melds'].extend(collectors[c_key].open_melds)
                self.buffer['steals'].extend(collectors[c_key].steals)
                self.buffer['actions'].extend(collectors[c_key].actions)
                self.buffer['rewards'].extend(collectors[c_key].rewards)
                self.buffer['scores'].extend(collectors[c_key].scores)
                self.buffer['lack_color'].extend(collectors[c_key].lack_colors)
                self.buffer['player_ids'].extend([c_key] * len(collectors[c_key].action_nums))

    def store_experience(self, folder_path, csv_file_name):
        dataframe = pd.DataFrame(self.buffer)
        dataframe.to_csv(folder_path + '/' + csv_file_name + '.csv', index=False, sep='|')

    def load_experience(self, folder_path, csv_file_name):
        self.buffer = pd.read_csv(folder_path + '/' + csv_file_name + '.csv', sep='|')
        return self.buffer
