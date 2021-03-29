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

__all__ = [
    'ExperienceCollector',
    'ExperienceBuffer'
]


class ExperienceCollector:
    def __init__(self):
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
        self.win = False

    # def record_feature_reward(self, q_dict):

    def record_decision(self, action_num, raw_state, state, discard, open_meld, steal, action, reward, score, lack_color, feature_tracer):
        if action[0] == 'HU':
            self.action_nums.append(deepcopy(action_num))
            self.states.append(deepcopy(raw_state))
            self.raw_states.append(deepcopy(state))
            self.discards.append(deepcopy(discard))
            self.open_melds.append(deepcopy(open_meld))
            self.steals.append(deepcopy(steal))
            self.actions.append(deepcopy(action))
            self.scores.append(deepcopy(score))
            self.lack_colors.append(deepcopy(lack_color))
            self.feature_tracers.append(deepcopy(feature_tracer))
            self.rewards.append(deepcopy(reward))
            for i in range(len(self.rewards)-1):
                self.rewards[i] += self.rewards[-1]
            self.win = True

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


class ExperienceBuffer:
    def __init__(self):
        keys = ['player_ids', 'lack_color', 'action_nums', 'raw_states', 'states', 'discards',
                'open_melds', 'steals', 'actions', 'rewards', 'scores']
        self.buffer = {key: [] for key in keys}
        self.x = []
        self.y = []

    def massage_experience(self, collectors):
        for c_key in collectors.keys():
            if collectors[c_key].win:
                for i in range(len(collectors[c_key].feature_tracers)):
                    self.x.append(collectors[c_key].feature_tracers[i].get_features(c_key))
                self.y.extend(collectors[c_key].rewards)

    def save_experience(self, folder_path):
        if len(self.x) != 0:
            x = torch.cat(self.x, dim=3)
            y = np.array(self.y)
            date_string = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
            with h5py.File("experiment_" + date_string + r'.h5', 'w') as experience_outf:
                experience_outf.create_group('experience')
                experience_outf['experience'].create_dataset('x', data=x)
                experience_outf['experience'].create_dataset('y', data=y)
        else:
            print('No HU experience data...')

    def read_experience(self, file_name):
        h5file = h5py.File(file_name, 'r')
        self.x = np.array(h5file['experience']['x'])
        self.y = np.array(h5file['experience']['y'])
        return self.x, self.y

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
        self.buffer = pd.read_csv(folder_path + '/' + csv_file_name, sep='|')
        return self.buffer
