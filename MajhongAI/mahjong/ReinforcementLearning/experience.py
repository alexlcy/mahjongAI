# -*- coding: utf-8 -*-
# @FileName : experience.py
# @Project  : MAHJONG AI
# @Author   : Koning LIU
# @Time     : 2021/3/22 17:12

import pandas as pd
from copy import deepcopy

__all__ = [
    'ExperienceCollector',
    'ExperienceBuffer'
]


class ExperienceCollector:
    def __init__(self):
        self.action_nums = []
        self.raw_states = []
        self.states = []
        self.discards = []
        self.open_melds = []
        self.steals = []
        self.actions = []
        self.rewards = []
        self.scores = []
        self.lack_color = None

    def record_decision(self, action_num, raw_state, state, discard, open_meld, steal, action, reward, score):
        self.action_nums.append(deepcopy(action_num))
        self.states.append(deepcopy(raw_state))
        self.raw_states.append(deepcopy(state))
        self.discards.append(deepcopy(discard))
        self.open_melds.append(deepcopy(open_meld))
        self.steals.append(deepcopy(steal))
        self.actions.append(deepcopy(action))
        self.rewards.append(deepcopy(reward))
        self.scores.append(deepcopy(score))


class ExperienceBuffer:
    def __init__(self):
        keys = ['player_ids', 'lack_color', 'action_nums', 'raw_states', 'states', 'discards',
                'open_melds', 'steals', 'actions', 'rewards', 'scores']
        self.buffer = {key: [] for key in keys}

    def combine_experience(self, collectors):
        for c_key in collectors.keys():
            self.buffer['action_nums'].extend(collectors[c_key].action_nums)
            self.buffer['raw_states'].extend(collectors[c_key].raw_states)
            self.buffer['states'].extend(collectors[c_key].states)
            self.buffer['discards'].extend(collectors[c_key].discards)
            self.buffer['open_melds'].extend(collectors[c_key].open_melds)
            self.buffer['steals'].extend(collectors[c_key].steals)
            self.buffer['actions'].extend(collectors[c_key].actions)
            self.buffer['rewards'].extend(collectors[c_key].rewards)
            self.buffer['scores'].extend(collectors[c_key].scores)
            self.buffer['lack_color'].extend([collectors[c_key].lack_color] * len(collectors[c_key].action_nums))
            self.buffer['player_ids'].extend([c_key] * len(collectors[c_key].action_nums))

    def store_experience(self, folder_path, csv_file_name):
        dataframe = pd.DataFrame(self.buffer)
        dataframe.to_csv(folder_path + '/' + csv_file_name + '.csv', index=False, sep='|')

    def load_experience(self, folder_path, csv_file_name):
        self.buffer = pd.read_csv(folder_path + '/' + csv_file_name, sep='|')
        return self.buffer
