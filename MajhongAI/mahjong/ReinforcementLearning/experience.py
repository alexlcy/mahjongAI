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
        self.player_ids = []
        self.raw_states = []
        self.states = []
        self.discards = []
        self.open_melds = []
        self.steals = []
        self.actions = []
        self.rewards = []
        self.scores = []
        self.lack_colors = []

    def record_decision(self, action_num, player_id, raw_state, state, discard, open_meld, steal, action, reward, score, lack_color):
        self.action_nums.append(deepcopy(action_num))
        self.player_ids.append(deepcopy(player_id))
        self.states.append(deepcopy(raw_state))
        self.raw_states.append(deepcopy(state))
        self.discards.append(deepcopy(discard))
        self.open_melds.append(deepcopy(open_meld))
        self.steals.append(deepcopy(steal))
        self.actions.append(deepcopy(action))
        self.rewards.append(deepcopy(reward))
        self.scores.append(deepcopy(score))
        self.lack_colors.append(deepcopy(lack_color))


class ExperienceBuffer:
    def __init__(self):
        keys = ['player_ids', 'lack_color', 'action_nums', 'raw_states', 'states', 'discards',
                'open_melds', 'steals', 'actions', 'rewards', 'scores', 'trace_back_reward']
        self.buffer = {key: [] for key in keys}

    def combine_experience(self, collectors):
        self.buffer['action_nums'].extend(collectors.action_nums)
        self.buffer['raw_states'].extend(collectors.raw_states)
        self.buffer['states'].extend(collectors.states)
        self.buffer['discards'].extend(collectors.discards)
        self.buffer['open_melds'].extend(collectors.open_melds)
        self.buffer['steals'].extend(collectors.steals)
        self.buffer['actions'].extend(collectors.actions)
        self.buffer['rewards'].extend(collectors.rewards)
        self.buffer['scores'].extend(collectors.scores)
        self.buffer['lack_color'].extend(collectors.lack_colors)
        self.buffer['player_ids'].extend(collectors.player_ids)

    def combine_new_reward(self, trace_back_reward):
        self.buffer['trace_back_reward'].extend(trace_back_reward)

    def store_experience(self, folder_path, csv_file_name):
        dataframe = pd.DataFrame(self.buffer)
        dataframe.to_csv(folder_path + '/' + csv_file_name + '.csv', index=False, sep='|')

    def load_experience(self, folder_path, csv_file_name):
        self.buffer = pd.read_csv(folder_path + '/' + csv_file_name, sep='|')
        return self.buffer
