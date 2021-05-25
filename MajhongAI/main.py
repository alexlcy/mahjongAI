import random
import logging
from tqdm import tqdm
from time import time
from mahjong.env import Env
from mahjong.agents.human import HumanAgent
from mahjong.agents.random import RandomAgent
from mahjong.agents.rule import RuleAgent
# import mahjong.settings
import os
from mahjong import online_encoder
from mahjong.Serialization import online_serialize
from mahjong.stats_logger.calc_functions import calc_win_rates, calc_hu_scores, calc_win_times, calc_hu_score_each_game, calc_win_rate_of_win_game_times

# mahjong.settings.init()

from mahjong.agents.DL import DeepLearningAgent
from mahjong.agents.RL import ReinforceLearningAgent
import time

import mahjong.mahjong_config as m_config
from mahjong.ReinforcementLearning.experience import ExperienceBuffer

import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import numpy as np

LOG_FORMAT = "%(message)s "
logging.basicConfig(level=logging.DEBUG, format=LOG_FORMAT)

start = time.time()
play_times = 1000
buffer = ExperienceBuffer(play_times)
random.seed(0)
# seed = time.time()
# seed = None
# config = {
#     'show_log': True,
#     'player_num': 4,
#     'seed': None  # to None for random run, if seed == None, will not save record
# }
# env = Env(config)
# env.set_agents([ReinforceLearningAgent(0), RuleAgent(1), RuleAgent(2), RuleAgent(3)])

config = {
    'show_log': False,
    'player_num': 4,
    'seed': None  # to None for random run, if seed == None, will not save record
}
env = Env(config)
RL_agent = ReinforceLearningAgent(0, play_times)
# game_agents = [RL_agent, RuleAgent(1), RuleAgent(2), RuleAgent(3)]
game_agents = [DeepLearningAgent(0), RuleAgent(1), RuleAgent(2), RuleAgent(3)]
env.set_agents(game_agents)

hu_reward_statistics = {0: [], 1: [], 2: [], 3: []}
for i in range(play_times):
    print(f'No.{i + 1} Game ing~')

    """
    reset & run
    """
    env.reset()
    # RL_agent.exploration_method.decay_step = 0
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
    # win rates of win game times
    calc_win_rate_of_win_game_times(buffer.win_times, buffer.win_game, buffer.game_no)
    # print(f'reward_in_each_game:{buffer.hu_reward}')


    # TODO: checking, can delete
    reward_sum = np.sum([buffer.hu_reward[b_key] for b_key in hu_reward_statistics.keys()])
    for h_key in hu_reward_statistics.keys():
        hu_reward_statistics[h_key].append(buffer.hu_reward[h_key])

    # TODO: checking the reward sum is zero, can delete
    if reward_sum != 0:
        print(f'Cal buffer: {buffer.hu_reward}, sum: {reward_sum}')

buffer.save_experience(m_config.buffer_folder_location)
print(f'win games:{buffer.win_game}, {buffer.win_game / play_times * 100:2f}%, tie games:{play_times - buffer.win_game}, {(play_times - buffer.win_game) / play_times * 100:2f}%')
end = time.time()
print('Agents ', end=' ')
for i in range(4):
    print(f'{i}: {game_agents[i].name}  ', end=' ')
print('')
print(f'Recording: {(end - start) / 60} min played {play_times} games')

# # Online encoder
# DL_agent = 0
# all_features = online_encoder.online_encoder(0)
# print(f'{DL_agent} player current all features:')
# print(all_features)
# Model_input = online_serialize(all_features)


# Batch encoder
# import time
# timestr = time.strftime("%Y%m%d-%H%M%S")
# extension = ".txt"
# file_name = timestr + extension
# file_path = 'datasets'
# if not os.path.exists(file_path):
#     os.mkdir(file_path)
# with open(file_path+'/'+file_name, 'w+') as file:
#     for i in mahjong.settings.myList:
#         file.write(f'%s'%str(i))

"""
for show log
"""
# config["show_log"] = True

"""
for step_back,  1 for steps
"""
# env.step_back(1)
# env.run()

"""
load & run (no need to reset)
"""
# env.load("202010070351367311", 79)
# env.run()

"""
many times
"""
# env.config["seed"] = None
# for i in tqdm(range(1000)):
#     env.reset()
#     env.run()
