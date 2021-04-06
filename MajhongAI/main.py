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
from mahjong.stats_logger.calc_functions import calc_win_times, calc_hu_scores

# mahjong.settings.init()

from mahjong.agents.DL import DeepLearningAgent
from mahjong.agents.RL import ReinforceLearningAgent
import time

import mahjong.mahjong_config as m_config
from mahjong.ReinforcementLearning.experience import ExperienceBuffer

import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"


LOG_FORMAT = "%(message)s"
logging.basicConfig(level=logging.DEBUG, format=LOG_FORMAT)

start = time.time()
play_times = 100
buffer = ExperienceBuffer(play_times)
random.seed(0)
# seed = time.time()
seed = None
config = {
    'show_log': True,
    'player_num': 4,
    'seed': None  # to None for random run, if seed == None, will not save record
}
env = Env(config)
env.set_agents([RuleAgent(0), RuleAgent(1), RuleAgent(2), ReinforceLearningAgent(3)])

for i in range(play_times):
    """
    reset & run
    """
    env.reset()
    buffer = env.run(buffer)

    # tensor board
    # win_times
    calc_win_times(buffer.win_times, buffer.game_no)
    # hu_score
    calc_hu_scores(buffer.hu_score, buffer.game_no)

buffer.save_experience(m_config.buffer_folder_location)
end = time.time()
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