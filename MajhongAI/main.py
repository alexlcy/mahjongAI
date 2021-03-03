import random
import logging
from tqdm import tqdm
from time import time
from mahjong.env import Env
from mahjong.agents.human import HumanAgent
# from mahjong.agents.random import RandomAgent
from mahjong.agents.rule import RuleAgent

from MajhongAI.mahjong.agents.DL import DeepLearningAgent

LOG_FORMAT = "%(message)s"
logging.basicConfig(level=logging.DEBUG, format=LOG_FORMAT)

random.seed(0)
seed = time()
config = {
    'show_log': True,
    'show_play': False,
    'player_num': 4,
    'seed': seed # to None for random run, if seed == None, will not save record
}

env = Env(config)
env.set_agents([DeepLearningAgent(0), RuleAgent(1), RuleAgent(2), RuleAgent(3)])

"""
reset & run
"""
env.reset()
env.run()

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