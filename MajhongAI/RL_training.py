import os
import glob
import math
import logging
import time
import random
from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn

from mahjong.env import Env
from mahjong.ReinforcementLearning.experience import ReplayBuffer
from mahjong.models.model import DiscardModel, KongModel, PongModel
from mahjong.stats_logger.calc_functions import calc_win_rates, calc_hu_scores, calc_win_times, calc_hu_score_each_game
from mahjong.agents.DL import DeepLearningAgent
from mahjong.agents.RL import ReinforceLearningAgent
from mahjong.agents.rule import RuleAgent

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def get_discounted_reward(r):
    discounted_r = (r - r.mean())/(r.std() + 1e-7)
    return discounted_r

def preprocess_data(s, a, r, p):
    s = torch.tensor(s, dtype=torch.float32, device=device)
    a = torch.tensor(a).long().to(device)
    r = torch.tensor(r, dtype=torch.float32, device=device)
    # r = get_discounted_reward(r)
    p = torch.tensor(p, dtype=torch.float32, device=device)
    return s, a, r, p

def update_policy(model, optim, loss_fn, exps):
    for exp_i, exp in enumerate(exps):
        states, rewards, actions, action_probs = exp['states'], exp['rewards'],\
            exp['actions'], exp['p_action']

        for i in tqdm(range(math.ceil(len(states)/BATCH_SIZE)), desc=f"Training on buffer {exp_i+1}: "):
            batch_s, batch_r, batch_p, batch_a = states[i*BATCH_SIZE:(i+1)*BATCH_SIZE],\
                                                 rewards[i*BATCH_SIZE:(i+1)*BATCH_SIZE],\
                                                 action_probs[i*BATCH_SIZE:(i+1)*BATCH_SIZE],\
                                                 actions[i*BATCH_SIZE:(i+1)*BATCH_SIZE]

            batch_s, batch_a, batch_r, batch_p = preprocess_data(batch_s, batch_a, batch_r, batch_p)
            action_logits = model(batch_s)
            softmax_preds = nn.Softmax(dim=-1)(action_logits)
            pred_probs = torch.gather(softmax_preds, 1, batch_a.unsqueeze(-1)).detach()  # [bs, 1]

            optim.zero_grad()
            loss = (pred_probs.squeeze(-1) / batch_p + 1e-6) * batch_r * loss_fn(action_logits, batch_a)
            loss = loss.mean()
            loss.backward()
            optim.step()

def replace_behavior_policy(agent, model, model_type='discard'):
    if model_type == 'discard':
        agent.discard_model.model.load_state_dict(model.state_dict())
    elif model_type == 'pong':
        agent.pong_model.model.load_state_dict(model.state_dict())
    elif model_type == 'kong':
        agent.kong_model.model.load_state_dict(model.state_dict())

    print('\n=============== Behavior policy is updated ===============\n')



# =========================== Training ===========================
# Hyper-parameters & settings
PLAY_TIMES = 10000
LR = 0.00001
BATCH_SIZE = 512
EXP_SAMPLE_SIZE = 100  # how many games to sample to train model each time
BEHAVIOR_POLICY_UPDATE_INTV = 500  # interval after which the behavior policy gets replaced by the newest target policy
SAVE_INTV = 1000
TRAIN_FREQENCY = 100
MODEL_TO_TRAIN = 'discard'

model = DiscardModel(device).model
loss_fn = nn.CrossEntropyLoss(reduction='none')
optim = torch.optim.Adam(model.parameters(), lr=LR)

LOG_FORMAT = "%(message)s "
logging.basicConfig(level=logging.DEBUG, format=LOG_FORMAT)

start = time.time()
replay_buffer = ReplayBuffer(PLAY_TIMES)
random.seed(0)

config = {
    'show_log': False,
    'player_num': 4,
    'seed': None  # to None for random run, if seed == None, will not save record
}
env = Env(config)
RL_agent0 = ReinforceLearningAgent(0,PLAY_TIMES)
RL_agent1 = ReinforceLearningAgent(1,PLAY_TIMES)
RL_agent2 = ReinforceLearningAgent(2,PLAY_TIMES)
RL_agent3 = ReinforceLearningAgent(3,PLAY_TIMES)

env.set_agents([RL_agent0, RL_agent1, RL_agent2, RL_agent3])

hu_reward_statistics = {0: [], 1: [], 2: [], 3: []}

prev_weights = None
for i in range(PLAY_TIMES):
    print(f'No.{i + 1} Game ing~')

    """
    reset & run
    """
    env.reset()
    # RL_agent.exploration_method.decay_step = 0
    env.run(replay_buffer)

    # tensor board
    # win_times
    calc_win_times(replay_buffer.win_times, replay_buffer.game_no)
    # win_rates
    calc_win_rates(replay_buffer.win_times, replay_buffer.game_no)
    # hu_score
    calc_hu_scores(replay_buffer.hu_score, replay_buffer.game_no)
    # hu_score each game
    calc_hu_score_each_game(replay_buffer.hu_reward, replay_buffer.game_no)

    # # TODO: checking, can delete
    # reward_sum = np.sum([replay_buffer.hu_reward[b_key] for b_key in hu_reward_statistics.keys()])
    # for h_key in hu_reward_statistics.keys():
    #     hu_reward_statistics[h_key].append(replay_buffer.hu_reward[h_key])

    # # TODO: checking the reward sum is zero, can delete
    # if reward_sum != 0:
    #     print(f'Cal buffer: {replay_buffer.hu_reward}, sum: {reward_sum}')

    replay_buffer.update_buffer()

    # Update policy
    if len(replay_buffer) < EXP_SAMPLE_SIZE:
        continue

    if i != 0 and i % TRAIN_FREQENCY == 0:
        exps = replay_buffer.sample(EXP_SAMPLE_SIZE)
        update_policy(model, optim, loss_fn, exps)

    # Replace behavior policy
    if i != 0 and i % BEHAVIOR_POLICY_UPDATE_INTV == 0:
        replace_behavior_policy(RL_agent0, model, MODEL_TO_TRAIN)
        replace_behavior_policy(RL_agent1, model, MODEL_TO_TRAIN)
        replace_behavior_policy(RL_agent2, model, MODEL_TO_TRAIN)
        replace_behavior_policy(RL_agent3, model, MODEL_TO_TRAIN)


    # Save policy
    if i != 0 and i % SAVE_INTV == 0:
        RL_SAVE_DIR = 'mahjong/models/weights/discard/'
        if not os.path.exists(RL_SAVE_DIR):
            os.makedirs(RL_SAVE_DIR)

        model_name = f'RL-discard-playtime_{i}.pth'
        torch.save(model.state_dict(), os.path.join(RL_SAVE_DIR, model_name))

    

end = time.time()
print(f'Recording: {(end - start) / 60} min played {PLAY_TIMES} games')