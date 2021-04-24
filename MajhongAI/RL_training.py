import os
import h5py
import glob
import math
from datetime import datetime
from tqdm import tqdm
from mahjong.ReinforcementLearning.experience import ExperienceBuffer
from mahjong.models.model import DiscardModel, KongModel, PongModel
import numpy as np

import torch
import torch.nn as nn

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Hyper-parameters
lr = 0.01
batch_size = 256

def get_discounted_reward(r):
    discounted_r = (r - r.mean())/(r.std() + 1e-7)
    return discounted_r

def preprocess_data(s, a, r, p):
    s = torch.tensor(s, dtype=torch.float32, device=device)
    a = torch.tensor(a).squeeze(1).argmax(-1).long().to(device)
    r = torch.tensor(r, dtype=torch.float32, device=device)
    discounted_r = get_discounted_reward(r)
    p = torch.tensor(p, dtype=torch.float32, device=device)
    # index = torch.tensor(index, dtype=torch.float32, device=device)
    return s, a, discounted_r, p

# Training
exp_paths = glob.glob('mahjong/data/*.h5')
exp_buffer = ExperienceBuffer(10)
model = DiscardModel(device).model
loss_fn = nn.CrossEntropyLoss(reduction='none')
optim = torch.optim.Adam(model.parameters(), lr=lr)

for exp_i, exp in enumerate(exp_paths):
    _, states, rewards, actions, whether_RL, prob_from_action_model, index_of_action = exp_buffer.read_experience(exp)
    # Get Rl agent data
    states_, rewards_, actions_, prob_from_action_model_, index_of_action_ = [],[],[],[],[]
    for idx,val in enumerate(whether_RL):
        if val == 1:
            states_.append(states[idx])
            rewards_.append(rewards[idx])
            actions_.append(actions[idx])
            prob_from_action_model_.append(prob_from_action_model[idx])
            index_of_action_.append(index_of_action[idx])
    states, rewards, actions, prob_from_action_model, index_of_action = \
        states_, rewards_, actions_, prob_from_action_model_, index_of_action_

    for i in tqdm(range(math.ceil(len(states)/batch_size)), desc=f"Training on buffer {exp_i}: "):
        batch_s, batch_a, batch_r, batch_p, batch_i = states[i*batch_size:(i+1)*batch_size],\
                                                      actions[i*batch_size:(i+1)*batch_size],\
                                                      rewards[i*batch_size:(i+1)*batch_size],\
                                                      prob_from_action_model[i*batch_size:(i+1)*batch_size],\
                                                      index_of_action[i*batch_size:(i+1)*batch_size]
        batch_s, batch_a, batch_r, batch_p = preprocess_data(batch_s, batch_a, batch_r, batch_p)
        action_logits = model(batch_s)

        softmax = torch.nn.Softmax(dim=1)
        softmax_prediction = softmax(action_logits)
        prob_from_training_model = torch.tensor([softmax_prediction[j][v] for j,v in enumerate(batch_i)],
                                                dtype=torch.float32, device=device)

        optim.zero_grad()
        loss = (prob_from_training_model / batch_p) * batch_r * loss_fn(action_logits, batch_a)
        loss = loss.mean()
        loss.backward()
        optim.step()


RL_SAVE_DIR = 'mahjong/models/weights/discard/'
if not os.path.exists(RL_SAVE_DIR):
    os.makedirs(RL_SAVE_DIR)

model_name = f'RL-discard-{datetime.now().strftime("%Y-%m-%d-%H%M")}.pth'
torch.save(model.state_dict(), os.path.join(RL_SAVE_DIR, model_name))