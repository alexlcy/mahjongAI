import os
import h5py
import glob
import math
from datetime import datetime
from tqdm import tqdm
from mahjong.ReinforcementLearning.experience import ExperienceBuffer
from mahjong.models.model import DiscardModel, KongModel, PongModel

import torch
import torch.nn as nn

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Hyper-parameters
lr = 0.01
batch_size = 256

def get_discounted_reward(r):
    discounted_r = (r - r.mean())/(r.std() + 1e-7)
    return discounted_r

def preprocess_data(s, a, r):
    s = torch.tensor(s, dtype=torch.float32, device=device)
    a = torch.tensor(a).squeeze(1).argmax(-1).long().to(device)
    r = torch.tensor(r, dtype=torch.float32, device=device)
    discounted_r = get_discounted_reward(r)
    return s, a, discounted_r

# Training
exp_paths = glob.glob('mahjong/data/*.h5')
exp_buffer = ExperienceBuffer()
model = DiscardModel(device).model
loss_fn = nn.CrossEntropyLoss(reduction='none')
optim = torch.optim.Adam(model.parameters(), lr=lr)

for exp in exp_paths:
    states, rewards, actions = exp_buffer.read_experience(exp)
    for i in tqdm(math.ceil(len(states)/batch_size), desc=f"Batch {i+1}: "):
        batch_s, batch_a, batch_r = states[i*batch_size:(i+1)*batch_size], \
        actions[i*batch_size:(i+1)*batch_size], rewards[i*batch_size:(i+1)*batch_size]
        batch_s, batch_a, batch_r = preprocess_data(batch_s, batch_a, batch_r)
        action_logits = model(batch_s)
    
        optim.zero_grad()
        loss = rewards * loss_fn(action_logits, batch_a)
        loss = loss.mean()
        loss.backward()
        optim.step()


RL_SAVE_DIR = 'mahjong/models/weights/discard/'
if not os.path.exists(RL_SAVE_DIR):
    os.makedirs(RL_SAVE_DIR)

model_name = f'RL-discard-{datetime.now().strftime("%Y-%m-%d-%H%M")}.pth'
torch.save(model.state_dict(), os.path.join(RL_SAVE_DIR, model_name))